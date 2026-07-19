# ============================================================
# MEDSAM COMPLETE PIPELINE — WEEK 3 ALL TASKS
# Covers:
#   1. Full pipeline: image → mask → ROI extraction
#   2. Auto-prompt bounding box generator
#   3. DICOM multi-slice support (CT/MRI stacks)
#   4. Batch processing for multiple file uploads
#   5. Standardized output schema: segmentation_mask + metadata
# ============================================================

# ── CELL 1: Install dependencies ────────────────────────────
# !pip install git+https://github.com/facebookresearch/segment-anything.git
# !pip install pydicom SimpleITK opencv-python-headless

# ── CELL 2: Imports ─────────────────────────────────────────
import os
import glob
import json
import time
import zipfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, UnidentifiedImageError

# DICOM support
import pydicom
import SimpleITK as sitk

# MedSAM / SAM
from segment_anything import sam_model_registry, SamPredictor

# ── CELL 3: Device + Model setup ────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def load_model(checkpoint_path: str, model_type: str = "vit_b") -> SamPredictor:
    """Load SAM/MedSAM checkpoint and return a SamPredictor."""
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found: {checkpoint_path}. "
        "Please upload medsam_vit_b.pth or sam_vit_b_01ec64.pth."
    )
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device)
    predictor = SamPredictor(model)
    print(f"Model loaded: {checkpoint_path}  ({os.path.getsize(checkpoint_path)/1e6:.1f} MB)")
    return predictor

# --- Try MedSAM first, fall back to standard SAM ---
_APP_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = _APP_ROOT / "ml" / "models"
MEDSAM_CKPT  = os.getenv("MEDSAM_CKPT_PATH", str(_DEFAULT_MODEL_DIR / "medsam_vit_b.pth"))
SAM_CKPT     = os.getenv("SAM_CKPT_PATH", str(_DEFAULT_MODEL_DIR / "sam_vit_b_01ec64.pth"))

MODEL_NAME = "SAM-ViT-B"

def get_predictor():
    global MODEL_NAME
    if os.path.exists(MEDSAM_CKPT) and os.path.getsize(MEDSAM_CKPT) > 300_000_000:
        predictor = load_model(MEDSAM_CKPT)
        MODEL_NAME = "MedSAM-ViT-B"
    elif os.path.exists(SAM_CKPT):
        predictor = load_model(SAM_CKPT)
        MODEL_NAME = "SAM-ViT-B"
    else:
        print("No checkpoint found. Downloading standard SAM …")
        os.makedirs(os.path.dirname(SAM_CKPT), exist_ok=True)
        os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O {SAM_CKPT}")
        predictor = load_model(SAM_CKPT)
        MODEL_NAME = "SAM-ViT-B"
    return predictor


# ════════════════════════════════════════════════════════════
# SECTION 1 — STANDARDISED OUTPUT SCHEMA
# ════════════════════════════════════════════════════════════

@dataclass
class SegmentationResult:
    """Standardised output for every processed slice / image."""
    # ── core outputs ──
    segmentation_mask: np.ndarray           # binary H×W array
    roi_crop: Optional[np.ndarray]          # cropped region (may be None)

    # ── geometry ──
    bbox: List[int]                         # [x1, y1, x2, y2]
    contours: List[np.ndarray]              # OpenCV contour list

    # ── quality ──
    confidence: float                       # mask-area / bbox-area ratio

    # ── metadata ──
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata keys (populated automatically):
    #   source_path, slice_index, modality, image_shape,
    #   model_name, timestamp, error (if any)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (arrays replaced by shape/stats)."""
        d = {
            "segmentation_mask_shape": list(self.segmentation_mask.shape),
            "segmentation_mask_nonzero_px": int(self.segmentation_mask.sum()),
            "roi_crop_shape": list(self.roi_crop.shape) if self.roi_crop is not None else None,
            "bbox": self.bbox,
            "num_contours": len(self.contours),
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
        }
        return d


def _empty_result(metadata: dict, error: str) -> SegmentationResult:
    """Return a placeholder result when processing fails."""
    metadata["error"] = error
    return SegmentationResult(
        segmentation_mask=np.zeros((1024, 1024), dtype=np.uint8),
        roi_crop=None,
        bbox=[0, 0, 0, 0],
        contours=[],
        confidence=0.0,
        metadata=metadata,
    )


# ════════════════════════════════════════════════════════════
# SECTION 2 — CORE PIPELINE FUNCTIONS
# ════════════════════════════════════════════════════════════

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Min-max normalize to [0,1] float32 and ensure 3-channel RGB.
    Handles grayscale, 16-bit (DICOM), and colour inputs.
    """
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    return img


def generate_bbox(
    img: np.ndarray,
    min_ratio: float = 0.02,
    max_ratio: float = 0.85,
) -> List[int]:
    """
    Auto-prompt generator: returns [x1, y1, x2, y2] around the most
    prominent anatomical structure using CLAHE + adaptive threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [0, 0, img.shape[1], img.shape[0]]

    h, w = img.shape[:2]
    img_area = h * w
    valid = [c for c in contours
             if img_area * min_ratio < cv2.contourArea(c) < img_area * max_ratio]

    if not valid:
        cx, cy = w // 2, h // 2
        s = min(w, h) // 3
        return [cx - s, cy - s, cx + s, cy + s]

    largest = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    return [x, y, x + bw, y + bh]


def run_medsam(model_predictor: SamPredictor, img: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Run SAM/MedSAM inference with a bounding-box prompt."""
    model_predictor.set_image(img)
    masks, _, _ = model_predictor.predict(
        box=np.array(bbox), multimask_output=False
    )
    return masks[0]


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Extract OpenCV contours from a binary mask."""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def compute_confidence(mask: np.ndarray, bbox: List[int]) -> float:
    """Confidence = mask_area / bbox_area (clipped to [0, 1])."""
    mask_area = float(mask.sum())
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + 1e-8
    return float(np.clip(mask_area / bbox_area, 0.0, 1.0))


def extract_roi(
    original_image: np.ndarray,
    mask: np.ndarray,
    display_size: tuple = (1024, 1024),
    pad: int = 10,
) -> Optional[np.ndarray]:
    """
    Crop the ROI from the *original* (pre-resize) image, correctly
    scaling mask coordinates back to the original resolution.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    orig_h, orig_w = original_image.shape[:2]
    disp_h, disp_w = display_size
    x_min = max(0, int(xs.min() * orig_w / disp_w) - pad)
    x_max = min(orig_w, int(xs.max() * orig_w / disp_w) + pad)
    y_min = max(0, int(ys.min() * orig_h / disp_h) - pad)
    y_max = min(orig_h, int(ys.max() * orig_h / disp_h) + pad)
    return original_image[y_min:y_max, x_min:x_max]


def run_pipeline(
    image: np.ndarray,
    model_predictor: SamPredictor,
    source_path: str = "",
    slice_index: Optional[int] = None,
    modality: str = "unknown",
) -> SegmentationResult:
    """
    Full end-to-end pipeline: raw array → SegmentationResult.

      1. Normalize
      2. Auto-generate bounding box prompt
      3. MedSAM / SAM inference
      4. Extract contours
      5. Compute confidence
      6. Crop ROI from original resolution
    """
    metadata = {
        "source_path": source_path,
        "slice_index": slice_index,
        "modality": modality,
        "image_shape": list(image.shape),
        "model_name": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        norm = normalize_image(image)
        display = (norm * 255).astype(np.uint8)
        display = cv2.resize(display, (1024, 1024))

        bbox       = generate_bbox(display)
        mask       = run_medsam(model_predictor, display, bbox)
        contours   = get_contours(mask)
        confidence = compute_confidence(mask, bbox)
        roi        = extract_roi(image, mask, display_size=(1024, 1024))

        return SegmentationResult(
            segmentation_mask=mask,
            roi_crop=roi,
            bbox=bbox,
            contours=contours,
            confidence=confidence,
            metadata=metadata,
        )
    except Exception as e:
        return _empty_result(metadata, traceback.format_exc())


# ════════════════════════════════════════════════════════════
# SECTION 3 — DICOM MULTI-SLICE SUPPORT (CT / MRI STACKS)
# ════════════════════════════════════════════════════════════

def load_dicom_slice(dcm_path: str) -> np.ndarray:
    """
    Load a single DICOM file and return a normalised uint8 numpy array.
    Applies Rescale Slope/Intercept (Hounsfield Units for CT) and
    window-level normalisation if present.
    """
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)

    # Apply HU transform if available
    slope     = float(getattr(dcm, "RescaleSlope",     1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Window/level from DICOM tags (fall back to full range)
    wc = float(getattr(dcm, "WindowCenter", arr.mean()))
    ww = float(getattr(dcm, "WindowWidth",  arr.max() - arr.min() + 1))
    if hasattr(wc, "__iter__"):   # some tags store sequences
        wc = float(wc[0])
    if hasattr(ww, "__iter__"):
        ww = float(ww[0])

    lo, hi = wc - ww / 2, wc + ww / 2
    arr = np.clip(arr, lo, hi)
    arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return arr


def load_dicom_series(folder: str) -> List[np.ndarray]:
    """
    Load a folder of DICOM slices, sort by InstanceNumber / filename,
    and return a list of uint8 arrays (one per slice).
    """
    dcm_files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not dcm_files:
        dcm_files = sorted(glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True))

    # Sort by DICOM InstanceNumber if available
    def sort_key(p):
        try:
            return int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber)
        except Exception:
            return p

    dcm_files = sorted(dcm_files, key=sort_key)
    print(f"  Found {len(dcm_files)} DICOM slices in: {folder}")
    return [load_dicom_slice(f) for f in dcm_files], dcm_files


def process_dicom_series(
    folder: str,
    model_predictor: SamPredictor,
    modality: str = "CT",
    max_slices: Optional[int] = None,
    min_confidence: float = 0.05,
) -> List[SegmentationResult]:
    """
    Process an entire CT/MRI DICOM series folder.

    Parameters
    ----------
    folder          : path containing .dcm files (searched recursively)
    model_predictor : loaded SamPredictor
    modality        : "CT" | "MRI" | "PET" etc.
    max_slices      : cap number of slices (None = all)
    min_confidence  : skip results below this threshold

    Returns
    -------
    List of SegmentationResult, one per slice.
    """
    slices, paths = load_dicom_series(folder)
    if max_slices:
        slices = slices[:max_slices]
        paths  = paths[:max_slices]

    results = []
    for i, (arr, path) in enumerate(zip(slices, paths)):
        print(f"  [DICOM] slice {i+1}/{len(slices)}: {os.path.basename(path)}")
        result = run_pipeline(
            arr, model_predictor,
            source_path=path, slice_index=i, modality=modality,
        )
        result.metadata["dicom_file"] = os.path.basename(path)
        if result.confidence >= min_confidence:
            results.append(result)
        else:
            print(f"    ↳ skipped (confidence {result.confidence:.3f} < {min_confidence})")

    print(f"  DICOM series done. {len(results)}/{len(slices)} slices kept.")
    return results


def visualize_dicom_series(
    results: List[SegmentationResult],
    raw_slices: Optional[List[np.ndarray]] = None,
    max_display: int = 6,
):
    """Show a montage of up to max_display DICOM slices with overlaid masks."""
    n = min(len(results), max_display)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, res in enumerate(results[:n]):
        # Top row: display image
        img_display = (normalize_image(
            raw_slices[col] if raw_slices else res.segmentation_mask
        ) * 255).astype(np.uint8)
        img_display = cv2.resize(img_display, (256, 256))

        axes[0, col].imshow(img_display, cmap="gray")
        axes[0, col].set_title(
            f"Slice {res.metadata.get('slice_index', col)}\n"
            f"Conf: {res.confidence:.2f}", fontsize=8
        )
        axes[0, col].axis("off")

        # Bottom row: mask overlay
        mask_small = cv2.resize(res.segmentation_mask.astype(np.uint8), (256, 256))
        axes[1, col].imshow(img_display, cmap="gray")
        axes[1, col].imshow(mask_small, alpha=0.4, cmap="Blues")
        axes[1, col].axis("off")
        axes[1, col].set_title("Mask", fontsize=8)

    plt.suptitle(
        f"DICOM Series — {results[0].metadata.get('modality','?')}  "
        f"({len(results)} slices processed)", fontsize=11
    )
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
# SECTION 4 — BATCH PROCESSING (standard images + ZIP uploads)
# ════════════════════════════════════════════════════════════

SUPPORTED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

def is_valid_medical_image(img_path: str) -> tuple:
    """
    Quality gate: reject screenshots and low-resolution captures.
    Returns (is_valid: bool, reason: str).
    """
    SCREEN_RESOLUTIONS = {(1920,1080),(1366,768),(1440,900),(1280,720),(2560,1440)}
    try:
        img = Image.open(img_path)
        w, h = img.size
        if (w, h) in SCREEN_RESOLUTIONS:
            return False, "Exact screen resolution — likely UI screenshot"
        if w / h > 1.8:
            return False, f"Wide aspect ratio ({w/h:.2f}) — likely UI capture"
        if w < 200 or h < 200:
            return False, "Resolution too low"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def load_standard_image(path: str) -> np.ndarray:
    """Load JPEG/PNG/BMP/TIFF as a uint8 RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def _process_single(
    path: str,
    model_predictor: SamPredictor,
    modality: str = "unknown",
) -> SegmentationResult:
    """Worker used by batch processor."""
    valid, reason = is_valid_medical_image(path)
    if not valid:
        return _empty_result(
            {"source_path": path, "modality": modality, "model_name": MODEL_NAME,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "image_shape": []},
            error=f"Quality gate: {reason}",
        )
    try:
        img = load_standard_image(path)
        return run_pipeline(img, model_predictor, source_path=path, modality=modality)
    except (UnidentifiedImageError, Exception) as e:
        return _empty_result(
            {"source_path": path, "modality": modality, "model_name": MODEL_NAME,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "image_shape": []},
            error=traceback.format_exc(),
        )


def batch_process_images(
    image_paths: List[str],
    model_predictor: SamPredictor,
    modality: str = "unknown",
    max_workers: int = 4,
) -> List[SegmentationResult]:
    """
    Parallel batch processor for standard image files.
    Uses ThreadPoolExecutor (safe for I/O + numpy; GPU inference
    serialises automatically via the GIL on the predictor).

    Returns list of SegmentationResult in the same order as input.
    """
    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_single, p, model_predictor, modality): i
            for i, p in enumerate(image_paths)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            path = image_paths[idx]
            conf = results[idx].confidence
            err  = results[idx].metadata.get("error")
            status = f"conf={conf:.3f}" if not err else "ERROR"
            print(f"  [{idx+1}/{len(image_paths)}] {os.path.basename(path)}  {status}")

    return results


def batch_process_upload(
    upload_dir: str = "/content/",
    model_predictor: SamPredictor = None,
    modality: str = "unknown",
) -> List[SegmentationResult]:
    """
    Scan a directory (e.g. Colab /content/) for images and ZIP archives,
    auto-extract ZIPs, then batch-process everything found.
    Compatible with google.colab files.upload() output.
    """
    all_paths = []

    # Extract any ZIPs first
    for zf in glob.glob(os.path.join(upload_dir, "*.zip")):
        extract_dir = os.path.join(upload_dir, os.path.splitext(os.path.basename(zf))[0])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(extract_dir)
        print(f"  Extracted: {zf} → {extract_dir}")

    # Collect all images recursively
    for ext in SUPPORTED_IMAGE_EXT:
        all_paths.extend(glob.glob(os.path.join(upload_dir, "**", f"*{ext}"), recursive=True))

    all_paths = sorted(set(all_paths))
    print(f"\nBatch: found {len(all_paths)} images in {upload_dir}")

    if not all_paths:
        print("No images found. Please upload files and retry.")
        return []

    return batch_process_images(all_paths, model_predictor, modality=modality)


# ════════════════════════════════════════════════════════════
# SECTION 5 — VISUALISATION HELPERS
# ════════════════════════════════════════════════════════════

def show_mask(mask: np.ndarray, ax, random_color: bool = False):
    color = (np.concatenate([np.random.random(3), [0.6]])
             if random_color
             else np.array([30/255, 144/255, 255/255, 0.6]))
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def visualize_result(
    raw_image: np.ndarray,
    result: SegmentationResult,
    title: str = "",
):
    """3-panel display: original | mask overlay | ROI crop."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    display = cv2.resize(
        (normalize_image(raw_image) * 255).astype(np.uint8), (1024, 1024)
    )

    # Panel 1 — original + bbox
    axes[0].imshow(display)
    x1, y1, x2, y2 = result.bbox
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].set_title("Original + Auto BBox")
    axes[0].axis("off")

    # Panel 2 — mask overlay
    axes[1].imshow(display)
    show_mask(result.segmentation_mask, axes[1])
    axes[1].set_title(f"Segmentation Mask\nConf: {result.confidence:.3f}")
    axes[1].axis("off")

    # Panel 3 — ROI
    if result.roi_crop is not None:
        axes[2].imshow(result.roi_crop)
        axes[2].set_title("Extracted ROI")
    else:
        axes[2].text(0.5, 0.5, "ROI unavailable",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[2].axis("off")

    suptitle = title or result.metadata.get("source_path", "")
    plt.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


def visualize_batch_results(
    image_paths: List[str],
    results: List[SegmentationResult],
    max_display: int = 9,
):
    """Grid display for a batch of results."""
    show_n = min(len(results), max_display)
    cols = 3
    rows = (show_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i in range(show_n):
        res  = results[i]
        path = image_paths[i]
        ax   = axes[i]

        if res.metadata.get("error"):
            ax.text(0.5, 0.5, "Error / Skipped",
                    ha="center", va="center", transform=ax.transAxes, color="red")
        else:
            try:
                img = cv2.resize(
                    (normalize_image(load_standard_image(path)) * 255).astype(np.uint8),
                    (256, 256)
                )
                ax.imshow(img)
                mask_small = cv2.resize(res.segmentation_mask.astype(np.uint8), (256, 256))
                ax.imshow(mask_small, alpha=0.4, cmap="Blues")
            except Exception:
                ax.text(0.5, 0.5, "Display error",
                        ha="center", va="center", transform=ax.transAxes)

        ax.set_title(
            f"{os.path.basename(path)[:20]}\nConf: {res.confidence:.3f}", fontsize=8
        )
        ax.axis("off")

    for j in range(show_n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS EXPORT
# ════════════════════════════════════════════════════════════

def export_results(
    results: List[SegmentationResult],
    output_dir: str = "/content/medsam_output",
    save_masks: bool = True,
    save_json: bool = True,
):
    """
    Save segmentation masks as PNG + a summary JSON manifest.

    output_dir/
      masks/
        000_mask.png
        001_mask.png
        ...
      manifest.json
    """
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    manifest = []

    for i, res in enumerate(results):
        entry = res.to_dict()
        entry["index"] = i

        if save_masks and not res.metadata.get("error"):
            mask_path = os.path.join(output_dir, "masks", f"{i:04d}_mask.png")
            cv2.imwrite(mask_path, (res.segmentation_mask * 255).astype(np.uint8))
            entry["mask_file"] = mask_path

            if res.roi_crop is not None:
                roi_path = os.path.join(output_dir, "masks", f"{i:04d}_roi.png")
                roi_bgr = cv2.cvtColor(res.roi_crop, cv2.COLOR_RGB2BGR) if res.roi_crop.ndim == 3 else res.roi_crop
                cv2.imwrite(roi_path, roi_bgr)
                entry["roi_file"] = roi_path

        manifest.append(entry)

    if save_json:
        json_path = os.path.join(output_dir, "manifest.json")
        with open(json_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved: {json_path}")

    print(f"Export complete → {output_dir}  ({len(results)} results)")
    return manifest


# ════════════════════════════════════════════════════════════
# SECTION 7 — MAIN RUNNER (all sources in one call)
# ════════════════════════════════════════════════════════════

def process_all(
    image_dir: Optional[str]  = None,   # folder of JPEG/PNG/etc
    dicom_dir: Optional[str]  = None,   # folder of .dcm files
    upload_dir: Optional[str] = None,   # Colab upload directory
    image_paths: Optional[List[str]] = None,  # explicit list of paths
    output_dir: str = "/content/medsam_output",
    modality: str = "unknown",
    max_dicom_slices: Optional[int] = None,
    min_confidence: float = 0.05,
    max_workers: int = 4,
    visualize: bool = True,
    export: bool = True,
    predictor: Optional[SamPredictor] = None,
) -> Dict[str, List[SegmentationResult]]:
    """
    Master function — handles ALL input types:
      • image_dir  : directory of standard images
      • dicom_dir  : DICOM series folder
      • upload_dir : Colab/Kaggle upload folder (auto-extracts ZIPs)
      • image_paths: explicit list of file paths

    Returns dict:
      {
        "standard": [...],   # SegmentationResult list from images
        "dicom":    [...],   # SegmentationResult list from DICOM
      }
    """
    if predictor is None:
        predictor = get_predictor()

    all_results: Dict[str, List[SegmentationResult]] = {
        "standard": [],
        "dicom":    [],
    }

    # ── Standard images from a directory ──
    if image_dir and os.path.isdir(image_dir):
        paths = []
        for ext in SUPPORTED_IMAGE_EXT:
            paths.extend(glob.glob(os.path.join(image_dir, "**", f"*{ext}"), recursive=True))
        if paths:
            print(f"\n[Standard] Processing {len(paths)} images from: {image_dir}")
            all_results["standard"] += batch_process_images(
                sorted(set(paths)), predictor, modality=modality, max_workers=max_workers
            )

    # ── Explicit path list ──
    if image_paths:
        print(f"\n[Standard] Processing {len(image_paths)} explicit image paths")
        all_results["standard"] += batch_process_images(
            image_paths, predictor, modality=modality, max_workers=max_workers
        )

    # ── Upload directory (Colab / Kaggle) ──
    if upload_dir and os.path.isdir(upload_dir):
        print(f"\n[Upload] Scanning upload dir: {upload_dir}")
        all_results["standard"] += batch_process_upload(
            upload_dir, predictor, modality=modality
        )

    # ── DICOM series ──
    if dicom_dir and os.path.isdir(dicom_dir):
        print(f"\n[DICOM] Processing series: {dicom_dir}")
        dicom_results = process_dicom_series(
            dicom_dir, predictor,
            modality=modality,
            max_slices=max_dicom_slices,
            min_confidence=min_confidence,
        )
        all_results["dicom"] += dicom_results

    # ── Visualise ──
    if visualize:
        std = all_results["standard"]
        if std:
            valid_std = [(i, r) for i, r in enumerate(std) if not r.metadata.get("error")]
            if valid_std:
                print("\n── Standard image results ──")
                paths_vis = [r.metadata["source_path"] for _, r in valid_std[:9]]
                results_vis = [r for _, r in valid_std[:9]]
                visualize_batch_results(paths_vis, results_vis)

        dcm = all_results["dicom"]
        if dcm:
            print("\n── DICOM series results ──")
            visualize_dicom_series(dcm, max_display=6)

    # ── Export ──
    if export:
        combined = all_results["standard"] + all_results["dicom"]
        if combined:
            export_results(combined, output_dir=output_dir)

    # ── Summary ──
    total_std  = len(all_results["standard"])
    total_dcm  = len(all_results["dicom"])
    ok_std  = sum(1 for r in all_results["standard"] if not r.metadata.get("error"))
    ok_dcm  = sum(1 for r in all_results["dicom"]    if not r.metadata.get("error"))

    print(f"""
╔══════════════════════════════════════════╗
║        process_all() — SUMMARY           ║
╠══════════════════════════════════════════╣
║  Standard images : {ok_std:>4} / {total_std:<4} processed     ║
║  DICOM slices    : {ok_dcm:>4} / {total_dcm:<4} processed     ║
║  Output dir      : {output_dir:<26}║
╚══════════════════════════════════════════╝
""")
    return all_results


# ════════════════════════════════════════════════════════════
# SECTION 8 — USAGE EXAMPLES
# ════════════════════════════════════════════════════════════

# ── Example A: Kaggle chest X-ray dataset ───────────────────
# import kagglehub
# xray_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
# results = process_all(
#     image_dir=f"{xray_path}/chest_xray/test/NORMAL",
#     modality="X-Ray",
#     output_dir="/content/xray_output",
# )

# ── Example B: Colab file upload ────────────────────────────
# from google.colab import files
# files.upload()   # upload images or a .zip
# results = process_all(upload_dir="/content", modality="X-Ray")

# ── Example C: DICOM CT series ──────────────────────────────
# results = process_all(
#     dicom_dir="/content/ct_series",
#     modality="CT",
#     max_dicom_slices=50,
#     output_dir="/content/ct_output",
# )

# ── Example D: All sources together ─────────────────────────
# results = process_all(
#     image_dir  = "/content/xrays",
#     dicom_dir  = "/content/ct_series",
#     upload_dir = "/content",
#     modality   = "CT",
#     output_dir = "/content/all_output",
# )

# ── Example E: Inspect a single result ──────────────────────
# r = results["standard"][0]
# print(json.dumps(r.to_dict(), indent=2))
# visualize_result(load_standard_image(r.metadata["source_path"]), r)
