"""Optional YOLO detector for image-analysis bounding boxes.

The detector tries a chest X-ray YOLO model first, then falls back to YOLOv8n
COCO for a broad body ROI. All imports are lazy so the core image pipeline keeps
working when optional YOLO dependencies or network-downloaded weights are absent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "data/models"))
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(MODELS_DIR / "chest_xray_yolo.pt")))
COCO_MODEL_NAME = os.getenv("YOLO_COCO_MODEL", "yolov8n.pt")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "chest-xrays-qjmia")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "chest-xrays-pneumonia")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))

CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
OUTPUT_DIR = Path(os.getenv("YOLO_OUTPUT_DIR", "data/yolo_outputs"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.getenv("YOLO_CONFIG_DIR", "/tmp"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_MAP = {
    "pneumonia": "HIGH",
    "high-pneumonia": "HIGH",
    "high_pneumonia": "HIGH",
    "low-pneumonia": "MODERATE",
    "low_pneumonia": "MODERATE",
    "opacity": "MODERATE",
    "consolidation": "MODERATE",
    "effusion": "MODERATE",
    "nodule": "MODERATE",
    "no-pneumonia": "LOW",
    "no_pneumonia": "LOW",
    "normal": "LOW",
    "person": None,
}

SEVERITY_COLORS = {
    "HIGH": (178, 24, 43),
    "MODERATE": (239, 159, 39),
    "LOW": (29, 158, 117),
}

_chest_model = None
_coco_model = None


def _load_chest_model():
    """Load a chest X-ray YOLO model if available."""
    global _chest_model
    if _chest_model is not None:
        return _chest_model

    if YOLO_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO

            _chest_model = YOLO(str(YOLO_MODEL_PATH))
            logger.info("[YOLO] Loaded chest X-ray model from %s", YOLO_MODEL_PATH)
            return _chest_model
        except Exception as exc:
            logger.warning("[YOLO] Failed to load saved chest model: %s", exc)

    if ROBOFLOW_API_KEY:
        try:
            from roboflow import Roboflow
            from ultralytics import YOLO

            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=str(MODELS_DIR / "roboflow"))
            weights = list(Path(dataset.location).rglob("*.pt"))
            if weights:
                _chest_model = YOLO(str(weights[0]))
                YOLO_MODEL_PATH.write_bytes(weights[0].read_bytes())
                logger.info("[YOLO] Downloaded and loaded Roboflow chest model")
                return _chest_model
        except Exception as exc:
            logger.warning("[YOLO] Roboflow chest model unavailable: %s", exc)

    logger.info("[YOLO] Chest X-ray model unavailable; using COCO fallback when possible")
    return None


def _load_coco_model():
    """Load YOLOv8n COCO fallback. Ultralytics may download weights on first run."""
    global _coco_model
    if _coco_model is not None:
        return _coco_model
    try:
        from ultralytics import YOLO

        _coco_model = YOLO(COCO_MODEL_NAME)
        logger.info("[YOLO] Loaded COCO fallback model %s", COCO_MODEL_NAME)
        return _coco_model
    except Exception as exc:
        logger.warning("[YOLO] COCO fallback unavailable: %s", exc)
        return None


def _run_yolo(model, image_path: str) -> list[dict[str, Any]]:
    try:
        results = model.predict(
            source=image_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("[YOLO] Inference failed: %s", exc)
        return []

    detections: list[dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        names = result.names or {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, f"class_{cls_id}")).lower()
            severity = SEVERITY_MAP.get(label)
            if severity is None:
                continue
            xyxy = box.xyxy[0].tolist()
            xywhn = box.xywhn[0].tolist() if box.xywhn is not None else []
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(box.conf[0]) * 100, 1),
                    "severity": severity,
                    "bbox_px": [round(value) for value in xyxy],
                    "bbox_norm": [round(value, 4) for value in xywhn] if xywhn else [],
                    "source": "chest_xray_yolo",
                }
            )
    return detections


def _run_coco_for_roi(model, image_path: str) -> list[dict[str, Any]]:
    try:
        results = model.predict(source=image_path, conf=0.3, classes=[0], imgsz=IMG_SIZE, verbose=False)
    except Exception as exc:
        logger.warning("[YOLO] COCO ROI detection failed: %s", exc)
        return []

    rois: list[dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            rois.append(
                {
                    "label": "body_roi",
                    "bbox_px": [round(value) for value in box.xyxy[0].tolist()],
                    "confidence": round(float(box.conf[0]) * 100, 1),
                    "source": "coco_fallback",
                }
            )
    return rois


def _annotate_image(image_path: str, detections: list[dict[str, Any]], job_id: str) -> str | None:
    try:
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for det in detections:
            bbox = det.get("bbox_px") or []
            if len(bbox) != 4:
                continue
            color = SEVERITY_COLORS.get(det.get("severity", "MODERATE"), SEVERITY_COLORS["MODERATE"])
            x1, y1, x2, y2 = bbox
            for offset in range(2):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
            label = str(det.get("label", "")).replace("_", " ").title()
            text = f"{label} {float(det.get('confidence') or 0):.0f}%"
            text_w = max(len(text) * 7, 36)
            text_h = 14
            y_label = max(y1 - text_h - 2, 0)
            draw.rectangle([x1, y_label, x1 + text_w, y_label + text_h], fill=color)
            draw.text((x1 + 3, y_label + 1), text, fill=(255, 255, 255))
        out_path = OUTPUT_DIR / f"yolo_{job_id}.jpg"
        image.save(out_path, quality=90)
        return str(out_path)
    except Exception as exc:
        logger.warning("[YOLO] Annotation failed: %s", exc)
        return None


def detect_lung_disease(image_path: str, job_id: str = "job") -> dict[str, Any]:
    """Run optional YOLO disease detection and return structured detection data."""
    result: dict[str, Any] = {
        "detections": [],
        "annotated_path": None,
        "model_used": "none",
        "roi_bbox": [],
        "primary_finding": None,
        "yolo_confidence": 0.0,
    }

    chest_model = _load_chest_model()
    if chest_model is not None:
        detections = _run_yolo(chest_model, image_path)
        if detections:
            result["detections"] = detections
            result["model_used"] = "chest_xray_yolo"

    if not result["detections"]:
        coco_model = _load_coco_model()
        if coco_model is not None:
            rois = _run_coco_for_roi(coco_model, image_path)
            result["roi_bbox"] = rois[0]["bbox_px"] if rois else []
            result["model_used"] = "coco_fallback"

    severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    result["detections"].sort(
        key=lambda item: (severity_order.get(item.get("severity", "LOW"), 2), -float(item.get("confidence") or 0))
    )
    if result["detections"]:
        top = result["detections"][0]
        result["primary_finding"] = top["label"]
        result["yolo_confidence"] = top["confidence"]

    display_detections = result["detections"]
    if not display_detections and result.get("roi_bbox") and len(result["roi_bbox"]) == 4:
        display_detections = [
            {
                "label": "body_roi",
                "confidence": 0,
                "severity": "LOW",
                "bbox_px": result["roi_bbox"],
            }
        ]
    result["annotated_path"] = _annotate_image(image_path, display_detections, job_id)

    return result


def get_yolo_gradcam_regions(yolo_result: dict[str, Any], img_w: int, img_h: int) -> list[str]:
    """Convert YOLO boxes into region strings for GradCAM/RAG context."""
    regions: list[str] = []
    for det in yolo_result.get("detections", []):
        bbox = det.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        zone = "upper" if cy < img_h * 0.33 else "middle" if cy < img_h * 0.66 else "lower"
        side = "right" if cx < img_w / 2 else "left"
        label = str(det.get("label", "")).replace("_", " ")
        regions.append(f"{side} {zone} lobe - {label}")
    return regions[:4]
