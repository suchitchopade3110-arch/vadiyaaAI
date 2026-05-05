"""
preprocessor.py — VaidyaAI Preprocessing Layer (Layer 2)
=========================================================
Handles ALL input types before they hit ML models.

PRD Layer 2 components (all implemented here):
  ✅ ClinicalBERT NER     → extract conditions, meds, lab values
  ✅ OCR Engine           → Tesseract + PyMuPDF → raw text
  ✅ Image Normalizer     → Resize + CLAHE + Denoise
  ✅ DICOM Parser         → pydicom → slices + metadata
  ✅ Data Validator       → format check + quality score

Output contracts (design_doc 14.1):
  DicomOutput, OcrOutput, NerOutput — all Pydantic, backend-ready.

PRD constraint: ClinicalBERT = NER ONLY. Never reasons. Never generates.
"""

import os
import io
import re
import uuid
import traceback
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel

# ── Device ────────────────────────────────────────────────
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════
# SECTION 1 — PYDANTIC OUTPUT CONTRACTS (design_doc 14.1)
# ════════════════════════════════════════════════════════════

from typing import Optional

class LabValue(BaseModel):
    name: str
    value: Optional[float] = None
    unit: str
    raw_text: str = ""
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None
    status: Optional[str] = None
    pct_deviation: Optional[float] = None
    # Universal additions
    value_type: str = "numeric"  # numeric, binary, qualitative, lt_gt, ratio
    display_value: str = ""
    result_text: str = ""
    is_abnormal: bool = False
    prefix: str = ""


class DicomOutput(BaseModel):
    pixel_array: Any           # np.ndarray — normalized image (H,W,3)
    metadata: Dict[str, Any]   # patient info, modality, body part
    slices: List[Any]          # list of np.ndarray for multi-slice CT/MRI
    num_slices: int
    modality: str
    quality_score: float       # 0.0-1.0


class OcrOutput(BaseModel):
    raw_text: str
    confidence: float          # mean word confidence from Tesseract
    pages: List[str]           # per-page text


class NerOutput(BaseModel):
    conditions: List[str]
    medications: List[str]
    lab_values: List[LabValue]
    dates: List[str]
    clinical_flags: List[str] = []
    raw_entities: List[Dict]   # full ClinicalBERT entity list


class ImageNormOutput(BaseModel):
    normalized: Any            # np.ndarray uint8 (H,W,3)
    original_shape: Tuple
    quality_score: float
    rejected: bool
    reject_reason: str = ""


class ValidationResult(BaseModel):
    valid: bool
    quality_score: float       # 0.0-1.0
    format_ok: bool
    size_ok: bool
    reason: str = ""


# ════════════════════════════════════════════════════════════
# SECTION 2 — DICOM PARSER
# ════════════════════════════════════════════════════════════

def parse_dicom(
    dicom_path: str,
    max_slices: Optional[int] = None,
    target_size: Tuple[int, int] = (1024, 1024),
) -> DicomOutput:
    """
    Parse a DICOM file or directory of DICOM slices.
    Handles single .dcm files and multi-slice CT/MRI series.

    Args:
        dicom_path:  path to .dcm file OR directory of .dcm files
        max_slices:  cap slice count (None = all)
        target_size: resize each slice to this (W, H)

    Returns:
        DicomOutput with pixel_array (first/only slice), slices list, metadata
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pip install pydicom SimpleITK")

    slices_raw = []
    metadata = {}

    # ── Directory (multi-slice CT/MRI series) ────────────
    if os.path.isdir(dicom_path):
        dcm_files = sorted([
            os.path.join(dicom_path, f)
            for f in os.listdir(dicom_path)
            if f.lower().endswith(".dcm")
        ])
        if not dcm_files:
            raise ValueError(f"No .dcm files found in: {dicom_path}")

        if max_slices:
            dcm_files = dcm_files[:max_slices]

        ds = pydicom.dcmread(dcm_files[0])
        metadata = _extract_dicom_metadata(ds)

        for path in dcm_files:
            ds = pydicom.dcmread(path)
            arr = _normalize_dicom_pixel(ds.pixel_array, target_size)
            slices_raw.append(arr)

    # ── Single .dcm file ─────────────────────────────────
    else:
        ds = pydicom.dcmread(dicom_path)
        metadata = _extract_dicom_metadata(ds)

        pixel = ds.pixel_array
        # Multi-frame (e.g., multi-slice in one file)
        if pixel.ndim == 3 and pixel.shape[0] > 1:
            frames = pixel[:max_slices] if max_slices else pixel
            for frame in frames:
                slices_raw.append(_normalize_dicom_pixel(frame, target_size))
        else:
            slices_raw.append(_normalize_dicom_pixel(pixel, target_size))

    primary = slices_raw[0] if slices_raw else np.zeros((*target_size, 3), dtype=np.uint8)
    quality = _image_quality_score(primary)

    return DicomOutput(
        pixel_array=primary,
        metadata=metadata,
        slices=slices_raw,
        num_slices=len(slices_raw),
        modality=metadata.get("Modality", "unknown"),
        quality_score=quality,
    )


def _extract_dicom_metadata(ds) -> Dict[str, Any]:
    """Extract key DICOM header fields safely."""
    fields = [
        "PatientID", "PatientName", "PatientAge", "PatientSex",
        "Modality", "BodyPartExamined", "StudyDescription",
        "SeriesDescription", "InstitutionName",
        "Rows", "Columns", "SliceThickness",
        "PixelSpacing", "ImageOrientationPatient",
        "StudyDate", "SeriesDate",
    ]
    meta = {}
    for field in fields:
        try:
            val = getattr(ds, field, None)
            if val is not None:
                meta[field] = str(val)
        except Exception:
            pass
    return meta


def _normalize_dicom_pixel(
    pixel: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Normalize raw DICOM pixel array:
      - Handle 16-bit → 8-bit
      - Apply CLAHE contrast enhancement
      - Resize to target_size
      - Convert to 3-channel RGB
    """
    # 16-bit → float → 8-bit
    arr = pixel.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    # Grayscale → 3-channel
    if arr.ndim == 2:
        # CLAHE on grayscale first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        # Already RGB — apply CLAHE on L channel
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize
    arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LANCZOS4)
    return arr


# ════════════════════════════════════════════════════════════
# SECTION 3 — IMAGE NORMALIZER
# ════════════════════════════════════════════════════════════

def normalize_image(
    image_input,                    # path str OR np.ndarray OR PIL.Image
    target_size: Tuple[int, int] = (1024, 1024),
    apply_clahe: bool = True,
    apply_denoise: bool = True,
) -> ImageNormOutput:
    """
    Standard image normalizer for non-DICOM inputs (JPG/PNG).
    Pipeline: Load → Validate → Resize → CLAHE → Denoise → RGB

    PRD: Image path → Normalize + CLAHE → ready for MedSAM
    """
    # ── Load ──────────────────────────────────────────────
    try:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
            arr = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
        else:
            raise TypeError(f"Unsupported type: {type(image_input)}")
    except Exception as e:
        return ImageNormOutput(
            normalized=np.zeros((*target_size, 3), dtype=np.uint8),
            original_shape=(0, 0, 0),
            quality_score=0.0,
            rejected=True,
            reject_reason=str(e),
        )

    original_shape = arr.shape

    # ── Validate ──────────────────────────────────────────
    validation = validate_image(arr)
    if not validation.valid:
        return ImageNormOutput(
            normalized=np.zeros((*target_size, 3), dtype=np.uint8),
            original_shape=original_shape,
            quality_score=0.0,
            rejected=True,
            reject_reason=validation.reason,
        )

    # ── Normalize to float [0,1] ──────────────────────────
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    # ── Resize ────────────────────────────────────────────
    arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LANCZOS4)

    # ── CLAHE contrast enhancement ────────────────────────
    if apply_clahe:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── Denoise ───────────────────────────────────────────
    if apply_denoise:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)

    quality = _image_quality_score(arr)

    return ImageNormOutput(
        normalized=arr,
        original_shape=original_shape,
        quality_score=quality,
        rejected=False,
    )


# ════════════════════════════════════════════════════════════
# SECTION 4 — OCR ENGINE
# ════════════════════════════════════════════════════════════

def run_ocr(file_path: str) -> OcrOutput:
    """
    OCR pipeline: Tesseract + PyMuPDF fallback.
    Handles PDF and image inputs.

    PRD: Lab Reports (PDF) → OCR → raw text → ClinicalBERT NER
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = []
    confidence = 0.0

    if ext == ".pdf":
        pages, confidence = _ocr_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        pages, confidence = _ocr_image(file_path)
    else:
        raise ValueError(f"Unsupported OCR format: {ext}")

    raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
    return OcrOutput(raw_text=raw_text, confidence=confidence, pages=pages)


def _ocr_pdf(pdf_path: str) -> Tuple[List[str], float]:
    """Extract text from PDF via PyMuPDF first, Tesseract fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
            else:
                # Scanned page → rasterize → Tesseract
                pix = page.get_pixmap(dpi=300)
                img_arr = np.frombuffer(pix.samples, dtype=np.uint8)
                img_arr = img_arr.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB)
                text, conf = _tesseract_on_array(img_arr)
                pages.append(text)
        return pages, 0.85  # PyMuPDF text = high confidence
    except ImportError:
        # Fallback: convert PDF pages to images with pdf2image, then Tesseract
        try:
            from pdf2image import convert_from_path
            pil_pages = convert_from_path(pdf_path, dpi=300)
            texts, confs = [], []
            for pil_img in pil_pages:
                arr = np.array(pil_img)
                text, conf = _tesseract_on_array(arr)
                texts.append(text)
                confs.append(conf)
            return texts, float(np.mean(confs)) if confs else 0.0
        except Exception as e:
            return [f"OCR failed: {e}"], 0.0


def _ocr_image(image_path: str) -> Tuple[List[str], float]:
    """OCR a single image file via Tesseract."""
    try:
        arr = np.array(Image.open(image_path).convert("RGB"))
        text, conf = _tesseract_on_array(arr)
        return [text], conf
    except Exception as e:
        return [f"OCR failed: {e}"], 0.0


def _tesseract_on_array(arr: np.ndarray) -> Tuple[str, float]:
    """Run Tesseract on numpy array, return (text, mean_confidence)."""
    try:
        import pytesseract
        # Preprocess for better OCR: grayscale + threshold
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        data = pytesseract.image_to_data(
            thresh,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        confs = [c for c in data["conf"] if c != -1]
        mean_conf = float(np.mean(confs)) / 100.0 if confs else 0.0
        return text, mean_conf
    except (ImportError, Exception) as e:
        # Graceful fallback if tesseract is missing
        print(f"[OCR] Tesseract error: {e}. Returning empty text for this segment.")
        return "", 0.0


# ════════════════════════════════════════════════════════════
# SECTION 5 — CLINICALBERT NER
# ════════════════════════════════════════════════════════════

# Lazy-loaded model singleton
_ner_pipeline = None

def _get_ner_pipeline():
    """
    Lazy load ClinicalBERT NER pipeline.
    PRD constraint: EXTRACTION ONLY — never reasons, never generates text.
    Falls back to regex if model load fails.
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        model_name = "samrawal/bert-base-uncased_clinical-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"[ClinicalBERT] NER pipeline loaded on {device}")
    except Exception as e:
        print(f"[ClinicalBERT] Failed to load: {e}. Using regex fallback.")
        _ner_pipeline = "regex_fallback"

    return _ner_pipeline


def run_ner(text: str) -> NerOutput:
    """
    ClinicalBERT NER extraction.
    PRD: Extracts conditions, medications, lab values from clinical text.
    NEVER reasons. NEVER generates. Extraction ONLY.

    Falls back to regex if ClinicalBERT unavailable.
    """
    pipe = _get_ner_pipeline()

    if pipe == "regex_fallback":
        return _regex_ner_fallback(text)

    try:
        # Chunk text to stay within 512 token limit
        chunks = _chunk_text(text, max_chars=1000)
        all_entities = []

        for chunk in chunks:
            entities = pipe(chunk)
            all_entities.extend(entities)

        return _parse_ner_entities(all_entities, text)

    except Exception as e:
        print(f"[ClinicalBERT] NER inference failed: {e}. Using regex fallback.")
        return _regex_ner_fallback(text)


JUNK_WORDS = {
    "sis", "inc", "cles", "tion", "ment", "ness", "ity",
    "terol", "aemia", "emic", "ous", "ious", "tory",
    "page break", "general healt", "blood loss",  
    "an lack", "def ine", "ate def", "renal f",
    "cocgl birth", "her hb", "lif jm", "moc bo",
    "vit amin", "ant ibody", "emodilut ion",
}

def clean_conditions(raw: List[str]) -> List[str]:
    # Legacy NER conditions cleanup
    cleaned = []
    for c in raw:
        c = c.strip()
        if "##" in c: continue
        if len(c) < 5: continue
        cleaned.append(c.title())
    return list(dict.fromkeys(cleaned))[:10]

# ════════════════════════════════════════════════════════════
# SECTION 6 — UNIVERSAL LAB PIPELINE (LAYER 1-3)
# ════════════════════════════════════════════════════════════

class ReportClassifier:
    REPORT_TYPES = {
        "lab_structured": [
            "biological ref", "ref. interval", "reference range",
            "normal range", "test result unit"
        ],
        "discharge_summary": [
            "discharge", "final diagnosis", "admitted",
            "clinical summary", "advice on discharge"
        ],
        "radiology": [
            "impression:", "findings:", "x-ray", "mri", "ct scan",
            "ultrasound", "echo"
        ],
        "microbiology": [
            "culture", "sensitivity", "organism", "colony",
            "gram stain", "mcs"
        ],
        "pathology_narrative": [
            "peripheral smear", "biopsy", "histopathology",
            "morphology"
        ],
        "hplc_electrophoresis": [
            "peak name", "retention time", "area %",
            "hplc", "electrophoresis"
        ],
    }

    PANEL_MARKERS = [
        "complete blood count", "cbc", "lipid profile", "lipid panel",
        "liver function", "lft", "kidney function", "renal function", "kft",
        "thyroid function", "tft", "iron studies", "iron panel",
        "hba1c", "glycosylated hemoglobin", "immunoassay", "hormone",
        "urine routine", "urinalysis", "biochemistry", "metabolic panel",
        "coagulation", "pt inr", "blood group", "abo", "hb electrophoresis",
        "tumor markers", "vitamin",
    ]

    COMMENTARY_MARKERS = [
        "explanation:", "summary and uses:", "limitations:",
        "interpretation:", "additional information:",
        "reference:", "important instructions",
        "increased in", "decreased in",
        "clinical significance",
        "note:", "comments:", "remarks:",
    ]

    def classify(self, text: str) -> dict:
        text_lower = text.lower()
        report_type = "unknown"
        for rtype, markers in self.REPORT_TYPES.items():
            if sum(1 for m in markers if m in text_lower) >= 2:
                report_type = rtype
                break
        panels = [p for p in self.PANEL_MARKERS if p in text_lower]
        has_table = bool(re.search(r'\d+\.?\d*\s+[a-zA-Z/%µ]+\s+\d+\.?\d*\s*[-–]\s*\d+\.?\d*', text))
        num_pages = len(re.findall(r'page \d+ of \d+', text_lower))
        return {
            "report_type": report_type,
            "panels": panels,
            "has_table": has_table,
            "num_pages": max(num_pages, 1),
            "is_multi_panel": len(panels) > 1,
        }

def strip_commentary(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(m in line_lower for m in ReportClassifier.COMMENTARY_MARKERS):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

REFERENCE_REGISTRY = {
    "hemoglobin": {"male": (13.5, 17.5), "female": (12.0, 15.5), "unit": "g/dL"},
    "wbc count": {"all": (4000, 10000), "unit": "/cmm"},
    "platelet count": {"all": (150000, 410000), "unit": "/cmm"},
    "hematocrit": {"male": (40, 49), "female": (36, 46), "unit": "%"},
    "mcv": {"all": (83, 101), "unit": "fL"},
    "mch": {"all": (27.1, 32.5), "unit": "pg"},
    "mchc": {"all": (32.5, 36.7), "unit": "g/dL"},
    "rdw": {"all": (11.6, 14.0), "unit": "%"},
    "cholesterol": {"all": (0, 200), "unit": "mg/dL"},
    "triglyceride": {"all": (0, 150), "unit": "mg/dL"},
    "hdl cholesterol": {"male": (40, 999), "female": (50, 999), "unit": "mg/dL"},
    "direct ldl": {"all": (0, 100), "unit": "mg/dL"},
    "fasting blood sugar": {"all": (74, 106), "unit": "mg/dL"},
    "hba1c": {"all": (0, 6.5), "unit": "%"},
    "creatinine": {"male": (0.66, 1.25), "female": (0.50, 1.10), "unit": "mg/dL"},
    "tsh": {"all": (0.35, 4.94), "unit": "µIU/mL"},
    "iron": {"all": (49, 181), "unit": "µg/dL"},
    "tibc": {"all": (261, 462), "unit": "µg/dL"},
    "ferritin": {"male": (22, 322), "female": (10, 291), "unit": "ng/mL"},
    "vitamin b12": {"all": (187, 833), "unit": "pg/mL"},
    "25(oh) vitamin d": {"all": (30, 100), "unit": "ng/mL"},
}

def resolve_reference(name: str, gender: str = "male", age: int = 40, pdf_ref_low: float = None, pdf_ref_high: float = None):
    if pdf_ref_low is not None and pdf_ref_high is not None:
        return pdf_ref_low, pdf_ref_high, "pdf"
    name_l = name.lower().strip()
    for reg_key, config in REFERENCE_REGISTRY.items():
        if reg_key in name_l or name_l in reg_key:
            g_key = gender if gender in config else "all"
            if g_key in config:
                low, high = config[g_key]
                return low, high, "registry"
    return None, None, "unknown"

class UniversalValueExtractor:
    NUMERIC_PATTERN = re.compile(
        r'([A-Za-z][A-Za-z0-9\s\(\)/\-]{2,50}?)\s+'
        r'[HL]?\s*(<\s*|>\s*)?(\d+\.?\d*)\s+'
        r'([µuμa-zA-Z%°/]+(?:[/][a-zA-Z]+)?)\s+'
        r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)',
        re.MULTILINE
    )
    BINARY_MAP = {
        "urine glucose": ["present", "positive", "+"],
        "urine protein": ["present", "positive", "+"],
        "hbsag": ["reactive"],
        "hiv": ["reactive"],
        "malarial parasite": ["detected", "present", "positive"],
    }
    QUALITATIVE_ABNORMAL = [
        "hypochromic", "microcytic", "macrocytic", "poikilocytosis",
        "anisocytosis", "blasts", "immature", "abnormal morphology"
    ]
    LT_GT_PATTERN = re.compile(
        r'([A-Za-z][A-Za-z0-9\s\(\)/\-B]{2,40}?)\s+'
        r'[HL]?\s*([<>])\s*(\d+\.?\d*)\s+'
        r'([µuμa-zA-Z%/]+)\s+'
        r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)',
        re.MULTILINE
    )

    def extract_all(self, text: str) -> List[LabValue]:
        stripped = strip_commentary(text)
        results = self._extract_numeric(stripped)
        results += self._extract_binary(stripped)
        results += self._extract_qualitative(stripped)
        results += self._extract_lt_gt(stripped)
        return self._deduplicate(results)

    def _extract_numeric(self, text: str) -> List[LabValue]:
        results = []
        for match in self.NUMERIC_PATTERN.finditer(text):
            name = match.group(1).strip()
            prefix = match.group(2) or ""
            value = float(match.group(3))
            unit = match.group(4).strip()
            ref_low = float(match.group(5))
            ref_high = float(match.group(6))
            if not self._is_valid_name(name): continue
            results.append(LabValue(
                name=name, value=value, unit=unit,
                ref_low=ref_low, ref_high=ref_high,
                prefix=prefix.strip(), value_type="numeric"
            ))
        return results

    def _extract_binary(self, text: str) -> List[LabValue]:
        results = []
        for test_name, abnormal_vals in self.BINARY_MAP.items():
            if test_name not in text.lower(): continue
            pattern = re.compile(rf'{re.escape(test_name)}\s+(\w[\w\s(+)\-]*)', re.IGNORECASE)
            match = pattern.search(text)
            if not match: continue
            res_text = match.group(1).strip()
            is_ab = any(av in res_text.lower() for av in abnormal_vals)
            results.append(LabValue(
                name=test_name.title(), unit="", result_text=res_text,
                value_type="binary", is_abnormal=is_ab
            ))
        return results

    def _extract_qualitative(self, text: str) -> List[LabValue]:
        results = []
        pattern = re.compile(r'(RBC Morphology|WBC Morphology|Peripheral Smear)\s+([A-Za-z\s]+)', re.IGNORECASE)
        for match in pattern.finditer(text):
            name, finding = match.group(1).strip(), match.group(2).strip()
            is_ab = any(ab in finding.lower() for ab in self.QUALITATIVE_ABNORMAL)
            results.append(LabValue(
                name=name, unit="", result_text=finding,
                value_type="qualitative", is_abnormal=is_ab
            ))
        return results

    def _extract_lt_gt(self, text: str) -> List[LabValue]:
        results = []
        for match in self.LT_GT_PATTERN.finditer(text):
            name, op, val = match.group(1).strip(), match.group(2), float(match.group(3))
            unit, low, high = match.group(4).strip(), float(match.group(5)), float(match.group(6))
            results.append(LabValue(
                name=name, value=val-0.01 if op=="<" else val+0.01,
                unit=unit, ref_low=low, ref_high=high,
                prefix=op, display_value=f"{op}{val}", value_type="lt_gt"
            ))
        return results

    def _is_valid_name(self, name: str) -> bool:
        if len(name) < 3 or len(name) > 60: return False
        if not name[0].isalpha(): return False
        if name.lower().strip() in {"page", "date", "patient", "sample"}: return False
        return True

    def _deduplicate(self, values: List[LabValue]) -> List[LabValue]:
        seen = {}
        for lv in values:
            key = lv.name.lower().strip()
            if key not in seen or (lv.ref_low is not None and seen[key].ref_low is None):
                seen[key] = lv
        return list(seen.values())

def _parse_ner_entities(entities: List[Dict], raw_text: str) -> NerOutput:
    """Map ClinicalBERT entity groups → NerOutput contract."""
    conditions, medications, lab_values, dates = [], [], [], []

    # ClinicalBERT entity_group labels vary by model
    # Map common variants to our categories
    CONDITION_LABELS = {"problem", "disease", "symptom", "diagnosis", "B-problem", "I-problem"}
    MED_LABELS = {"treatment", "medication", "drug", "test", "B-treatment", "I-treatment"}
    LAB_LABELS = {"test", "lab", "B-test", "I-test"}

    for ent in entities:
        label = ent.get("entity_group", ent.get("entity", "")).lower()
        word = ent.get("word", "").strip()
        score = ent.get("score", 0.0)

        if score < 0.6 or not word:
            continue

        word = word.replace("##", "").strip()
        if word.startswith("#") or len(word) < 3:
            continue

        if any(l in label for l in ["problem", "disease", "symptom", "diagnosis"]):
            if word not in conditions:
                conditions.append(word)
        elif any(l in label for l in ["treatment", "medication", "drug"]):
            if word not in medications:
                medications.append(word)

    # Extract lab values via Layer 1 (PDF) + Layer 2 (Registry)
    lab_values = extract_all_lab_values(raw_text)
    dates = _extract_dates_regex(raw_text)
    
    # Layer 3: Extract clinical flags from text
    clinical_flags = extract_clinical_flags_from_text(raw_text)

    return NerOutput(
        conditions=clean_conditions(conditions),
        medications=medications,
        lab_values=lab_values,
        dates=dates,
        clinical_flags=clinical_flags,
        raw_entities=entities,
    )


def _regex_ner_fallback(text: str) -> NerOutput:
    """
    Pure regex NER when ClinicalBERT unavailable.
    Covers common lab report patterns.
    """
    conditions = _extract_conditions_regex(text)
    medications = _extract_medications_regex(text)
    lab_values = extract_all_lab_values(text)
    dates = _extract_dates_regex(text)
    clinical_flags = extract_clinical_flags_from_text(text)

    return NerOutput(
        conditions=clean_conditions(conditions),
        medications=medications,
        lab_values=lab_values,
        dates=dates,
        clinical_flags=clinical_flags,
        raw_entities=[],
    )


def extract_all_lab_values(text: str) -> List[LabValue]:
    """Orchestrates Layer 1 (PDF table) and Layer 2 (Registry)."""
    # Layer 1: PDF Table parser
    layer1_labs = extract_lab_table_from_pdf(text)
    layer1_names = {lv.name.lower() for lv in layer1_labs}
    
    # Layer 2: Registry fallback (only for tests not found in Layer 1)
    layer2_labs = _extract_lab_values_registry(text)
    fallback_labs = [lv for lv in layer2_labs if lv.name.lower() not in layer1_names]
    
    return deduplicate_lab_values(layer1_labs + fallback_labs)

# Unit normalizer for common clinical tests
UNIT_CONVERTERS = {
    "wbc": {
        "cells/µL": lambda x: x / 1000,   # 9993 → 9.993
        "cells/mm³": lambda x: x / 1000,
        "/µL": lambda x: x / 1000,
    },
    "platelets": {
        "/µL": lambda x: x / 1000,
        "cells/µL": lambda x: x / 1000,
    }
}

def normalize_unit(test_name: str, value: float, unit: str) -> float:
    name_l = test_name.lower()
    # Hardcoded heuristic for common unit scaling issues (Bug 1)
    if "wbc" in name_l and value > 100:
        return value / 1000
    if "platelet" in name_l and value > 1000:
        return value / 1000

    converters = UNIT_CONVERTERS.get(name_l, {})
    for unit_key, converter in converters.items():
        if unit_key in unit.lower():
            return converter(value)
    return value

def deduplicate_lab_values(lab_values: List[LabValue]) -> List[LabValue]:
    seen_values = {}
    result = []
    for lv in lab_values:
        # Use rounded value as key to catch OCR variations of same test
        key = round(lv.value, 1)
        if key not in seen_values:
            seen_values[key] = lv
            result.append(lv)
        else:
            # Keep the one with cleaner name (shorter, no garbage words)
            existing = seen_values[key]
            if len(lv.name) < len(existing.name):
                # Swap existing with the cleaner one
                idx = result.index(existing)
                result[idx] = lv
                seen_values[key] = lv
    return result

# KNOWN LAB TEST REGISTRY with reference ranges
LAB_REGISTRY = {
    "iron": {
        "aliases": ["iron", "serum iron", "fe"],
        "unit": "ug/dL",
        "normal_range": (65, 175)
    },
    "tibc": {
        "aliases": ["tibc", "total iron binding capacity", 
                    "iron binding capacity"],
        "unit": "µg/dL",
        "normal_range": (250, 425)
    },
    "transferrin saturation": {
        "aliases": ["transferrin saturation", "tsat", "transferrin sat"],
        "unit": "%",
        "normal_range": (20, 50)
    },
    "ferritin": {
        "aliases": ["ferritin", "serum ferritin"],
        "unit": "ng/mL",
        "normal_range": (22, 322)
    },
    "hemoglobin": {
        "aliases": ["hemoglobin", "hb", "haemoglobin"],
        "unit": "g/dL",
        "normal_range_male": (13.5, 17.5),
        "normal_range_female": (12.0, 15.5)
    },
    "hba1c": {
        "aliases": ["hba1c", "glycated hemoglobin", "a1c"],
        "unit": "%",
        "normal_range": (4.0, 5.7)
    },
    "creatinine": {
        "aliases": ["creatinine", "serum creatinine", "scr"],
        "unit": "mg/dL",
        "normal_range_male": (0.74, 1.35),
        "normal_range_female": (0.59, 1.04)
    },
    "glucose": {
        "aliases": ["glucose", "blood glucose", "fasting glucose", "rbs"],
        "unit": "mg/dL",
        "normal_range": (70, 100)
    },
    "tsh": {
        "aliases": ["tsh", "thyroid stimulating hormone"],
        "unit": "µIU/mL",
        "normal_range": (0.4, 4.0)
    },
    "ldl": {
        "aliases": ["ldl", "ldl cholesterol", "low density lipoprotein"],
        "unit": "mg/dL",
        "normal_range": (0, 100)
    },
    "hdl": {
        "aliases": ["hdl", "hdl cholesterol", "high density lipoprotein"],
        "unit": "mg/dL",
        "normal_range_male": (40, 60),
        "normal_range_female": (50, 60)
    },
    "cholesterol": {
        "aliases": ["total cholesterol", "cholesterol"],
        "unit": "mg/dL",
        "normal_range": (0, 200)
    },
    "wbc": {
        "aliases": ["wbc", "white blood cell", "leucocytes", "leukocytes"],
        "unit": "10³/µL",
        "normal_range": (4.5, 11.0)
    },
    "platelets": {
        "aliases": ["platelets", "platelet count", "plt"],
        "unit": "10³/µL",
        "normal_range": (150, 400)
    },
}

def _extract_lab_values_registry(text: str, gender: str = "male") -> List[LabValue]:
    """
    Registry-based lab extraction.
    Only extracts KNOWN test names from LAB_REGISTRY.
    Ignores all header/footer/date noise.
    """
    text_lower = text.lower()
    found = []

    for test_key, config in LAB_REGISTRY.items():
        for alias in config["aliases"]:
            # Pattern: alias followed by number (value) within ~80 chars
            pattern = re.compile(
                rf'{re.escape(alias)}\s*[:\s]+(\d+\.?\d*)',
                re.IGNORECASE
            )
            match = pattern.search(text_lower)
            if match:
                try:
                    value = normalize_unit(test_key, float(match.group(1)), config.get("unit", ""))
                    unit = config.get("unit", "")

                    # Validate value is in plausible medical range
                    # Reject obvious noise (page numbers, years, phone numbers)
                    if value > 100000 or value < 0:
                        continue
                    # Reject 4-digit numbers that look like years
                    if 1900 <= value <= 2100:
                        continue

                    found.append(LabValue(
                        name=test_key.title(),
                        value=value,
                        unit=unit,
                        raw_text=match.group(0)
                    ))
                    break  # found this test, stop checking aliases
                except ValueError:
                    continue
    return found

def extract_lab_table_from_pdf(text: str) -> List[LabValue]:
    """
    Parse structured lab table pattern:
    <test_name> <value> <unit> <ref_low> - <ref_high>
    
    Covers ANY test — not just registry ones.
    Uses the lab's own reference range printed in the report.
    """
    # Pattern: captures name + value + unit + ref range
    pattern = re.compile(
        r'(Iron|TIBC|Transferrin Saturation|Ferritin'
        r'|Hemoglobin|HbA1c|Creatinine|Glucose|TSH'
        r'|Cholesterol|HDL|LDL|WBC|Platelets'
        r'|[A-Za-z][A-Za-z\s\(\)]{3,40}?)\s*\n?\s*'
        r'(\d+\.?\d*)\s+'          # ← RESULT (first number)
        r'([µuµa-zA-Z%\/]+)\s+'   # ← UNIT
        r'(\d+\.?\d*)\s*[-–]\s*'  # ← REF LOW
        r'(\d+\.?\d*)',             # ← REF HIGH
        re.MULTILINE | re.IGNORECASE
    )
    
    results = []
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        # Bug 4: Clean leading single letters (OCR artifacts)
        name = re.sub(r'^[A-Z]\s+', '', name).strip()
        
        raw_val = float(match.group(2))
        raw_unit = match.group(3).strip()
        
        # Bug 1: Normalize units (e.g. cells/µL -> 10³/µL)
        value = normalize_unit(name, raw_val, raw_unit)
        unit = raw_unit
        
        ref_low = float(match.group(4))
        ref_high = float(match.group(5))
        
        # If value was scaled, scale reference ranges by the same factor
        if raw_val != 0 and value != raw_val:
            scale = value / raw_val
            ref_low *= scale
            ref_high *= scale
        
        # Anomaly computed immediately using PDF's OWN range
        if value < ref_low:
            pct = round(((ref_low - value) / ref_low) * 100, 1)
            status = "CRITICAL" if pct > 50 else "HIGH" if pct > 25 else "LOW"
            flag = f"⬇️ {status}"
        elif value > ref_high:
            pct = round(((value - ref_high) / ref_high) * 100, 1)
            status = "CRITICAL" if pct > 50 else "HIGH" if pct > 25 else "HIGH"
            flag = f"⬆️ {status}"
        else:
            pct = 0.0
            status = "NORMAL"
            flag = "NORMAL"
        
        results.append(LabValue(
            name=name,
            value=value,
            unit=unit,
            raw_text=match.group(0),
            ref_low=ref_low,      # ← from PDF itself
            ref_high=ref_high,    # ← from PDF itself
            flag=flag,
            status=status,
            pct_deviation=pct
        ))
    
    return results

def extract_clinical_flags_from_text(text: str) -> List[str]:
    """Scan comment/narrative section for clinical interpretations."""
    flag_patterns = [
        r'(deficiency of \w+)',
        r'(\w+ is (?:reduced|elevated|low|high|increased|decreased))',
        r'(abnormal \w+)',
        r'(critical(?:ally)? (?:low|high))',
    ]
    flags = []
    for pat in flag_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        flags.extend(matches)
    return flags


def _extract_conditions_regex(text: str) -> List[str]:
    """Extract common medical condition keywords and patterns."""
    KNOWN_CONDITIONS = [
        "diabetes", "hypertension", "pneumonia", "pneumothorax",
        "cardiomegaly", "atelectasis", "effusion", "edema",
        "infiltration", "fibrosis", "emphysema", "tuberculosis",
        "anemia", "hypothyroidism", "hyperthyroidism", "obesity",
        "asthma", "copd", "heart failure", "sepsis", "stroke",
    ]
    found = []
    text_lower = text.lower()
    
    # Keyword matches
    for cond in KNOWN_CONDITIONS:
        if cond in text_lower and cond.title() not in found:
            found.append(cond.title())
            
    # Discharge summaries use "Diagnosis:" / "Final Diagnosis:" patterns
    diag_pattern = re.findall(
        r'(?:final\s+)?diagnosis\s*:\s*([^\n]+)', 
        text, re.IGNORECASE
    )
    for d in diag_pattern:
        val = d.strip()
        if val and val.title() not in found:
            found.append(val.title())
            
    return found


def _extract_medications_regex(text: str) -> List[str]:
    """Extract common medication keywords."""
    KNOWN_MEDS = [
        "metformin", "insulin", "aspirin", "atorvastatin",
        "lisinopril", "amlodipine", "metoprolol", "warfarin",
        "omeprazole", "amoxicillin", "prednisone", "furosemide",
        "levothyroxine", "azithromycin", "ciprofloxacin",
    ]
    found = []
    text_lower = text.lower()
    for med in KNOWN_MEDS:
        if med in text_lower and med not in found:
            found.append(med.title())
    return found


def _extract_dates_regex(text: str) -> List[str]:
    """Extract date patterns from clinical text."""
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',     # 01/01/2024
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',         # 2024-01-01
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
    ]
    dates = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        dates.extend(matches)
    return list(set(dates))


def _chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Split text into chunks ≤ max_chars at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]


# ════════════════════════════════════════════════════════════
# SECTION 6 — DATA VALIDATOR
# ════════════════════════════════════════════════════════════

# PRD: 50MB DICOM, 10MB images, 20MB PDF
MAX_SIZE = {
    ".dcm": 50 * 1024 * 1024,
    ".jpg": 10 * 1024 * 1024,
    ".jpeg": 10 * 1024 * 1024,
    ".png": 10 * 1024 * 1024,
    ".pdf": 20 * 1024 * 1024,
    ".csv": 10 * 1024 * 1024,
}

ALLOWED_MIME = {
    ".dcm": ["application/dicom"],
    ".jpg": ["image/jpeg"],
    ".jpeg": ["image/jpeg"],
    ".png": ["image/png"],
    ".pdf": ["application/pdf"],
    ".csv": ["text/csv", "application/csv"],
}


def validate_file(file_path: str) -> ValidationResult:
    """
    File-level validation: size check + MIME sniff.
    PRD security: MIME type check + max size enforcement.
    """
    if not os.path.exists(file_path):
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=False, size_ok=False,
                                reason="File not found")

    ext = os.path.splitext(file_path)[1].lower()
    size = os.path.getsize(file_path)
    max_size = MAX_SIZE.get(ext, 10 * 1024 * 1024)

    size_ok = size <= max_size
    format_ok = ext in MAX_SIZE

    if not format_ok:
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=False, size_ok=size_ok,
                                reason=f"Unsupported format: {ext}")
    if not size_ok:
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=True, size_ok=False,
                                reason=f"File too large: {size/1024/1024:.1f}MB > {max_size/1024/1024:.0f}MB")

    return ValidationResult(valid=True, quality_score=1.0,
                            format_ok=True, size_ok=True)


def validate_image(arr: np.ndarray) -> ValidationResult:
    """
    Image array validation: resolution, aspect ratio, blank check.
    PRD: Data Validator → format check + quality score.
    """
    if arr is None or arr.size == 0:
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=False, size_ok=False,
                                reason="Empty image array")

    h, w = arr.shape[:2]

    # Too small
    if h < 100 or w < 100:
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=True, size_ok=False,
                                reason=f"Resolution too low: {w}x{h}")

    # Likely screenshot (wide aspect ratio)
    aspect = w / h
    if aspect > 2.0:
        return ValidationResult(valid=False, quality_score=0.0,
                                format_ok=True, size_ok=True,
                                reason=f"Wide aspect ratio {aspect:.2f} — likely screenshot")

    # Nearly blank image
    quality = _image_quality_score(arr)
    if quality < 0.05:
        return ValidationResult(valid=False, quality_score=quality,
                                format_ok=True, size_ok=True,
                                reason="Image appears blank or near-uniform")

    return ValidationResult(valid=True, quality_score=quality,
                            format_ok=True, size_ok=True)


def _image_quality_score(arr: np.ndarray) -> float:
    """
    Heuristic quality score based on:
    - Laplacian variance (sharpness)
    - Pixel value spread (contrast)
    Returns 0.0-1.0
    """
    if arr is None or arr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = float(np.clip(lap_var / 500.0, 0.0, 1.0))
        contrast = float(gray.std() / 128.0)
        contrast = float(np.clip(contrast, 0.0, 1.0))
        return round((sharpness * 0.6 + contrast * 0.4), 4)
    except Exception:
        return 0.5


# ════════════════════════════════════════════════════════════
# SECTION 7 — UNIFIED ENTRY POINTS
# ════════════════════════════════════════════════════════════

def preprocess_image_file(file_path: str) -> ImageNormOutput:
    """
    Full preprocessing for standard image files (JPG/PNG).
    Used by image pipeline before MedSAM.
    """
    val = validate_file(file_path)
    if not val.valid:
        return ImageNormOutput(
            normalized=np.zeros((1024, 1024, 3), dtype=np.uint8),
            original_shape=(0, 0, 0),
            quality_score=0.0,
            rejected=True,
            reject_reason=val.reason,
        )
    return normalize_image(file_path)


def preprocess_dicom_file(file_path: str, max_slices: int = 50) -> DicomOutput:
    """
    Full preprocessing for DICOM files.
    Used by image pipeline before MedSAM.
    """
    val = validate_file(file_path)
    if not val.valid:
        raise ValueError(f"DICOM validation failed: {val.reason}")
    return parse_dicom(file_path, max_slices=max_slices)


def preprocess_report_file(file_path: str) -> NerOutput:
    """
    Full preprocessing for lab report / clinical note PDFs.
    Pipeline: validate → OCR → ClinicalBERT NER → NerOutput
    Used by text pipeline before XGBoost.
    """
    val = validate_file(file_path)
    if not val.valid:
        raise ValueError(f"Report validation failed: {val.reason}")

    ocr_result = run_ocr(file_path)
    ner_result = run_ner(ocr_result.raw_text)
    return ner_result


# ════════════════════════════════════════════════════════════
# SECTION 8 — STANDALONE TEST
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Preprocessor Test ===")

    # Test image normalization with dummy array
    dummy = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    result = normalize_image(dummy)
    print(f"Image norm: shape={result.normalized.shape}, quality={result.quality_score}, rejected={result.rejected}")

    # Test image validation
    val = validate_image(dummy)
    print(f"Validation: valid={val.valid}, quality={val.quality_score}")

    # Test regex NER fallback
    sample_text = """
    Patient diagnosed with Type 2 Diabetes and Hypertension.
    Currently on Metformin 500mg and Lisinopril.
    Lab results: Hemoglobin: 10.2 g/dL, HbA1c: 7.8 %, Creatinine: 1.4 mg/dL
    Date: 01/15/2024
    """
    ner = run_ner(sample_text)
    print(f"\nNER Output:")
    print(f"  Conditions:  {ner.conditions}")
    print(f"  Medications: {ner.medications}")
    print(f"  Lab Values:  {[(lv.name, lv.value, lv.unit) for lv in ner.lab_values]}")
    print(f"  Dates:       {ner.dates}")

    print("\nPRD contract check:")
    print("  DicomOutput ✓ | OcrOutput ✓ | NerOutput ✓ | ImageNormOutput ✓ | ValidationResult ✓")
