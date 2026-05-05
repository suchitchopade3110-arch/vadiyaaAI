"""Clinical NER adapter."""


def extract_ner_features(text: str):
    """Extract clinical entities from OCR/report text."""
    from app.services.preprocessor import run_ner

    return run_ner(text)


def run_ner(text: str):
    """Lazy passthrough to the preprocessing NER engine."""
    from app.services.preprocessor import run_ner as _run_ner

    return _run_ner(text)


__all__ = ["extract_ner_features", "run_ner"]
