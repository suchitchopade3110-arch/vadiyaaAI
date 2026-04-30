from types import SimpleNamespace

from app.services.clinicalbert import clinicalbert_service


def extract_entities(text: str):
    """
    ClinicalBERT NER -> structured output.

    The current implementation delegates to the repo's existing extractor and
    exposes `.features` for the skeleton prediction contract.
    """
    entities = clinicalbert_service.extract_entities(text)
    features = {
        "conditions": entities.get("conditions", []),
        "medications": entities.get("medications", []),
        "lab_values": entities.get("lab_values", {}),
        "procedures": entities.get("procedures", []),
    }
    return SimpleNamespace(**entities, features=features, as_dict=lambda: entities)

