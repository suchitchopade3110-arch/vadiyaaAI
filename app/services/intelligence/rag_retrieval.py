from app.services.rag_pipeline import rag_pipeline

MIN_SOURCES = 2

def retrieve_evidence(query: str, top_k: int = 5) -> dict:
    """
    BioGPT encodes -> ChromaDB retrieves -> returns evidence dict.
    """
    results = rag_pipeline.retrieve_evidence(query, top_k=top_k)
    return {
        "results": results,
        "count": len(results),
        "sufficient": len(results) >= MIN_SOURCES
    }
