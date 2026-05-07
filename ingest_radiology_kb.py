"""CLI wrapper for app.services.ingest_radiology_kb."""

from app.services.ingest_radiology_kb import ingest_all, query_radiology_kb


if __name__ == "__main__":
    counts = ingest_all()
    print(f"Ingested {counts['radiology_patterns']} radiology patterns")
    print(f"Ingested {counts['chexnet_labels']} CheXNet labels")
    for result in query_radiology_kb("pneumonia consolidation right lower lobe", n_results=3):
        print(f"[{result['score']}] {result['pattern']} ({result['urgency']})")
