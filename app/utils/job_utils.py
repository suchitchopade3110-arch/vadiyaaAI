def update_progress(job_id: str, progress: float, status: str = "running"):
    return {"job_id": job_id, "progress": progress, "status": status}


def mark_failed(record_id: str, job_id: str, message: str):
    return {"id": record_id, "job_id": job_id, "status": "failed", "error": message}


def save_result(record_id: str, job_id: str, result: dict):
    return {"id": record_id, "job_id": job_id, "status": "complete", "result": result}

