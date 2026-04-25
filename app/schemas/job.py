from pydantic import BaseModel
from typing import Optional
from app.schemas.common import BaseResponse

class JobStatus(BaseResponse):
    task_id: str
    status: str   # pending | processing | complete | failed
    progress_pct: Optional[int] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
