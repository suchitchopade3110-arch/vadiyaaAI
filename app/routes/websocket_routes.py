from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from celery.result import AsyncResult

router = APIRouter()


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            result = AsyncResult(job_id)
            progress = result.info.get("pct", 0) if isinstance(result.info, dict) else 0
            await websocket.send_json({
                "job_id": job_id,
                "status": result.state,
                "progress": progress,
            })
            if result.ready():
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except:
            pass
