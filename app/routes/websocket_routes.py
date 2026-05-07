import asyncio

from celery.result import AsyncResult
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.workers.celery_app import celery_app

router = APIRouter()


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                result = AsyncResult(job_id, app=celery_app)
                state = result.state
                info = result.info if isinstance(result.info, dict) else {}
                payload = {
                    "job_id": job_id,
                    "status": state,
                    "progress": info.get("pct", 0),
                    "step": info.get("step", ""),
                }

                if state == "SUCCESS":
                    payload["result"] = result.result
                    await websocket.send_json(payload)
                    break

                if state == "FAILURE":
                    payload["error"] = str(result.result)
                    await websocket.send_json(payload)
                    break

                await websocket.send_json(payload)

            except Exception as celery_exc:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": "ERROR",
                    "progress": 0,
                    "step": "",
                    "error": str(celery_exc),
                })
                break

            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
