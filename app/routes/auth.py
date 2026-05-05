from fastapi import APIRouter, Depends

from app.core.auth import get_current_user

router = APIRouter()


@router.get("/me")
async def read_current_user(user=Depends(get_current_user)):
    return {"user": user}

