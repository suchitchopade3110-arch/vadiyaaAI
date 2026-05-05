from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

from app.config import settings

security = HTTPBearer(auto_error=False)


async def get_current_user(token=Depends(security)):
    """JWT guard for production routes.

    Dev note: routes can choose whether to depend on this. The helper is present
    for the skeleton contract and validates HS256 tokens when python-jose is
    installed.
    """
    if token is None:
        raise HTTPException(status_code=403, detail="Missing token")

    try:
        from jose import JWTError, jwt
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="JWT support is not installed. Add python-jose[cryptography].",
        ) from exc

    try:
        return jwt.decode(token.credentials, settings.SECRET_KEY, algorithms=["HS256"])
    except JWTError as exc:
        raise HTTPException(status_code=403, detail="Invalid token") from exc

