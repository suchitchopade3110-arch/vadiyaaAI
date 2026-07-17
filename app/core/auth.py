from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

from app.config import settings

security = HTTPBearer(auto_error=False)


async def get_current_user(token=Depends(security)):
    """JWT guard for production routes.

    Validates HS256 access tokens (python-jose). Refresh tokens are rejected
    here — they carry `type: refresh` and must only be redeemed via
    POST /auth/refresh.
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
        payload = jwt.decode(token.credentials, settings.SECRET_KEY, algorithms=["HS256"])
    except JWTError as exc:
        raise HTTPException(status_code=403, detail="Invalid token") from exc

    if payload.get("type") != "access":
        raise HTTPException(status_code=403, detail="Invalid token type")

    return payload


def require_role(*roles: str):
    """Dependency factory: raise 403 unless the current user's role is in `roles`.

    Usage: Depends(require_role("admin"))
    """

    async def _check_role(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return _check_role
