import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.services.qr_service import ALGORITHM

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"


def _create_access_token(user: User) -> str:
    """Minimal claims only: user id, role, exp. No PHI."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user.id), "role": user.role, "type": "access", "exp": expire},
        settings.SECRET_KEY,
        algorithm=ALGORITHM,
    )


async def _create_refresh_token(user: User, db: AsyncSession) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    row = RefreshToken(id=uuid.uuid4(), user_id=user.id, revoked=False, expires_at=expires_at)
    db.add(row)
    await db.flush()

    return jwt.encode(
        {"sub": str(user.id), "jti": str(row.id), "type": "refresh", "exp": expires_at},
        settings.SECRET_KEY,
        algorithm=ALGORITHM,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user account. Defaults to the `clinician` role."""
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already registered")

    user = User(
        id=uuid.uuid4(),
        username=body.username,
        hashed_password=pwd_context.hash(body.password),
    )
    db.add(user)
    await db.flush()
    return {"id": str(user.id), "username": user.username, "role": user.role}


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate with username/password; returns an access token (~20 min)
    and a refresh token (~7 days)."""
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    if not user or not user.is_active or not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = _create_access_token(user)
    refresh_token = await _create_refresh_token(user, db)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Exchange a valid, non-revoked refresh token for a new access token."""
    try:
        payload = jwt.decode(body.refresh_token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    jti = payload.get("jti")
    try:
        token_id = uuid.UUID(jti)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await db.execute(select(RefreshToken).where(RefreshToken.id == token_id))
    token_row = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if (
        not token_row
        or token_row.revoked
        or token_row.expires_at < now
    ):
        raise HTTPException(status_code=401, detail="Refresh token revoked or expired")

    user_result = await db.execute(select(User).where(User.id == token_row.user_id))
    user = user_result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return TokenResponse(access_token=_create_access_token(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Revoke a refresh token (e.g. on sign-out)."""
    try:
        payload = jwt.decode(body.refresh_token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return  # already unusable — nothing to revoke

    jti = payload.get("jti")
    try:
        token_id = uuid.UUID(jti)
    except (TypeError, ValueError):
        return

    result = await db.execute(select(RefreshToken).where(RefreshToken.id == token_id))
    token_row = result.scalar_one_or_none()
    if token_row:
        token_row.revoked = True


@router.get("/me", summary="Get the current authenticated user's token claims")
async def read_current_user(user=Depends(get_current_user)):
    """Return the decoded access-token claims (`sub`, `role`) for the caller."""
    return {"user": user}
