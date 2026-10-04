from datetime import datetime, timedelta, timezone
import jwt
import secrets
from fastapi import HTTPException

from app.config import (
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_SECRECT,
    JWT_REFRESH_TOKEN_SECRET,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
)


def signing_key(value: str | None) -> str:
    if not value or len(value) < 32 or value.lower().startswith(("replace", "dummy", "your_", "placeholder")):
        raise HTTPException(503, "JWT signing keys are not configured. Set separate strong server signing keys.")
    return value


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=int(JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    to_encode["jti"] = secrets.token_urlsafe(16)

    encoded_jwt = jwt.encode(
        to_encode, signing_key(JWT_ACCESS_TOKEN_SECRECT), algorithm=JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(data: dict):
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        days=int(JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire})
    to_encode["jti"] = secrets.token_urlsafe(16)

    encoded_jwt = jwt.encode(
        to_encode, signing_key(JWT_REFRESH_TOKEN_SECRET), algorithm=JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, signing_key(JWT_ACCESS_TOKEN_SECRECT), algorithms=[JWT_ALGORITHM])


def decode_refresh_token(token: str) -> dict:
    return jwt.decode(token, signing_key(JWT_REFRESH_TOKEN_SECRET), algorithms=[JWT_ALGORITHM])


def create_auth_tokens(data: dict) -> dict:

    return {
        "access_token": create_access_token(data),
        "refresh_token": create_refresh_token(data),
    }
