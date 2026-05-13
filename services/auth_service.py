"""
Auth Service  —  JWT + bcrypt (direct, no passlib)
====================================================
Handles password hashing and JWT token creation / verification.
Uses bcrypt directly to avoid passlib version compatibility issues.
"""
from __future__ import annotations

import datetime
import os
from typing import Optional

import bcrypt
import jwt

# Use a strong secret from .env in production; fallback for dev only
_SECRET_KEY  = os.getenv("SECRET_KEY", "deutsch-buddy-dev-secret-2026")
_ALGORITHM   = "HS256"
_EXPIRE_DAYS = 30


def hash_password(plain: str) -> str:
    """Return bcrypt hash of a plain-text password."""
    hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: int, email: str) -> str:
    """Create a signed JWT valid for 30 days."""
    payload = {
        "sub":   str(user_id),
        "email": email,
        "exp":   datetime.datetime.utcnow() + datetime.timedelta(days=_EXPIRE_DAYS),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT.
    Returns the payload dict or None if invalid / expired.
    """
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except Exception:
        return None
