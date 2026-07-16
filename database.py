# SQLite database via SQLAlchemy.
# Two tables: User (accounts) and Message (chat history per user).
from __future__ import annotations

import datetime
from pathlib import Path

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

BASE_DIR     = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'deutsch_buddy.db'}"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Registered user account."""

    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    name          = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    # Path relative to /media/ — e.g. "/media/avatars/3.jpg"
    avatar_url    = Column(String(255), nullable=True, default=None)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


class Message(Base):
    """Single chat turn (user or assistant)."""

    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    role       = Column(String(20), nullable=False)   # "user" | "assistant"
    content_de = Column(Text, nullable=False)         # German text
    content_en = Column(Text, default="")             # English translation (assistant only)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="messages")


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)
