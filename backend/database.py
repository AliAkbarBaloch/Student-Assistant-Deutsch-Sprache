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
    # Path relative to /static/ — e.g. "avatars/3.jpg"
    avatar_url    = Column(String(255), nullable=True, default=None)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    calls    = relationship("Call", back_populates="user", cascade="all, delete-orphan")


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


class Call(Base):
    """A single Live-Anruf (LiveKit voice call) session, holding its transcript."""

    __tablename__ = "calls"

    id         = Column(String(36), primary_key=True)   # uuid4
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)

    user     = relationship("User", back_populates="calls")
    messages = relationship(
        "CallMessage", back_populates="call",
        cascade="all, delete-orphan", order_by="CallMessage.id",
    )


class CallMessage(Base):
    """Single turn (user or assistant) within a Live-Anruf transcript."""

    __tablename__ = "call_messages"

    id         = Column(Integer, primary_key=True, index=True)
    call_id    = Column(String(36), ForeignKey("calls.id"), nullable=False)
    role       = Column(String(20), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    call = relationship("Call", back_populates="messages")


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
