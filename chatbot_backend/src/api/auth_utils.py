from __future__ import annotations

import os
from typing import Generator

from dotenv import load_dotenv
from passlib.context import CryptContext
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Load environment variables early so DB URL is available on import
load_dotenv()

# SQLAlchemy base
Base = declarative_base()

# Enforce Supabase Postgres usage for all connections
# Prefer container_env style variable, fallback to legacy name
DATABASE_URL = (
    os.environ.get("REACT_APP_SUPABASE_DB_URL")
    or os.environ.get("SUPABASE_DB_URL")
)

if not DATABASE_URL:
    # Fail fast to avoid accidental local SQLite usage
    raise RuntimeError(
        "Supabase Postgres connection string is required. "
        "Please set REACT_APP_SUPABASE_DB_URL (preferred) or SUPABASE_DB_URL in the environment, e.g., "
        "postgresql://<user>:<password>@<host>:<port>/<db>?sslmode=require"
    )

# Create engine configured for cloud Postgres (Supabase)
# pool_pre_ping helps avoid stale connections on managed services
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    # sslmode should be in the URL for Supabase
)

# Session factory; expire_on_commit=False to keep objects usable post-commit
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base):
    """ORM model representing an application user (app-managed credentials)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)

    __table_args__ = (
        UniqueConstraint("username", name="uix_username"),
        UniqueConstraint("email", name="uix_email"),
    )


class Conversation(Base):
    """ORM model representing a single conversation thread, keyed by session_id."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Text, unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    # Optional link to app-managed users; on user delete manage cascading in app logic or via policies
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Message(Base):
    """ORM model representing messages in a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("role in ('user','assistant')", name="chk_role_user_assistant"),
    )


def create_tables() -> None:
    """Create all tables in the database (idempotent)."""
    Base.metadata.create_all(bind=engine)


# PUBLIC_INTERFACE
def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# PUBLIC_INTERFACE
def hash_password(password: str) -> str:
    """Hash the provided plain password."""
    return pwd_context.hash(password)


# PUBLIC_INTERFACE
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify the provided password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


# PUBLIC_INTERFACE
def get_user_by_username(db: Session, username: str) -> User | None:
    """Fetch a user by username."""
    return db.query(User).filter(User.username == username).first()


# PUBLIC_INTERFACE
def get_user_by_email(db: Session, email: str) -> User | None:
    """Fetch a user by email address."""
    return db.query(User).filter(User.email == email).first()


# PUBLIC_INTERFACE
def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Attempt to create a new user; returns user or raises ValueError."""
    if get_user_by_username(db, username):
        raise ValueError("Username already registered")
    if get_user_by_email(db, email):
        raise ValueError("Email already registered")
    hashed_pwd = hash_password(password)
    user = User(username=username, email=email, hashed_password=hashed_pwd)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---- Conversation helpers ----

# PUBLIC_INTERFACE
def get_or_create_conversation(
    db: Session,
    session_id: str,
    user_id: int | None = None,
    title: str | None = None,
) -> Conversation:
    """Get a conversation by session_id, or create it if missing. If provided, set user_id/title when absent."""
    convo = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if convo:
        updated = False
        if user_id is not None and convo.user_id is None:
            convo.user_id = user_id
            updated = True
        if title and not convo.title:
            convo.title = title
            updated = True
        if updated:
            db.add(convo)
            db.commit()
            db.refresh(convo)
        return convo
    convo = Conversation(session_id=session_id, user_id=user_id, title=title)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


# PUBLIC_INTERFACE
def create_message(db: Session, conversation_id: int, role: str, content: str) -> Message:
    """Create a message row for the given conversation."""
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# PUBLIC_INTERFACE
def get_messages_for_session(db: Session, session_id: str, limit: int = 50) -> list[dict]:
    """Return messages for a given session_id ordered by created_at ascending."""
    from sqlalchemy import select, join

    j = join(Message, Conversation, Message.conversation_id == Conversation.id)
    stmt = (
        select(Message.id, Message.role, Message.content, Message.created_at)
        .select_from(j)
        .where(Conversation.session_id == session_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(limit)
    )
    rows = db.execute(stmt).fetchall()
    # Normalize into dicts
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# PUBLIC_INTERFACE
def get_conversations_for_user(db: Session, user_id: int, limit: int = 50) -> list[dict]:
    """Return a list of conversations for a given user, ordered by creation date descending."""
    from sqlalchemy import select

    stmt = (
        select(Conversation.id, Conversation.session_id, Conversation.title, Conversation.created_at)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).fetchall()
    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "title": r.title,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# PUBLIC_INTERFACE
def attach_session_to_user(db: Session, session_id: str, user_id: int, title: str | None = None) -> Conversation:
    """Attach a session to a user, creating the conversation if needed. Set title if provided and missing."""
    convo = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if convo:
        updated = False
        if convo.user_id is None:
            convo.user_id = user_id
            updated = True
        if title and not convo.title:
            convo.title = title
            updated = True
        if updated:
            db.add(convo)
            db.commit()
            db.refresh(convo)
        return convo
    # Create new conversation for this user
    convo = Conversation(session_id=session_id, user_id=user_id, title=title)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo
