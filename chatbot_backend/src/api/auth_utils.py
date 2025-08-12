from passlib.context import CryptContext
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    UniqueConstraint,
    DateTime,
    ForeignKey,
    CheckConstraint,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import os

# Setup SQLAlchemy base and session
Base = declarative_base()

# Prefer Supabase Postgres if provided, else fallback to explicit DB var, else SQLite
DATABASE_URL = (
    os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("CHATBOT_SQLALCHEMY_DATABASE_URL")
    or "sqlite:///./chatbot_users.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User model for authentication
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)
    __table_args__ = (
        UniqueConstraint('username', name='uix_username'),
        UniqueConstraint('email', name='uix_email'),
    )

# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Text, unique=True, nullable=False, index=True)
    title = Column(Text, nullable=True)
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# Message model
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint("role in ('user','assistant')", name="chk_role_user_assistant"),
    )

def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

# PUBLIC_INTERFACE
def get_db():
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
def get_user_by_username(db: Session, username: str):
    """Fetch a user by username."""
    return db.query(User).filter(User.username == username).first()

# PUBLIC_INTERFACE
def get_user_by_email(db: Session, email: str):
    """Fetch a user by email address."""
    return db.query(User).filter(User.email == email).first()

# PUBLIC_INTERFACE
def create_user(db: Session, username: str, email: str, password: str):
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
def get_or_create_conversation(db: Session, session_id: str, user_id: int | None = None, title: str | None = None):
    """Get a conversation by session_id, or create it if missing."""
    convo = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if convo:
        return convo
    convo = Conversation(session_id=session_id, user_id=user_id, title=title)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo

# PUBLIC_INTERFACE
def create_message(db: Session, conversation_id: int, role: str, content: str):
    """Create a message row for the given conversation."""
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

# PUBLIC_INTERFACE
def get_messages_for_session(db: Session, session_id: str, limit: int = 50):
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
        {"id": r.id, "role": r.role, "content": r.content, "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]
