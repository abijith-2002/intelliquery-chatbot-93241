from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import os

# Setup SQLAlchemy base and session
Base = declarative_base()
DATABASE_URL = os.environ.get("CHATBOT_SQLALCHEMY_DATABASE_URL", "sqlite:///./chatbot_users.db")

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
    __table_args__ = (UniqueConstraint('username', name='uix_username'),
                      UniqueConstraint('email', name='uix_email'))

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
