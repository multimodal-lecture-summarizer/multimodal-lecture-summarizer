import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

logger = logging.getLogger(__name__)

# Try to connect to PostgreSQL. If it fails, fallback to local SQLite.
db_uri = settings.sqlalchemy_database_uri
use_sqlite = False

if "postgresql" in db_uri:
    try:
        # Create a temporary engine to test connection with a short timeout
        temp_engine = create_engine(db_uri, connect_args={"connect_timeout": 3})
        with temp_engine.connect() as conn:
            pass
        temp_engine.dispose()
    except Exception as e:
        logger.warning(
            f"Failed to connect to PostgreSQL at {settings.POSTGRES_SERVER}: {e}. "
            "Falling back to local SQLite database (sqlite:///./local.db) as specified in BACKEND_GUIDE.md."
        )
        use_sqlite = True

import os

if use_sqlite or "sqlite" in db_uri:
    # Resolve absolute path to backend/local.db so both uvicorn (CWD: backend) and seed.py (CWD: root) use the same file
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sqlite_path = os.path.join(backend_dir, "local.db")
    db_uri = f"sqlite:///{sqlite_path}"
    engine = create_engine(
        db_uri,
        connect_args={"check_same_thread": False},
        echo=settings.APP_ENV == "development",
    )
else:
    engine = create_engine(
        db_uri,
        pool_pre_ping=True,
        echo=settings.APP_ENV == "development",
    )

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base model class using the modern SQLAlchemy DeclarativeBase
class Base(DeclarativeBase):
    pass


def get_db():
    """
    Dependency helper to get a database session.
    Automatically closes session after request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
