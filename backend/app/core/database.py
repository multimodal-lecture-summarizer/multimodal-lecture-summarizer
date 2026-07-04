import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

logger = logging.getLogger(__name__)

import os
import time

# Establish initial database engine setup.
# If postgresql is configured, we configure the postgresql engine.
# Fallback validation is deferred to runtime initialization.
db_uri = settings.sqlalchemy_database_uri
use_sqlite = False

if "postgresql" in db_uri:
    engine = create_engine(
        db_uri,
        pool_pre_ping=True,
        echo=False,
    )
else:
    use_sqlite = True
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sqlite_path = os.path.join(backend_dir, "local.db")
    db_uri = f"sqlite:///{sqlite_path}"
    engine = create_engine(
        db_uri,
        connect_args={"check_same_thread": False},
        echo=False,
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


def verify_db_connection(retries: int = 5, delay: float = 2.0):
    """
    Verifies connection to the database. If it is PostgreSQL and fails,
    it retries. If it still fails:
    - If APP_ENV == "development", it falls back to local SQLite and configures engine/SessionLocal.
    - If APP_ENV != "development" (e.g. production), it raises RuntimeError to abort startup.
    """
    global engine, SessionLocal
    db_uri = settings.sqlalchemy_database_uri
    uvicorn_logger = logging.getLogger("uvicorn.error")
    
    if "postgresql" not in db_uri:
        # SQLite is local, verify we can connect/write
        try:
            with engine.connect() as conn:
                pass
            uvicorn_logger.info("Connected to SQLite database successfully.")
            return
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SQLite database: {e}")

    # For PostgreSQL, check connection with retries
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            # Try to connect
            with engine.connect() as conn:
                pass
            uvicorn_logger.info("Connected to PostgreSQL database successfully.")
            return
        except Exception as e:
            last_error = e
            uvicorn_logger.warning(f"PostgreSQL connection attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
                
    # If we reached here, PostgreSQL connection failed after all retries
    if settings.APP_ENV == "development":
        uvicorn_logger.warning(
            f"All connection attempts to PostgreSQL failed. Falling back to local SQLite "
            f"since APP_ENV is '{settings.APP_ENV}'."
        )
        # Re-initialize engine to SQLite
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sqlite_path = os.path.join(backend_dir, "local.db")
        sqlite_uri = f"sqlite:///{sqlite_path}"
        engine = create_engine(
            sqlite_uri,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        SessionLocal.configure(bind=engine)
        # Try to connect to SQLite to verify
        try:
            with engine.connect() as conn:
                pass
            uvicorn_logger.info("Fallback SQLite engine initialized and verified successfully.")
        except Exception as e:
            raise RuntimeError(f"Fallback SQLite connection failed: {e}")
    else:
        # Production/Staging: Fail startup immediately!
        uvicorn_logger.error("PostgreSQL database connection failed and fallback is disabled in non-development environment.")
        raise RuntimeError(
            f"Failed to connect to PostgreSQL at {settings.POSTGRES_SERVER} after {retries} attempts: {last_error}"
        )
