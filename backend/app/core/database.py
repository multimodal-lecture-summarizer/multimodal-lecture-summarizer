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


def initialize_database_data():
    """
    Initializes default database records (default admin and normal users, default video standards)
    if they do not already exist.
    """
    from app.models.user import User, UserRole
    from app.models.video import VideoStandard
    from app.api.deps import get_password_hash
    import uuid
    
    db = SessionLocal()
    try:
        # 1. Initialize Default Standard if empty
        std_count = db.query(VideoStandard).count()
        if std_count == 0:
            default_std = VideoStandard(
                max_duration=settings.DEFAULT_MAX_DURATION_SECONDS,
                allowed_formats=settings.DEFAULT_ALLOWED_FORMATS,
                max_file_size=settings.DEFAULT_MAX_FILE_SIZE_MB,
            )
            db.add(default_std)
            db.commit()
            logger.info("Initialized default VideoStandard record.")
            
        # 2. Initialize Default Admin User
        admin_email = "hungphitran.22@gmail.com"
        admin_exists = db.query(User).filter(User.email == admin_email).first()
        if not admin_exists:
            admin_pwd = "AdminPass123@"
            admin_user = User(
                user_id=uuid.uuid4(),
                email=admin_email,
                password_hash=get_password_hash(admin_pwd),
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            logger.info(f"Initialized default Admin account: {admin_email} / {admin_pwd}")
            
        # 3. Initialize Default Normal User
        user_email = "nguyen.van.a@gmail.com"
        user_exists = db.query(User).filter(User.email == user_email).first()
        if not user_exists:
            user_pwd = "UserPass123@"
            normal_user = User(
                user_id=uuid.uuid4(),
                email=user_email,
                password_hash=get_password_hash(user_pwd),
                role=UserRole.USER,
                is_active=True,
            )
            db.add(normal_user)
            db.commit()
            logger.info(f"Initialized default User account: {user_email} / {user_pwd}")
    except Exception as e:
        logger.error(f"Failed to initialize database defaults: {e}")
        db.rollback()
    finally:
        db.close()
