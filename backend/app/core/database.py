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
    from sqlalchemy import text, inspect
    
    db = SessionLocal()
    try:
        # Self-healing migration: check if title column exists in videos and key_takeaways_json in summaries
        try:
            with db.bind.connect() as conn:
                inspector = inspect(db.bind)
                columns = [c["name"] for c in inspector.get_columns("videos")]
                if "title" not in columns:
                    logger.info("Migrating database: Adding 'title' column to 'videos' table.")
                    conn.execute(text("ALTER TABLE videos ADD COLUMN title VARCHAR(256);"))
                    conn.commit()

                summary_columns = [c["name"] for c in inspector.get_columns("summaries")]
                if "key_takeaways_json" not in summary_columns:
                    logger.info("Migrating database: Adding 'key_takeaways_json' column to 'summaries' table.")
                    conn.execute(text("ALTER TABLE summaries ADD COLUMN key_takeaways_json JSON;"))
                    conn.commit()

                # Migration for video_standards
                std_columns = [c["name"] for c in inspector.get_columns("video_standards")]
                if "allowed_codecs" not in std_columns:
                    logger.info("Migrating database: Adding 'allowed_codecs' column to 'video_standards' table.")
                    conn.execute(text("ALTER TABLE video_standards ADD COLUMN allowed_codecs JSON DEFAULT '[\"h264\", \"hevc\"]';"))
                    conn.commit()
                if "min_width" not in std_columns:
                    logger.info("Migrating database: Adding 'min_width' column to 'video_standards' table.")
                    conn.execute(text("ALTER TABLE video_standards ADD COLUMN min_width INTEGER NOT NULL DEFAULT 640;"))
                    conn.commit()
                if "min_height" not in std_columns:
                    logger.info("Migrating database: Adding 'min_height' column to 'video_standards' table.")
                    conn.execute(text("ALTER TABLE video_standards ADD COLUMN min_height INTEGER NOT NULL DEFAULT 360;"))
                    conn.commit()
                if "max_width" not in std_columns:
                    logger.info("Migrating database: Adding 'max_width' column to 'video_standards' table.")
                    conn.execute(text("ALTER TABLE video_standards ADD COLUMN max_width INTEGER NOT NULL DEFAULT 1920;"))
                    conn.commit()
                if "max_height" not in std_columns:
                    logger.info("Migrating database: Adding 'max_height' column to 'video_standards' table.")
                    conn.execute(text("ALTER TABLE video_standards ADD COLUMN max_height INTEGER NOT NULL DEFAULT 1080;"))
                    conn.commit()

                # Migration for video_metadata
                meta_columns = [c["name"] for c in inspector.get_columns("video_metadata")]
                if "video_codec" not in meta_columns:
                    logger.info("Migrating database: Adding 'video_codec' column to 'video_metadata' table.")
                    conn.execute(text("ALTER TABLE video_metadata ADD COLUMN video_codec VARCHAR(50);"))
                    conn.commit()
                if "audio_codec" not in meta_columns:
                    logger.info("Migrating database: Adding 'audio_codec' column to 'video_metadata' table.")
                    conn.execute(text("ALTER TABLE video_metadata ADD COLUMN audio_codec VARCHAR(50);"))
                    conn.commit()
                if "bitrate" not in meta_columns:
                    logger.info("Migrating database: Adding 'bitrate' column to 'video_metadata' table.")
                    conn.execute(text("ALTER TABLE video_metadata ADD COLUMN bitrate INTEGER;"))
                    conn.commit()

                # Migration for users: output_language
                user_columns = [c["name"] for c in inspector.get_columns("users")]
                if "output_language" not in user_columns:
                    logger.info("Migrating database: Adding 'output_language' column to 'users' table.")
                    conn.execute(text("ALTER TABLE users ADD COLUMN output_language VARCHAR(10) NOT NULL DEFAULT 'vi';"))
                    conn.commit()

                # Backfill any null standard values and user preferences
                conn.execute(text("UPDATE users SET output_language = 'vi' WHERE output_language IS NULL;"))
                conn.execute(text("UPDATE video_standards SET allowed_codecs = '[\"h264\", \"hevc\"]' WHERE allowed_codecs IS NULL;"))
                conn.execute(text("UPDATE video_standards SET min_width = 640 WHERE min_width IS NULL;"))
                conn.execute(text("UPDATE video_standards SET min_height = 360 WHERE min_height IS NULL;"))
                conn.execute(text("UPDATE video_standards SET max_width = 1920 WHERE max_width IS NULL;"))
                conn.execute(text("UPDATE video_standards SET max_height = 1080 WHERE max_height IS NULL;"))
                conn.commit()
        except Exception as e:
            logger.warning(f"Self-healing database migration check failed: {e}")
        # 1. Initialize Default Standard if empty
        std_count = db.query(VideoStandard).count()
        if std_count == 0:
            default_std = VideoStandard(
                max_duration=settings.DEFAULT_MAX_DURATION_SECONDS,
                allowed_formats=settings.DEFAULT_ALLOWED_FORMATS,
                max_file_size=settings.DEFAULT_MAX_FILE_SIZE_MB,
                allowed_codecs=["h264", "hevc"],
                min_width=640,
                min_height=360,
                max_width=1920,
                max_height=1080,
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
