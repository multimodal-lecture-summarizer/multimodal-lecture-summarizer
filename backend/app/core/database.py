from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

# Create engine for PostgreSQL
# We configure pool_pre_ping=True to prevent stale connections
engine = create_engine(
    settings.sqlalchemy_database_uri,
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
