from datetime import date
from sqlalchemy import Integer, Date
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class SystemStat(Base):
    __tablename__ = "system_stats"

    stat_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    total_users: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    new_users: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_videos_processed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    total_jobs_run: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
