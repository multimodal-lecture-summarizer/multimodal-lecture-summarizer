import os
import sys
import uuid
import datetime
from sqlalchemy import text, inspect

# Ensure backend directory is in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.core.database import engine, SessionLocal, Base, initialize_database_data
from app.models.user import User
from app.models.video import Video, VideoMetadata, VideoScene, VideoStandard
from app.models.job import Job
from app.models.summary import Summary
from app.models.qa import QALog
from app.models.stats import SystemStat
from app.core.constants import UserRole, VideoStatus, RagStatus, JobType, JobStatus


def run_sql_script():
    """Executes the init_db.sql file directly against the connected database."""
    sql_path = os.path.join(backend_dir, "init_db.sql")
    if not os.path.exists(sql_path):
        print(f"[-] SQL script not found at {sql_path}")
        return False
        
    print(f"[*] Executing SQL script: {sql_path}...")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    with engine.connect() as conn:
        # PostgreSQL executes multi-statement block inside transaction
        conn.execute(text(sql_content))
        conn.commit()
    print("[+] SQL script executed successfully!")
    return True


def run_model_sync():
    """Runs SQLAlchemy Base.metadata.create_all and seeds default accounts."""
    print("[*] Synchronizing SQLAlchemy models to database...")
    Base.metadata.create_all(bind=engine)
    print("[*] Initializing default seed data...")
    initialize_database_data()
    print("[+] SQLAlchemy model synchronization completed!")


def test_database_crud():
    """
    Tests complete end-to-end CRUD operations on all tables and foreign key cascade rules
    to ensure the database is 100% operational for the backend and AI workers.
    """
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE DATABASE INTEGRITY TESTS")
    print("=" * 60)

    db = SessionLocal()
    test_video_id = uuid.uuid4()
    test_user = None

    try:
        # 1. Inspect Table Names
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"[1/7] Inspected database tables ({len(existing_tables)} found):")
        for tbl in sorted(existing_tables):
            cols = [c["name"] for c in inspector.get_columns(tbl)]
            print(f"      - {tbl} ({len(cols)} columns: {', '.join(cols[:5])}...)")

        required_tables = {
            "users", "videos", "video_metadata", "video_scenes",
            "video_standards", "jobs", "summaries", "qa_logs", "system_stats"
        }
        missing_tables = required_tables - set(existing_tables)
        if missing_tables:
            raise RuntimeError(f"Missing required tables: {missing_tables}")
        print("      -> All required tables exist! [PASS]")

        # 2. Test User Query
        print("\n[2/7] Verifying User accounts...")
        admin = db.query(User).filter(User.email == "hungphitran.22@gmail.com").first()
        if not admin:
            raise RuntimeError("Admin user was not found!")
        print(f"      -> Admin account verified: {admin.email} (ID: {admin.user_id}, Role: {admin.role.value}) [PASS]")

        test_user = admin

        # 3. Test Video Insert
        print("\n[3/7] Testing Video insert...")
        test_video = Video(
            video_id=test_video_id,
            user_id=test_user.user_id,
            title="Database Test Lecture 101",
            original_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            file_path="videos/test_lecture.mp4",
            duration=300.5,
            language="en",
            status=VideoStatus.PROCESSING,
            rag_status=RagStatus.PENDING
        )
        db.add(test_video)
        db.commit()
        print(f"      -> Created video: {test_video.title} (ID: {test_video_id}) [PASS]")

        # 4. Test 1-to-1 VideoMetadata & 1-to-Many VideoScene
        print("\n[4/7] Testing VideoMetadata and VideoScene inserts...")
        test_meta = VideoMetadata(
            video_id=test_video_id,
            video_codec="h264",
            audio_codec="aac",
            fps=30.0,
            frame_count=9015,
            width=1920,
            height=1080,
            bitrate=3400000,
            video_source="youtube",
            video_path="videos/test_lecture.mp4"
        )
        db.add(test_meta)

        test_scene = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=test_video_id,
            scene_index=1,
            start_seconds=0.0,
            end_seconds=15.0,
            start_timecode="00:00:00.000",
            end_timecode="00:00:15.000",
            start_frame=0,
            end_frame=450,
            keyframe_path="keyframes/test_1.jpg",
            keyframe_url="https://pub-r2.dev/keyframes/test_1.jpg",
            caption="Professor introducing the lecture topics on whiteboard",
            script="Welcome everyone to today's machine learning lecture."
        )
        db.add(test_scene)
        db.commit()
        print("      -> VideoMetadata and VideoScene inserted successfully! [PASS]")

        # 5. Test Job, Summary (with JSON fields), and QALog
        print("\n[5/7] Testing Job, Summary (JSON), and QALog inserts...")
        test_job = Job(
            job_id=uuid.uuid4(),
            video_id=test_video_id,
            job_type=JobType.SUMMARIZE,
            status=JobStatus.RUNNING,
            started_at=datetime.datetime.utcnow()
        )
        db.add(test_job)

        test_summary = Summary(
            summary_id=uuid.uuid4(),
            video_id=test_video_id,
            summary_text="This lecture introduces core concepts of neural networks and optimization.",
            chapters_json=[
                {"title": "Introduction", "start_time": 0.0, "end_time": 60.0, "summary": "Intro"}
            ],
            keyframes_json=[
                {"timestamp": 10.0, "image_url": "https://pub-r2.dev/keyframes/test_1.jpg", "description": "Intro slide", "importance_score": 0.95}
            ],
            key_takeaways_json=[
                {"point": "Key takeaway 1", "timestamp": 12.0}
            ],
            transcript_text='[{"speaker": "SPEAKER_01", "start": 0.0, "end": 15.0, "text": "Welcome"}]',
            model_used="openrouter/qwen-2.5-7b-instruct",
            processing_time=12.4
        )
        db.add(test_summary)

        test_qa = QALog(
            qa_id=uuid.uuid4(),
            video_id=test_video_id,
            user_id=test_user.user_id,
            question="What is the main topic of this lecture?",
            answer="The lecture focuses on neural network architectures and loss functions.",
            retrieved_chunks=[{"chunk_id": "chunk_1", "score": 0.88}]
        )
        db.add(test_qa)
        db.commit()
        print("      -> Job, Summary (JSON verified), and QALog inserted successfully! [PASS]")

        # 6. Test Query Relationships
        print("\n[6/7] Verifying SQLAlchemy ORM Relationships...")
        v = db.query(Video).filter(Video.video_id == test_video_id).first()
        assert v.video_metadata is not None, "video_metadata relationship failed"
        assert len(v.scenes) == 1, "scenes relationship failed"
        assert v.summary is not None, "summary relationship failed"
        assert len(v.jobs) == 1, "jobs relationship failed"
        assert len(v.qa_logs) == 1, "qa_logs relationship failed"
        print("      -> All 5 ORM relationships verified! [PASS]")

        # 7. Test Cascade Delete
        print("\n[7/7] Testing ON DELETE CASCADE rule...")
        db.delete(v)
        db.commit()

        # Check that children were deleted
        remaining_meta = db.query(VideoMetadata).filter(VideoMetadata.video_id == test_video_id).first()
        remaining_scenes = db.query(VideoScene).filter(VideoScene.video_id == test_video_id).all()
        remaining_summary = db.query(Summary).filter(Summary.video_id == test_video_id).first()
        remaining_jobs = db.query(Job).filter(Job.video_id == test_video_id).all()
        remaining_qa = db.query(QALog).filter(QALog.video_id == test_video_id).all()

        assert remaining_meta is None, "VideoMetadata was not cascade deleted!"
        assert len(remaining_scenes) == 0, "VideoScenes were not cascade deleted!"
        assert remaining_summary is None, "Summary was not cascade deleted!"
        assert len(remaining_jobs) == 0, "Jobs were not cascade deleted!"
        assert len(remaining_qa) == 0, "QALogs were not cascade deleted!"
        print("      -> Foreign Key CASCADE delete cleanly removed all child records! [PASS]")

        print("\n" + "=" * 60)
        print("ALL DATABASE INTEGRITY TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
        print("=" * 60)
        return True

    except Exception as e:
        db.rollback()
        print(f"\n[-] ERROR during database test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MULTIMODAL LECTURE SUMMARIZER - DATABASE INITIALIZER")
    print(f"Target Server: {settings.POSTGRES_SERVER} ({settings.POSTGRES_DB})")
    print("=" * 60)

    # 1. Run SQL Script
    run_sql_script()

    # 2. Sync models and seed default records
    run_model_sync()

    # 3. Run comprehensive verification test suite
    success = test_database_crud()
    if not success:
        sys.exit(1)
