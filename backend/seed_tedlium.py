import sys
import os
import subprocess
import logging
from datetime import datetime

# Disable SQLAlchemy engine logs to prevent UnicodeEncodeError in Windows terminals
logging.basicConfig(level=logging.INFO)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Add current directory to path so we can import app modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Auto-install datasets package if not present
try:
    import datasets
except ImportError:
    print("Installing 'datasets' package using pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets

from datasets import load_dataset

from app.core.database import SessionLocal, Base, engine, verify_db_connection
from app.models.user import User
from app.models.video import Video
from app.models.summary import Summary
from app.models.job import Job
from app.core.constants import UserRole, VideoStatus, JobStatus, JobType
from app.services.chromadb import chromadb_service


def seed_tedlium():
    print("--- Verifying Database Connection ---")
    verify_db_connection(retries=3, delay=1.0)

    db = SessionLocal()
    try:
        # Get admin user to associate with the video
        admin_email = "hungphitran.22@gmail.com"
        admin_user = db.query(User).filter(User.email == admin_email).first()
        if not admin_user:
            print("ERROR: Admin user not found. Please run seed.py first to create the admin user.")
            return

        # Check if Barry Schwartz video is already seeded
        url = "https://www.youtube.com/watch?v=VO6XEQIsCoM"
        video = db.query(Video).filter(Video.original_url == url).first()
        if video:
            print("INFO: Barry Schwartz TED-LIUM video is already seeded in the database.")
            return

        dataset = load_dataset(
            "distil-whisper/tedlium",
            "default",
            split="validation",
            revision="refs/convert/parquet",
            streaming=True
        )
        # Disable column decoding globally to prevent audio decoding errors
        dataset = dataset.decode(False)
        # Remove audio column to avoid decoding dependencies like soundfile/torchcodec
        dataset = dataset.remove_columns(["audio"])

        # Get first 10 valid validation segments
        print("--- Fetching segments from validation split ---")
        val_stream = iter(dataset)
        chunks = []
        durations = []

        for sample in val_stream:
            text = sample.get("text", "").strip()
            if not text or text == "ignore_time_segment_in_scoring":
                continue

            # Estimate duration based on word count (approx. 0.4 seconds per word)
            word_count = len(text.split())
            duration = max(word_count * 0.4, 5.0)

            chunks.append(text)
            durations.append(duration)

            if len(chunks) >= 10:
                break

        if not chunks:
            print("ERROR: Failed to fetch any segments from TED-LIUM dataset.")
            return

        print(f"SUCCESS: Fetched {len(chunks)} segments from TED-LIUM dataset.")

        # 1. Create Video Record
        total_duration = sum(durations)
        video = Video(
            user_id=admin_user.user_id,
            original_url=url,
            duration=total_duration,
            language="en",
            status=VideoStatus.DONE,
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        print(f"SUCCESS: Created Video in SQL DB: {video.video_id}")

        # 2. Create Summary Record
        full_transcript = " ".join(chunks)
        summary_text = (
            "Bài thuyết trình mang tính giáo dục và truyền cảm hứng dựa trên dữ liệu TED-LIUM. "
            "Các phân đoạn âm thanh thực tế mô tả cuộc sống, khoa học và tư duy của các diễn giả TED. "
            "Nghịch lý của sự lựa chọn chỉ ra rằng nhiều sự lựa chọn hơn có thể dẫn đến lo lắng và kém hạnh phúc hơn."
        )

        # Create chapters based on segment timings
        chapters_json = []
        current_time = 0.0
        for i, (chunk, dur) in enumerate(zip(chunks, durations)):
            chapters_json.append({
                "title": f"Phần {i + 1}: Phân tích phân đoạn TED-LIUM {i + 1}",
                "startTime": current_time,
                "endTime": current_time + dur,
                "summary": f"Diễn thuyết đoạn: {chunk[:100]}..."
            })
            current_time += dur

        keyframes_json = [
            {
                "timestamp": 12.0,
                "imageUrl": "/static/mock_r2/keyframes/slide1.png",
                "description": "Hình ảnh Barry Schwartz bắt đầu bài diễn thuyết",
                "importanceScore": 0.90
            },
            {
                "timestamp": total_duration / 2,
                "imageUrl": "/static/mock_r2/keyframes/slide2.png",
                "description": "Phân cấp các sự lựa chọn trong xã hội hiện đại",
                "importanceScore": 0.85
            }
        ]

        summary = Summary(
            video_id=video.video_id,
            summary_text=summary_text,
            chapters_json=chapters_json,
            keyframes_json=keyframes_json,
            transcript_text=full_transcript,
            model_used="WhisperX + distil-whisper/tedlium",
            processing_time=4.12
        )
        db.add(summary)

        # 3. Create Job Record
        job = Job(
            video_id=video.video_id,
            job_type=JobType.SUMMARIZE,
            status=JobStatus.COMPLETED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error_log=None
        )
        db.add(job)
        db.commit()
        print("SUCCESS: Created Summary and Job records in SQL DB.")

        # 4. Add transcript chunks to ChromaDB
        metadatas = [{"video_id": str(video.video_id), "source": "tedlium_segment"} for _ in chunks]
        chromadb_service.add_transcript_chunks(
            str(video.video_id),
            chunks,
            metadatas
        )
        print("SUCCESS: Added TED-LIUM transcript chunks to ChromaDB.")
        print("--- TED-LIUM SEEDING COMPLETED SUCCESSFULLY ---")

    except Exception as e:
        db.rollback()
        print(f"ERROR: Seeding TED-LIUM failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_tedlium()
