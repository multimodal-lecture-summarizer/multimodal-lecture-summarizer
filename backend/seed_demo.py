import sys
import os
import uuid
import logging
import json

# Disable SQLAlchemy engine logs to prevent encoding issues in some environments
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Add backend directory to system path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

from datetime import datetime
from app.core.database import SessionLocal, Base, engine, verify_db_connection
from app.models.user import User
from app.models.video import Video, VideoMetadata, VideoScene
from app.models.summary import Summary
from app.models.job import Job
from app.api.deps import get_password_hash
from app.core.constants import UserRole, VideoStatus, JobType, JobStatus
from app.services.r2 import r2_service
from app.services.chromadb import chromadb_service

# Define predefined scripts for each scene of the youtube_clip.mp4
SCENE_SCRIPTS = {
    1: "Welcome everyone to our lecture on multimodal video summarization. Today, we'll introduce the architecture of our system.",
    2: "Let's start with the system architecture. Our system utilizes a three-tier microservice architecture, containing a user-friendly frontend, a robust FastAPI backend API, and Celery AI workers for asynchronous background processing.",
    3: "We have integrated multiple specialized pipeline modules to analyze different modalities of video data.",
    4: "For visual processing, our pipeline utilizes PySceneDetect to segment the video into semantic scenes based on pixel transitions. After that, we apply CLIP semantic filtering to cluster and filter keyframes, keeping only the most visually unique and informative slides.",
    5: "Once the keyframes are filtered, we pass them through a state-of-the-art vision-language model, BLIP-2, to generate detailed semantic captions for each slide.",
    6: "Simultaneously, the audio processing module extracts the speech track, uses noise reduction, and runs WhisperX for high-accuracy speech-to-text transcription aligned with timestamps.",
    7: "The frontend dashboard is developed using ReactJS and Tailwind CSS. It features a modern interface to upload files, view generated chapters, scroll through keyframe galleries, and export reports in PDF or TXT formats.",
    8: "We store the structured metadata in a PostgreSQL database and host the media files and extracted keyframe assets on Cloudflare R2 storage to guarantee scalability.",
    9: "Finally, the fusion module uses LangChain to combine speech transcripts and visual captions, creating a unified chapter timeline and enabling Q&A RAG using ChromaDB.",
    10: "This concludes the overview of our multimodal lecture summarizer. Let's proceed to the live demonstration and check the results on the dashboard."
}

def seed_demo_data():
    print("--- Verifying Database Connection ---")
    verify_db_connection(retries=3, delay=1.0)

    print("--- Creating Database Tables if needed ---")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. Fetch or create admin user
        print("--- Finding or creating Admin User ---")
        admin_email = "hungphitran.22@gmail.com"
        admin_user = db.query(User).filter(User.email == admin_email).first()
        if not admin_user:
            hashed_pwd = get_password_hash("AdminPass123@")
            admin_user = User(
                email=admin_email,
                password_hash=hashed_pwd,
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            print(f"Admin user created successfully! ({admin_email})")
        else:
            print(f"Admin user already exists ({admin_email})")

        # Define path to demo data directory
        demo_data_dir = os.path.abspath(os.path.join(backend_dir, "..", "experiments", "notebooks", "demo_data"))
        if not os.path.exists(demo_data_dir):
            print(f"ERROR: Demo data directory not found at: {demo_data_dir}")
            return

        visual_result_path = os.path.join(demo_data_dir, "results", "visual_result.json")
        if not os.path.exists(visual_result_path):
            print(f"ERROR: visual_result.json not found at: {visual_result_path}")
            return

        with open(visual_result_path, "r", encoding="utf-8") as f:
            visual_data = json.load(f)

        # 2. Upload video files to R2
        print("--- Uploading video files to Cloudflare R2 ---")
        video_files = ["youtube_clip.mp4", "synthetic_lecture.mp4"]
        uploaded_video_urls = {}

        for video_name in video_files:
            local_video_path = os.path.join(demo_data_dir, "videos", video_name)
            if not os.path.exists(local_video_path):
                print(f"WARNING: Video file not found: {local_video_path}")
                continue
            
            object_name = f"videos/{video_name}"
            print(f"Uploading {video_name} to R2...")
            r2_url = r2_service.upload_file(local_video_path, object_name)
            if r2_url:
                uploaded_video_urls[video_name] = r2_url
                print(f"SUCCESS: Uploaded {video_name} -> {r2_url}")
            else:
                print(f"ERROR: Failed to upload {video_name}")

        # 3. Process youtube_clip.mp4 & Seed Video + VideoMetadata + VideoScenes
        youtube_clip_name = "youtube_clip.mp4"
        if youtube_clip_name in uploaded_video_urls:
            youtube_url = "https://www.youtube.com/watch?v=VO6XEQIsCoM"
            
            # Check if Video record already exists
            db_video = db.query(Video).filter(Video.original_url == youtube_url).first()
            if db_video:
                print(f"INFO: Video record for {youtube_url} already exists. Recreating it to refresh metadata.")
                db.delete(db_video)
                db.commit()

            # Create Video record
            db_video = Video(
                user_id=admin_user.user_id,
                original_url=youtube_url,
                file_path=uploaded_video_urls[youtube_clip_name],
                duration=visual_data["video_info"]["duration"],
                language="en",
                status=VideoStatus.DONE
            )
            db.add(db_video)
            db.commit()
            db.refresh(db_video)
            print(f"SUCCESS: Created Video record (ID: {db_video.video_id})")

            # Create VideoMetadata record
            print("--- Creating VideoMetadata record ---")
            db_metadata = VideoMetadata(
                video_id=db_video.video_id,
                fps=visual_data["video_info"]["fps"],
                frame_count=visual_data["video_info"]["frame_count"],
                width=visual_data["video_info"]["width"],
                height=visual_data["video_info"]["height"],
                video_source=visual_data["video_source"],
                video_path=visual_data["video_path"]
            )
            db.add(db_metadata)
            db.commit()
            print("SUCCESS: Created VideoMetadata record.")

            # Upload Keyframes & Create VideoScene records
            print("--- Uploading keyframes & creating VideoScene records ---")
            scene_records = []
            for scene in visual_data["scenes"]:
                scene_id_val = scene["scene_id"]
                keyframe_name = os.path.basename(scene["keyframe_path"])
                local_keyframe_path = os.path.join(demo_data_dir, "keyframes", keyframe_name)

                keyframe_url = ""
                if os.path.exists(local_keyframe_path):
                    object_keyframe_name = f"keyframes/youtube_clip/{keyframe_name}"
                    r2_kf_url = r2_service.upload_file(local_keyframe_path, object_keyframe_name)
                    if r2_kf_url:
                        keyframe_url = r2_kf_url
                        print(f"Uploaded keyframe {keyframe_name} -> {keyframe_url}")
                    else:
                        print(f"WARNING: Failed to upload keyframe {keyframe_name} to R2")
                else:
                    print(f"WARNING: Local keyframe image not found at: {local_keyframe_path}")

                db_scene = VideoScene(
                    video_id=db_video.video_id,
                    scene_index=scene_id_val,
                    start_seconds=scene["start_seconds"],
                    end_seconds=scene["end_seconds"],
                    start_timecode=scene["start_timecode"],
                    end_timecode=scene["end_timecode"],
                    start_frame=scene["start_frame"],
                    end_frame=scene["end_frame"],
                    keyframe_path=scene["keyframe_path"],
                    keyframe_url=keyframe_url,
                    caption=scene["caption"],
                    script=SCENE_SCRIPTS.get(scene_id_val, "")
                )
                db.add(db_scene)
                scene_records.append(db_scene)
            
            db.commit()
            print("SUCCESS: Seeded all VideoScenes records.")

            # Create Summary for YouTube Clip
            print("--- Creating Summary record for YouTube Clip ---")
            chapters_json = [
                {
                    "title": "Giới thiệu & Tổng quan kiến trúc",
                    "startTime": 0.0,
                    "endTime": 27.8,
                    "summary": "Chào mừng và giới thiệu tổng quan về hệ thống tóm tắt video bài giảng đa phương tiện với kiến trúc 3 lớp."
                },
                {
                    "title": "Phân tích hình ảnh (PySceneDetect & CLIP)",
                    "startTime": 27.8,
                    "endTime": 116.567,
                    "summary": "Phân cảnh video bằng PySceneDetect và lọc khung hình chính bằng CLIP nhằm tối ưu hóa slide giảng dạy."
                },
                {
                    "title": "Nhận diện & Chuyển âm thanh thành văn bản",
                    "startTime": 116.567,
                    "endTime": 153.567,
                    "summary": "Tạo mô tả chi tiết bằng BLIP-2 và chuyển đổi giọng nói chính xác với WhisperX."
                },
                {
                    "title": "Giao diện Dashboard & Lưu trữ Cloudflare R2",
                    "startTime": 153.567,
                    "endTime": 213.133,
                    "summary": "Trải nghiệm người dùng trên dashboard ReactJS và giải pháp lưu trữ mở rộng kết hợp PostgreSQL + Cloudflare R2."
                },
                {
                    "title": "Tổng hợp thông tin & RAG Q&A",
                    "startTime": 213.133,
                    "endTime": 300.0,
                    "summary": "Hợp nhất hình ảnh/âm thanh với LangChain và triển khai dịch vụ RAG Hỏi Đáp qua ChromaDB."
                }
            ]

            selected_scene_ids = [1, 2, 3, 4, 8]
            keyframes_json = []
            for s_rec in scene_records:
                if s_rec.scene_index in selected_scene_ids:
                    keyframes_json.append({
                        "timestamp": (s_rec.start_seconds + s_rec.end_seconds) / 2.0,
                        "imageUrl": s_rec.keyframe_url,
                        "description": s_rec.caption,
                        "importanceScore": 0.95 if s_rec.scene_index in [1, 4] else 0.85
                    })

            transcript_text = " ".join([SCENE_SCRIPTS[i] for i in sorted(SCENE_SCRIPTS.keys())])

            db_summary = Summary(
                video_id=db_video.video_id,
                summary_text=(
                    "Đây là bài giảng giới thiệu về hệ thống tóm tắt bài giảng đa phương tiện (Multimodal Lecture Summarizer). "
                    "Hệ thống sử dụng kiến trúc ba lớp bao gồm Frontend ReactJS/Tailwind CSS, Backend FastAPI, và Celery AI Workers cho việc xử lý không đồng bộ. "
                    "Quy trình phân tích dữ liệu đa phương tiện được thiết kế với các module chuyên biệt: phân tách cảnh bằng PySceneDetect, "
                    "lọc khung hình thông minh thông qua CLIP, mô tả khung hình bằng BLIP-2, nhận dạng giọng nói bằng WhisperX, và tổng hợp thông tin "
                    "bằng LangChain kết hợp ChromaDB cho phép hỏi đáp RAG."
                ),
                chapters_json=chapters_json,
                keyframes_json=keyframes_json,
                transcript_text=transcript_text,
                model_used="WhisperX + BLIP-2 + Gemini 1.5 Pro",
                processing_time=45.2
            )
            db.add(db_summary)

            # Create Job record for YouTube Clip
            db_job = Job(
                video_id=db_video.video_id,
                job_type=JobType.SUMMARIZE,
                status=JobStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_log=None,
            )
            db.add(db_job)
            db.commit()
            print("SUCCESS: Seeded Summary and Job record for YouTube Clip.")

            # Seed Vector Database (ChromaDB) for YouTube Clip
            print("--- Seeding vector chunks in ChromaDB for YouTube Clip ---")
            chunks = list(SCENE_SCRIPTS.values())
            metadatas = [{"video_id": str(db_video.video_id)} for _ in chunks]
            try:
                chromadb_service.add_transcript_chunks(
                    str(db_video.video_id), chunks, metadatas
                )
                print(f"SUCCESS: Seeded vector chunks in ChromaDB for Video {db_video.video_id}")
            except Exception as e:
                print(f"WARNING: Failed to seed ChromaDB: {e}")
        else:
            print("ERROR: youtube_clip.mp4 was not uploaded successfully; skipping scene, summary and job seeding.")

        # 4. Process synthetic_lecture.mp4 (Create video + summary + job entry)
        synthetic_lecture_name = "synthetic_lecture.mp4"
        if synthetic_lecture_name in uploaded_video_urls:
            db_syn_video = db.query(Video).filter(Video.file_path.like(f"%{synthetic_lecture_name}%")).first()
            if db_syn_video:
                print(f"INFO: Video record for {synthetic_lecture_name} already exists. Recreating it to refresh metadata.")
                db.delete(db_syn_video)
                db.commit()

            db_syn_video = Video(
                user_id=admin_user.user_id,
                original_url=None,
                file_path=uploaded_video_urls[synthetic_lecture_name],
                duration=185.0,
                language="en",
                status=VideoStatus.DONE
            )
            db.add(db_syn_video)
            db.commit()
            db.refresh(db_syn_video)
            print(f"SUCCESS: Created Video record for Synthetic Lecture (ID: {db_syn_video.video_id})")

            # Create Summary for Synthetic Lecture
            print("--- Creating Summary record for Synthetic Lecture ---")
            chapters_syn = [
                {
                    "title": "Giới thiệu Học máy & NLP",
                    "startTime": 0.0,
                    "endTime": 60.0,
                    "summary": "Giới thiệu các khái niệm cơ bản về trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên."
                },
                {
                    "title": "Mô hình học máy truyền thống",
                    "startTime": 60.0,
                    "endTime": 120.0,
                    "summary": "Phân tích các thuật toán học máy cổ điển như SVM, Naive Bayes và hạn chế của chúng."
                },
                {
                    "title": "Kỷ nguyên Deep Learning & Transformer",
                    "startTime": 120.0,
                    "endTime": 185.0,
                    "summary": "Giới thiệu kiến trúc Transformer và cách nó cách mạng hóa các bài toán xử lý chuỗi và ngôn ngữ."
                }
            ]

            syn_transcript = (
                "Welcome to the lecture on Machine Learning and Natural Language Processing. Today we will cover the history and evolution of NLP. "
                "In the early days, NLP systems were based on complex hand-written rules. While highly interpretable, they were fragile and failed to scale. "
                "In the 1990s and 2000s, statistical machine learning models like SVM and Naive Bayes took over, using data to make probabilistic decisions. "
                "Finally, the deep learning revolution, especially the introduction of the Transformer architecture in 2017, completely transformed the field, "
                "enabling models to capture long-range context efficiently and perform diverse language tasks at scale."
            )

            db_syn_summary = Summary(
                video_id=db_syn_video.video_id,
                summary_text=(
                    "Bài giảng tổng hợp giới thiệu về các khái niệm cơ bản trong học máy và xử lý ngôn ngữ tự nhiên. "
                    "Video trình bày sự tiến hóa từ các hệ thống dựa trên luật lệ truyền thống sang các mô hình học sâu hiện đại như Transformer, "
                    "phân tích ưu nhược điểm của từng phương pháp và ứng dụng thực tế."
                ),
                chapters_json=chapters_syn,
                keyframes_json=[],
                transcript_text=syn_transcript,
                model_used="WhisperX + Gemini 1.5 Flash",
                processing_time=28.5
            )
            db.add(db_syn_summary)

            # Create Job record for Synthetic Lecture
            db_syn_job = Job(
                video_id=db_syn_video.video_id,
                job_type=JobType.SUMMARIZE,
                status=JobStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_log=None,
            )
            db.add(db_syn_job)
            db.commit()
            print("SUCCESS: Seeded Summary and Job record for Synthetic Lecture.")

            # Seed Vector Database (ChromaDB) for Synthetic Lecture
            print("--- Seeding vector chunks in ChromaDB for Synthetic Lecture ---")
            syn_chunks = [
                "Welcome to the lecture on Machine Learning and Natural Language Processing. Today we will cover the history and evolution of NLP.",
                "In the early days, NLP systems were based on complex hand-written rules. While highly interpretable, they were fragile and failed to scale.",
                "In the 1990s and 2000s, statistical machine learning models like SVM and Naive Bayes took over, using data to make probabilistic decisions.",
                "Finally, the deep learning revolution, especially the introduction of the Transformer architecture in 2017, completely transformed the field."
            ]
            syn_metadatas = [{"video_id": str(db_syn_video.video_id)} for _ in syn_chunks]
            try:
                chromadb_service.add_transcript_chunks(
                    str(db_syn_video.video_id), syn_chunks, syn_metadatas
                )
                print(f"SUCCESS: Seeded vector chunks in ChromaDB for Video {db_syn_video.video_id}")
            except Exception as e:
                print(f"WARNING: Failed to seed ChromaDB for Synthetic Lecture: {e}")

        print("--- DEMO SEEDING COMPLETED SUCCESSFULLY ---")

    except Exception as e:
        db.rollback()
        print(f"ERROR: Seeding failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_demo_data()
