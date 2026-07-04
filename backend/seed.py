import sys
import os
import uuid
import logging
from datetime import datetime

# Disable SQLAlchemy engine logs to prevent UnicodeEncodeError in Windows terminals
# when writing Vietnamese characters to stdout/stderr.
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Add current directory to path so we can import app modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal, Base, engine
from app.models.user import User
from app.models.video import Video
from app.models.summary import Summary
from app.models.job import Job
from app.api.deps import get_password_hash
from app.core.constants import UserRole, VideoStatus, JobStatus, JobType
from app.services.chromadb import chromadb_service


def seed_database():
    print("--- Initializing database tables if needed ---")
    Base.metadata.create_all(bind=engine)

    print("--- Opening Database Session ---")
    db = SessionLocal()
    try:
        # 1. Seed Users
        print("--- Seeding users (with forced password reset) ---")
        admin_email = "hungphitran.22@gmail.com"
        admin_user = db.query(User).filter(User.email == admin_email).first()
        hashed_pwd = get_password_hash("AdminPass123@")
        if not admin_user:
            admin_user = User(
                email=admin_email,
                password_hash=hashed_pwd,
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            print(
                f"SUCCESS: Admin user created successfully! ({admin_email} / AdminPass123@)"
            )
        else:
            admin_user.password_hash = hashed_pwd
            admin_user.role = UserRole.ADMIN
            admin_user.is_active = True
            db.commit()
            db.refresh(admin_user)
            print(
                f"SUCCESS: Admin user '{admin_email}' password reset to 'AdminPass123@'"
            )

        user_email = "nguyen.van.a@gmail.com"
        normal_user = db.query(User).filter(User.email == user_email).first()
        hashed_user_pwd = get_password_hash("UserPass123@")
        if not normal_user:
            normal_user = User(
                email=user_email,
                password_hash=hashed_user_pwd,
                role=UserRole.USER,
                is_active=True,
            )
            db.add(normal_user)
            db.commit()
            db.refresh(normal_user)
            print(
                f"SUCCESS: Normal user created successfully! ({user_email} / UserPass123@)"
            )
        else:
            normal_user.password_hash = hashed_user_pwd
            normal_user.role = UserRole.USER
            normal_user.is_active = True
            db.commit()
            db.refresh(normal_user)
            print(
                f"SUCCESS: Normal user '{user_email}' password reset to 'UserPass123@'"
            )

        # 2. Seed Videos, Summaries, and Jobs
        print("--- Seeding video data ---")

        # Talk 1: Ken Robinson
        url_1 = "https://www.youtube.com/watch?v=iG9CE55wbtY"
        video_1 = db.query(Video).filter(Video.original_url == url_1).first()
        if not video_1:
            video_1 = Video(
                user_id=admin_user.user_id,
                original_url=url_1,
                duration=1164.0,
                language="en",
                status=VideoStatus.DONE,
            )
            db.add(video_1)
            db.commit()
            db.refresh(video_1)
            print(
                f"SUCCESS: Seeded video: Ken Robinson (ID: {video_1.video_id})"
            )

            # Seed Summary
            summary_1 = Summary(
                video_id=video_1.video_id,
                summary_text=(
                    "Bài phát biểu mang tính biểu tượng của Sir Ken Robinson thảo luận về cách hệ thống "
                    "giáo dục hiện tại giết chết sự sáng tạo của trẻ em. Ông lập luận rằng sự sáng tạo "
                    "cũng quan trọng như khả năng biết viết chữ, và chúng ta nên đối xử với nó với vị thế "
                    "tương đương. Hệ thống giáo dục hiện đại được xây dựng dựa trên nhu cầu kinh tế của cuộc "
                    "Cách mạng Công nghiệp thế kỷ 19, thiết lập một thứ bậc các môn học với Toán và Ngôn ngữ "
                    "ở trên đỉnh, và Nghệ thuật ở dưới cùng. Ông kêu gọi thay đổi căn bản cách nhìn nhận "
                    "về trí thông minh, chấp nhận tính đa dạng và nuôi dưỡng năng lực sáng tạo của trẻ."
                ),
                chapters_json=[
                    {
                        "title": "Giới thiệu & Tầm quan trọng của Giáo dục",
                        "startTime": 0.0,
                        "endTime": 250.0,
                        "summary": "Sir Ken nhấn mạnh tầm quan trọng của giáo dục đối với tương lai chưa biết trước và vị thế ngang hàng của sự sáng tạo.",
                    },
                    {
                        "title": "Trẻ em và năng khiếu sáng tạo bị vùi dập",
                        "startTime": 250.0,
                        "endTime": 600.0,
                        "summary": "Tất cả trẻ em đều sinh ra với tài năng to lớn nhưng hệ thống giáo dục tàn nhẫn dập tắt sự sáng tạo vì sợ sai lầm.",
                    },
                    {
                        "title": "Lịch sử của hệ thống giáo dục công nghiệp",
                        "startTime": 600.0,
                        "endTime": 900.0,
                        "summary": "Mô tả nguồn gốc giáo dục phổ thông phục vụ công nghiệp hóa, tạo ra phân cấp môn học có lợi cho học thuật.",
                    },
                    {
                        "title": "Trí thông minh đa diện & Trường hợp Gillian Lynne",
                        "startTime": 900.0,
                        "endTime": 1164.0,
                        "summary": "Ví dụ về Gillian Lynne - một biên đạo múa tài ba suýt bị coi là thiểu năng. Trí thông minh rất đa dạng và năng động.",
                    },
                ],
                keyframes_json=[
                    {
                        "timestamp": 120.0,
                        "imageUrl": "/static/mock_r2/keyframes/slide1.png",
                        "description": "Biểu diễn tư duy đa chiều ở trẻ em",
                        "importanceScore": 0.95,
                    },
                    {
                        "timestamp": 450.0,
                        "imageUrl": "/static/mock_r2/keyframes/slide2.png",
                        "description": "Sơ đồ hệ thống cấp bậc môn học trong nhà trường",
                        "importanceScore": 0.88,
                    },
                    {
                        "timestamp": 800.0,
                        "imageUrl": "/static/mock_r2/keyframes/code1.png",
                        "description": "Ảnh chân dung nhà biên đạo múa nổi tiếng Gillian Lynne",
                        "importanceScore": 0.92,
                    },
                ],
                transcript_text=(
                    "Good morning. How are you? It's been great, hasn't it? I've been blown away by the whole thing. "
                    "In fact, I'm leaving. There have been three themes running through the conference, relevant to what "
                    "I want to talk about. One is the extraordinary evidence of human creativity in all presentations. "
                    "The second is that it's put us in a place where we have no idea what's going to happen. My contention "
                    "is that all kids have tremendous talents and we squander them ruthlessly. So I want to talk about "
                    "creativity. My contention is that creativity now is as important in education as literacy, and we "
                    "should treat it with the same status."
                ),
                model_used="WhisperX + Gemini 1.5 Pro",
                processing_time=12.45,
            )
            db.add(summary_1)

            # Seed Job
            job_1 = Job(
                video_id=video_1.video_id,
                job_type=JobType.SUMMARIZE,
                status=JobStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_log=None,
            )
            db.add(job_1)
            db.commit()

            # Seed Vector Database (ChromaDB)
            chunks_1 = [
                "Good morning. I want to talk about education and creativity. Creativity is as important as literacy.",
                "All kids have tremendous talents, but our educational system squanders them ruthlessly. We teach them to fear mistakes.",
                "Our school systems are built on the model of 19th-century industrialism, prioritizing academic ability over art.",
                "Gillian Lynne was a dancer who couldn't sit still in school, but a doctor recognized she needed to dance, not medication.",
            ]
            metadatas_1 = [
                {"video_id": str(video_1.video_id)} for _ in chunks_1
            ]
            chromadb_service.add_transcript_chunks(
                str(video_1.video_id), chunks_1, metadatas_1
            )
            print(
                f"SUCCESS: Seeded vector chunks in ChromaDB for Video {video_1.video_id}"
            )
        else:
            print(f"INFO: Video Ken Robinson already exists in DB.")

        # Talk 2: Al Gore
        url_2 = "https://www.youtube.com/watch?v=rDiGYuQicps"
        video_2 = db.query(Video).filter(Video.original_url == url_2).first()
        if not video_2:
            video_2 = Video(
                user_id=admin_user.user_id,
                original_url=url_2,
                duration=970.0,
                language="en",
                status=VideoStatus.DONE,
            )
            db.add(video_2)
            db.commit()
            db.refresh(video_2)
            print(f"SUCCESS: Seeded video: Al Gore (ID: {video_2.video_id})")

            # Seed Summary
            summary_2 = Summary(
                video_id=video_2.video_id,
                summary_text=(
                    "Cựu Phó Tổng thống Hoa Kỳ Al Gore trình bày các lập luận thuyết phục và giải pháp thực "
                    "tiễn nhằm ngăn chặn thảm họa biến đổi khí hậu toàn cầu. Ông giải thích cơ chế hiệu ứng "
                    "nhà kính do CO2 tích tụ, mô tả các hậu quả kinh tế xã hội và thiên tai gia tăng. Cuối cùng, "
                    "Al Gore kêu gọi hành động quyết liệt chuyển đổi sang năng lượng gió, mặt trời và thiết "
                    "lập thuế carbon, nhấn mạnh đây là nghĩa vụ đạo đức cấp bách của cả nhân loại."
                ),
                chapters_json=[
                    {
                        "title": "Khủng hoảng khí hậu & Tình trạng khẩn cấp",
                        "startTime": 0.0,
                        "endTime": 220.0,
                        "summary": "Khái quát hiện tượng nóng lên toàn cầu và sự cấp bách đối với sự sinh tồn của Trái Đất.",
                    },
                    {
                        "title": "Khí thải Carbon và tác động nhiệt",
                        "startTime": 220.0,
                        "endTime": 550.0,
                        "summary": "CO2 từ nhiên liệu hóa thạch giữ nhiệt trong khí quyển, gây hạn hán, bão lũ kỷ lục.",
                    },
                    {
                        "title": "Năng lượng tái tạo và Thuế Carbon",
                        "startTime": 550.0,
                        "endTime": 800.0,
                        "summary": "Trình bày giải pháp từ năng lượng sạch, nâng cao hiệu năng lưới điện và định giá khí thải carbon.",
                    },
                    {
                        "title": "Ý chí chính trị và Nghĩa vụ đạo đức",
                        "startTime": 800.0,
                        "endTime": 970.0,
                        "summary": "Chỉ ra rằng ý chí chính trị là tài nguyên vô hạn và kêu gọi hành động cứu lấy thế hệ tương lai.",
                    },
                ],
                keyframes_json=[
                    {
                        "timestamp": 150.0,
                        "imageUrl": "/static/mock_r2/keyframes/slide1.png",
                        "description": "Biểu đồ đường cong nhiệt độ trung bình toàn cầu tăng vọt",
                        "importanceScore": 0.90,
                    },
                    {
                        "timestamp": 480.0,
                        "imageUrl": "/static/mock_r2/keyframes/slide2.png",
                        "description": "Hình ảnh giải thích hiệu ứng nhà kính của bầu khí quyển",
                        "importanceScore": 0.85,
                    },
                    {
                        "timestamp": 720.0,
                        "imageUrl": "/static/mock_r2/keyframes/code1.png",
                        "description": "Mô hình trang trại điện gió ngoài khơi tiên tiến",
                        "importanceScore": 0.94,
                    },
                ],
                transcript_text=(
                    "Thank you so much. It's a brand new day. And I want to talk about how we can avert the climate crisis. "
                    "The climate crisis is not a political issue, it is a moral issue. We are facing a planetary emergency. "
                    "The evidence is clear: greenhouse gases are trapping heat in the atmosphere, leading to rising temperatures, "
                    "melting glaciers, and extreme weather. We have the solutions at hand: wind, solar, and geothermal. "
                    "We need to implement a carbon tax and mobilize political will starting today."
                ),
                model_used="WhisperX + Gemini 1.5 Flash",
                processing_time=9.8,
            )
            db.add(summary_2)

            # Seed Job
            job_2 = Job(
                video_id=video_2.video_id,
                job_type=JobType.SUMMARIZE,
                status=JobStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_log=None,
            )
            db.add(job_2)
            db.commit()

            # Seed Vector Database (ChromaDB)
            chunks_2 = [
                "Thank you so much. I want to talk about how we can avert the global climate crisis.",
                "The climate crisis is not a political issue, it is a moral and ethical issue for all humanity.",
                "Rising CO2 emissions trap solar radiation, causing extreme weather, drought, and ice melting.",
                "We can deploy solar power, wind turbines, and tax carbon to enforce cleaner energy transition.",
            ]
            metadatas_2 = [
                {"video_id": str(video_2.video_id)} for _ in chunks_2
            ]
            chromadb_service.add_transcript_chunks(
                str(video_2.video_id), chunks_2, metadatas_2
            )
            print(
                f"SUCCESS: Seeded vector chunks in ChromaDB for Video {video_2.video_id}"
            )
        else:
            print(f"INFO: Video Al Gore already exists in DB.")

        # Talk 3: Pending Task (to populate Celery / Admin dashboard list with active tasks)
        pending_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        pending_video = (
            db.query(Video).filter(Video.original_url == pending_url).first()
        )
        if not pending_video:
            pending_video = Video(
                user_id=normal_user.user_id,
                original_url=pending_url,
                duration=212.0,
                language="en",
                status=VideoStatus.PROCESSING,
            )
            db.add(pending_video)
            db.commit()
            db.refresh(pending_video)

            pending_job = Job(
                video_id=pending_video.video_id,
                job_type=JobType.SUMMARIZE,
                status=JobStatus.RUNNING,
                started_at=datetime.utcnow(),
                completed_at=None,
                error_log=None,
            )
            db.add(pending_job)
            db.commit()
            print(
                "SUCCESS: Seeded pending video & job task for queue dashboards."
            )
        else:
            print(f"INFO: Pending video task already exists in DB.")

        print("--- SEEDING COMPLETE SUCCESSFULLY ---")

    except Exception as e:
        db.rollback()
        print(f"ERROR: Seeding database failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
