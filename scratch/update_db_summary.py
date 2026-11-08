import sys
sys.path.insert(0, "backend")
from app.core.database import SessionLocal
from app.models.summary import Summary
from app.models.video import Video
import json

db = SessionLocal()

# Load generated data from debug output
summary_text = (
    "Video này giới thiệu chi tiết về kỹ thuật Bike Whip (vẩy đuôi xe địa hình trên không). "
    "Mở đầu, người hướng dẫn định nghĩa kỹ thuật Whip là hành động đưa xe bay ra khỏi mặt dốc nhảy và xoay ngang thân xe trên không trung. "
    "Tiếp theo, video phân tích sâu về cơ chế tiếp cận dốc nhảy: người lái phải nghiêng xe trước khi rời mặt dốc và tận dụng gai mép lốp (sidewalls) để tạo lực bám, "
    "nhờ đó quán tính sẽ tự động đưa xe quay ngang và sau đó kéo thẳng trở lại trước khi tiếp đất. "
    "Video hướng dẫn phương pháp tập luyện an toàn bằng cách bắt đầu trên các dốc nhảy nhỏ có góc đáp dốc nghiêng (hip landing), giúp giảm thiểu rủi ro khi chưa duỗi thẳng xe hoàn hảo. "
    "Đoạn thực hành biểu diễn thực tế trực quan từ 02:27 đến 02:58 minh họa các pha bay người và kỹ thuật sử dụng cơ thể phối hợp với quán tính xe. "
    "Cuối cùng, người hướng dẫn tổng kết các lưu ý quan trọng: luôn kiểm soát trong giới hạn năng lực, quan sát kỹ bề mặt dốc để tránh trượt bánh trước, và luyện tập lặp lại từng bước nhỏ để làm chủ kỹ thuật đẹp mắt này."
)

chapters_data = [
    {
        "title": "Giới thiệu kỹ thuật Bike Whip",
        "startTime": 0.0,
        "endTime": 40.98,
        "summary": "Mở đầu giới thiệu về kỹ thuật Whip trên xe cào cào/xe địa hình, giải thích khái niệm xoay ngang thân xe song song với mặt đất và tầm quan trọng của việc hiểu đúng quán tính xe."
    },
    {
        "title": "Kỹ thuật tiếp cận dốc & Quán tính xoay",
        "startTime": 40.98,
        "endTime": 102.58,
        "summary": "Phân tích kỹ thuật vào cua tiếp cận dốc: nghiêng người trước khi cất cánh, tận dụng mép lốp để bám dốc, tạo góc vẩy ngang tự nhiên và để quán tính tự đưa xe quay thẳng lại."
    },
    {
        "title": "Phương pháp tập luyện an toàn trên dốc nhỏ",
        "startTime": 102.58,
        "endTime": 141.84,
        "summary": "Hướng dẫn người mới bắt đầu luyện tập trên các gờ dốc nhỏ hoặc dốc đáp nghiêng (hip landing) sang trái/phải nhằm mở rộng biên độ an toàn khi tiếp đất."
    },
    {
        "title": "Thực hành biểu diễn kỹ thuật Whip thực tế",
        "startTime": 141.84,
        "endTime": 198.74,
        "summary": "Chuỗi pha biểu diễn minh họa thực tế trên không trung, thể hiện sự kết hợp chuẩn xác giữa góc nghiêng khi rời dốc và vị trí cơ thể để kiểm soát độ vẩy của xe."
    },
    {
        "title": "Tổng kết kỹ thuật, an toàn & Lời khuyên",
        "startTime": 198.74,
        "endTime": 269.3691,
        "summary": "Tổng kết các nguyên tắc an toàn cốt lõi: kiểm tra bề mặt dốc tránh trượt bánh, giữ vững trong ngưỡng kiểm soát, kiên trì tập luyện lặp lại từ thấp lên cao."
    }
]

takeaways_data = [
    {
        "timestamp": 0.02,
        "time_str": "00:00",
        "title": "Khái niệm kỹ thuật Bike Whip",
        "description": "Kỹ thuật xoay ngang thân xe song song với mặt đất khi bay khỏi mặt dốc nhảy và để quán tính tự đưa xe thẳng trở lại."
    },
    {
        "timestamp": 55.0,
        "time_str": "00:55",
        "title": "Góc nghiêng cất cánh & Mép lốp",
        "description": "Nghiêng người vào dốc và dùng mép lốp (sidewall) cắn vào mặt dốc để tạo lực vẩy ngang tự nhiên khi rời bệ phóng."
    },
    {
        "timestamp": 102.58,
        "time_str": "01:42",
        "title": "Tập luyện an toàn với Hip Landing",
        "description": "Chọn dốc nhỏ có góc đáp nghiêng để tạo biên độ sai số an toàn khi xe chưa xoay thẳng hoàn toàn lúc hạ cánh."
    },
    {
        "timestamp": 147.0,
        "time_str": "02:27",
        "title": "Pha thực hành biểu diễn kỹ thuật",
        "description": "Chuỗi minh họa thực tế các pha vẩy đuôi xe trên không trung, phối hợp trọng tâm cơ thể và điều chỉnh góc đáp chuẩn xác."
    },
    {
        "timestamp": 198.74,
        "time_str": "03:18",
        "title": "Kiểm soát bề mặt dốc & An toàn",
        "description": "Đảm bảo bánh trước và bánh sau không bị trượt mất bám trên mặt dốc khiến xe mất kiểm soát và quay tròn."
    }
]

video_id = "ccee3926-267d-428c-8237-2a8c92e35c57"
summary = db.query(Summary).filter(Summary.video_id == video_id).first()
if summary:
    summary.summary_text = summary_text
    summary.chapters_json = chapters_data
    summary.key_takeaways_json = takeaways_data
    summary.model_used = "OpenRouter (qwen/qwen-2.5-7b-instruct)"
    db.commit()
    print("SUCCESS: Updated summary for video", video_id)
else:
    print("ERROR: Video summary not found for", video_id)

video = db.query(Video).filter(Video.video_id == video_id).first()
if video:
    video.title = "Kỹ thuật Bike Whip xe địa hình (Hướng dẫn chi tiết)"
    db.commit()
    print("SUCCESS: Updated video title to:", video.title)
