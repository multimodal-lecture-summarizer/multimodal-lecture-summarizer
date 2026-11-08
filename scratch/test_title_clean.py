import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

from app.core.database import SessionLocal
from app.models.summary import Summary
from app.models.video import Video
from ai_workers.modules.summarization.summarizer import (
    clean_video_title,
    is_generic_title,
    extract_topic_from_chapters_or_takeaways,
    extract_core_topic_from_text,
)

db = SessionLocal()
video_id = '1b613e7c-8f83-4e77-a388-3a08d4c44455'
v = db.query(Video).filter(Video.video_id == video_id).first()
s = db.query(Summary).filter(Summary.video_id == video_id).first()

print("Current v.title:", repr(v.title))
print("is_generic_title(v.title):", is_generic_title(v.title))
print("chapters:", [c.get("title") for c in s.chapters_json or []])
print("takeaways:", [tk.get("title") for tk in s.key_takeaways_json or []])

extracted = extract_topic_from_chapters_or_takeaways(s.chapters_json, s.key_takeaways_json)
print("extracted from chapters/takeaways:", repr(extracted))

test_cases = [
    "STEVEN ERDMANCZYK | CHARLOTTE PARKOUR",
    "So just kind of roll your wrists like",
    "Phương pháp tập luyện Parkour",
    "Tổng quan bài giảng",
    "Giới thiệu về nhóm Charlotte Parkour",
]

for tc in test_cases:
    res = clean_video_title(
        raw_title=tc,
        summary_text=s.summary_text or "",
        lang_code="vi",
        chapters=s.chapters_json,
        takeaways=s.key_takeaways_json,
    )
    print(f"Input: {repr(tc):42} -> Output: {repr(res)}")

