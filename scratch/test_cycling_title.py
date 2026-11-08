import sys, json, re
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
)

db = SessionLocal()
video_id = '1b613e7c-8f83-4e77-a388-3a08d4c44455'
v = db.query(Video).filter(Video.video_id == video_id).first()
s = db.query(Summary).filter(Summary.video_id == video_id).first()

print(f"Current Video ID: {video_id}")
print(f"Current v.title in DB: {repr(v.title)}")
print("Chapters:")
for c in (s.chapters_json or []):
    print(" -", c.get("title"))

PRONOUN_SENTENCE_PATTERN = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|he|him|his|she|her|hers|they|them|their)\b|"
    r"\b(?:tôi|chúng tôi|chúng ta|mình|bạn|các bạn|anh ấy|cô ấy|họ)\b",
    re.IGNORECASE,
)

CONVERSATIONAL_OPENERS = re.compile(
    r"^(?:it\s+(?:inspires|is|seems|looks|feels|helps|keeps)|"
    r"i\s+(?:think|like|want|believe|guess|love|feel|have|am|get|know)|"
    r"we\s+(?:call|have|are|can|want|started|progressed|train)|"
    r"you\s+(?:can|know|see|want|might|have|are)|"
    r"they\s+(?:have|are|can|want|really)|"
    r"there\s+(?:is|are)|so\s+|and\s+|well\s+|just\s+|now\s+|basically\s+|actually\s+|for\s+me\s+|in\s+my\s+|to\s+me\s+)",
    re.IGNORECASE,
)

def is_conversational_quote(title: str) -> bool:
    if not title: return True
    t = title.strip()
    if PRONOUN_SENTENCE_PATTERN.search(t): return True
    if CONVERSATIONAL_OPENERS.match(t): return True
    if re.search(r"\b(?:as well|kind of|sort of|like this|you know|at all|really hit|teeth fit)\b", t, re.IGNORECASE): return True
    return False

print("is_conversational_quote('It inspires me as well'):", is_conversational_quote('It inspires me as well'))
print("is_conversational_quote('I think. Keep it in contact in Belgium'):", is_conversational_quote('I think. Keep it in contact in Belgium'))
print("is_conversational_quote('So just kind of roll your wrists like'):", is_conversational_quote('So just kind of roll your wrists like'))
print("is_conversational_quote('Dự án thành phố thân thiện với xe đạp'):", is_conversational_quote('Dự án thành phố thân thiện với xe đạp'))

def clean_chapter_topic(title: str) -> str:
    clean = re.sub(r"^(?:chương\s*\d+|phần\s*\d+|chapter\s*\d+)[\:\-—\s]*", "", title, flags=re.IGNORECASE).strip()
    clean = re.sub(
        r"^(?:giới\s*thiệu\s*(?:về)?|tìm\s*hiểu\s*(?:về)?|tổng\s*quan\s*(?:về)?|hướng\s*dẫn\s*(?:về)?|bài\s*học\s*(?:về)?|"
        r"introduction\s*(?:to)?|overview\s*(?:of)?|learning\s*about)\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    return clean

print("\nCleaned chapter 1:", repr(clean_chapter_topic(s.chapters_json[0]["title"])))

