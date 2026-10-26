import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

from app.core.database import SessionLocal
from app.models.summary import Summary
from app.models.video import Video
from ai_workers.modules.summarization.summarizer import clean_video_title, Summarizer

db = SessionLocal()
video_id = '994b8439-38dc-4339-9598-e604783392e1'
v = db.query(Video).filter(Video.video_id == video_id).first()
s = db.query(Summary).filter(Summary.video_id == video_id).first()

if not v or not s:
    print("Video or summary not found!")
    sys.exit(1)

print(f"Old video.title: {repr(v.title)}")
healed_title = clean_video_title(
    raw_title=v.title,
    summary_text=s.summary_text or "",
    lang_code="vi",
    chapters=s.chapters_json,
    takeaways=s.key_takeaways_json,
)
v.title = healed_title
print(f"Healed video.title: {repr(v.title)}")

# Check summary variants
variants = s.summary_variants_json or {}
detailed_variant = variants.get("detailed")
if detailed_variant:
    old_words = detailed_variant.get("word_count", len((detailed_variant.get("summary_text", "")).split()))
    print(f"Old detailed variant words: {old_words}")

# Invalidate stale short detailed variant so on-demand re-summarization or cache is fresh
if detailed_variant and old_words < 300:
    del variants["detailed"]
    s.summary_variants_json = dict(variants)
    print("Removed stale <300 words detailed variant from cache.")

db.commit()
print("Database updated successfully.")
