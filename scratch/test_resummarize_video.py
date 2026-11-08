import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
sys.path.insert(0, 'backend')

from app.core.database import SessionLocal
from app.models.summary import Summary
from app.models.video import Video, VideoScene
from ai_workers.modules.summarization.summarizer import Summarizer, clean_video_title

db = SessionLocal()
video_id = '994b8439-38dc-4339-9598-e604783392e1'
video = db.query(Video).filter(Video.video_id == video_id).first()
summary = db.query(Summary).filter(Summary.video_id == video_id).first()

raw_transcript = summary.transcript_text or ""
utterances = json.loads(raw_transcript) if raw_transcript else []

db_scenes = db.query(VideoScene).filter(VideoScene.video_id == video_id).order_by(VideoScene.scene_index).all()
keyframes = []
for sc in db_scenes:
    keyframes.append({
        "start_seconds": sc.start_seconds,
        "end_seconds": sc.end_seconds,
        "timestamp": sc.start_seconds,
        "start_timecode": sc.start_timecode,
        "caption": sc.caption or "",
        "ocr_text": sc.script or "",
        "description": sc.caption or "",
        "importanceScore": 0.85,
        "imageUrl": sc.keyframe_url or "",
        "detected_objects": sc.detected_objects_json or {},
        "object_counts": sc.object_counts_json or {},
    })

chapters = summary.chapters_json or []

print(f"Testing Summarizer with length='detailed'...")
print(f"Input utterances: {len(utterances)}, keyframes: {len(keyframes)}, chapters: {len(chapters)}")
print(f"Current Video Title in DB: {repr(video.title)}")
print(f"Current Summary words in DB: {len((summary.summary_text or '').split())}")

summarizer = Summarizer()
result = summarizer.process(
    utterances=utterances,
    slides=keyframes,
    chapters=chapters,
    job_id=str(video_id),
    output_language="vi",
    summary_length="detailed",
)


print("\n--- RESULTS ---")
print("Generated video_title:", repr(result.get("video_title")))
summary_text = result.get("summary", "")
words = len(summary_text.split())
print("Generated summary word count:", words)
print("Summary preview (first 400 chars):")
print(summary_text[:400] + "...")
print("Total chapters returned:", len(result.get("chapters", [])))


