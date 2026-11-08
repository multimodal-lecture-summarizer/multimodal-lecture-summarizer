import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "backend")
from app.core.database import SessionLocal
from app.models.video import Video
from app.models.summary import Summary

db = SessionLocal()
videos = db.query(Video).order_by(Video.uploaded_at.desc()).all()
print(f"Total videos: {len(videos)}")
for v in videos[:10]:
    print(f"\nID: {v.video_id} | Title: {v.title} | Duration: {v.duration}")
    s = db.query(Summary).filter(Summary.video_id == v.video_id).first()
    if s:
        print(f"   Takeaways count: {len(s.key_takeaways_json or [])}")
        for tk in (s.key_takeaways_json or []):
            if isinstance(tk, dict):
                print(f"     - [{tk.get('time_str') or tk.get('timestamp')}] {tk.get('title')}: {tk.get('description', '')[:70]}")
            else:
                print(f"     - {tk}")
db.close()
