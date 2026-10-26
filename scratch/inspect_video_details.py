import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'backend')
from app.core.database import SessionLocal
from app.models.summary import Summary
from app.models.video import VideoScene

db = SessionLocal()
s = db.query(Summary).filter(Summary.video_id == '994b8439-38dc-4339-9598-e604783392e1').first()
scenes = db.query(VideoScene).filter(VideoScene.video_id == '994b8439-38dc-4339-9598-e604783392e1').order_by(VideoScene.scene_index).all()
print('Number of scenes:', len(scenes))
for sc in scenes[:5]:
    print(f'Scene {sc.scene_index} [{sc.start_timecode}-{sc.end_timecode}]: script={repr(sc.script)} caption={repr(sc.caption)}')
parsed = json.loads(s.transcript_text) if s.transcript_text else []
print('Transcript segments count:', len(parsed))
for p in parsed[:8]:
    print(f"  [{p.get('start')}s]: {p.get('text')}")
