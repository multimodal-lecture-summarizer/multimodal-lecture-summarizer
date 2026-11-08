import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'backend')
from app.core.database import SessionLocal
from app.models.video import Video
from app.models.summary import Summary

db = SessionLocal()
videos = db.query(Video).order_by(Video.uploaded_at.desc()).limit(3).all()
for v in videos:
    print('VIDEO:', v.video_id, '| Title:', repr(v.title), '| File:', repr(v.file_path), '| URL:', repr(v.original_url))
    s = db.query(Summary).filter(Summary.video_id == v.video_id).first()
    if s:
        print('  SUMMARY length:', repr(s.summary_length), '| words:', len((s.summary_text or '').split()))
        print('  SUMMARY text:', repr((s.summary_text or '')[:150]))
        print('  VARIANTS keys:', list((s.summary_variants_json or {}).keys()))
        for k, var in (s.summary_variants_json or {}).items():
            print(f'    Variant {k}: words={var.get("word_count")} model={var.get("model_used")}')
