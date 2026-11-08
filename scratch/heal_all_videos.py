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
)

db = SessionLocal()
videos = db.query(Video).all()

fix_map = {
    'Hướng dẫn kỹ thuật Bike Whip xe địa': 'Hướng dẫn kỹ thuật Bike Whip xe địa hình',
    'Street Violin Performance (Nghệ sĩ vĩ cầm đường': 'Street Violin Performance (Nghệ sĩ vĩ cầm đường phố)',
    'Bài học từ sự thất bại & Lòng': 'Bài học từ sự thất bại & Lòng kiên trì (TEDx)',
}

print("HEALING VIDEOS IN DATABASE...")
for v in videos:
    s = db.query(Summary).filter(Summary.video_id == v.video_id).first()
    old_title = v.title
    print(f"\nVideo ID: {v.video_id}")
    print(f"Old title: {repr(old_title)}")
    if v.title in fix_map:
        v.title = fix_map[v.title]
        print(f"  Fixed known chopped title -> {repr(v.title)}")
    elif s:
        extracted = extract_topic_from_chapters_or_takeaways(s.chapters_json, s.key_takeaways_json)
        if extracted and (
            is_generic_title(v.title)
            or v.title == 'Dự án thành phố thân thiện với xe'
            or (len(v.title or "") < len(extracted) and extracted.lower().startswith((v.title or "").lower()[:15]))
        ):
            print(f"  Overriding truncated/generic title with chapter topic: {repr(extracted)}")
            v.title = extracted
        else:
            new_title = clean_video_title(
                raw_title=v.title,
                summary_text=s.summary_text or "",
                lang_code=getattr(v, "language", "vi") or "vi",
                chapters=s.chapters_json or [],
                takeaways=s.key_takeaways_json or [],
            )
            print(f"Healed title: {repr(new_title)}")
            if new_title and not is_generic_title(new_title):
                v.title = new_title
        
        # Clean any bracketed markers like [Đoạn 1: ...] in summary_text
        import re
        if s.summary_text:
            cleaned_text = re.sub(r"\[(?:Đoạn|Paragraph)\s*\d+[^\]]*\]\s*", "", s.summary_text, flags=re.IGNORECASE).strip()
            if cleaned_text != s.summary_text:
                s.summary_text = cleaned_text
                print("  Cleaned bracketed paragraph headers from s.summary_text")

        variants = s.summary_variants_json or {}
        for k, var in variants.items():
            if isinstance(var, dict) and var.get("summary_text"):
                cleaned_var = re.sub(r"\[(?:Đoạn|Paragraph)\s*\d+[^\]]*\]\s*", "", var["summary_text"], flags=re.IGNORECASE).strip()
                var["summary_text"] = cleaned_var
        s.summary_variants_json = dict(variants)

db.commit()
print("\nAll database records committed successfully.")
