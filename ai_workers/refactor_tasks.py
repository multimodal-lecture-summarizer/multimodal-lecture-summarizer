import re

path = r"d:\Document\Code\Exercise\Workspace\multimodal-lecture-summarizer\ai_workers\tasks.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove R2 upload block
r2_pattern = r"        # Upload keyframes to R2 if configured.*?log_step\(\"Đã tải xong toàn bộ slide lên R2\.\", \"storage\", 74\)\n"
content = re.sub(r2_pattern, "", content, flags=re.DOTALL)

# 2. Remove RAG index block
rag_pattern = r"        # Build Multimodal RAG vector index in ChromaDB.*?log_step\(\"Lỗi khi tạo chỉ mục RAG\.\", \"rag\", 98\)\n"
content = re.sub(rag_pattern, "", content, flags=re.DOTALL)

# 3. Add `build_rag_index` task at the end of the file
new_task = """
@app.task(bind=True, name="ai_workers.build_rag_index", max_retries=3, autoretry_for=(Exception,), retry_backoff=True)
def build_rag_index(self, video_id: str):
    import sys
    import os
    import json
    
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
        
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        from app.models.summary import Summary
        from app.models.video import Video
        from app.core.constants import RagStatus
        from ai_workers.modules.fusion.summarizer import Summarizer
        
        summary = db.query(Summary).filter(Summary.video_id == video_id).first()
        if not summary:
            raise ValueError(f"No summary found for video {video_id}")
            
        utterances = []
        if summary.transcript_text:
            try:
                utterances = json.loads(summary.transcript_text)
            except Exception:
                pass
                
        slides = summary.keyframes_json or []
        
        summarizer = Summarizer()
        rag_success = summarizer.build_rag_index(video_id, utterances, slides)
        
        if rag_success:
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.READY})
        else:
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.FAILED})
        db.commit()
    except Exception as e:
        if self.request.retries >= self.max_retries:
            from app.models.video import Video
            from app.core.constants import RagStatus
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.FAILED})
            db.commit()
        raise e
    finally:
        db.close()
"""
if "def build_rag_index" not in content:
    content += new_task

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Removed R2 and RAG blocks. Added build_rag_index.")
