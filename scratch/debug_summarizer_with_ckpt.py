import json
import sys
sys.path.insert(0, ".")
from ai_workers.modules.summarization.summarizer import Summarizer

with open("storage/checkpoints/1b3580b4-56d7-4e94-b1a2-347a865a8d48.json", "r", encoding="utf-8") as f:
    ckpt = json.load(f)

data = ckpt["data"]
utterances = data["utterances"]
keyframes = data["keyframes"]
chapters = data["chapters"]

print(f"Utterances: {len(utterances)}, Keyframes: {len(keyframes)}, Chapters: {len(chapters)}")

summarizer = Summarizer()
res = summarizer.process(utterances, keyframes, chapters, job_id="debug-run")

sys.stdout.reconfigure(encoding="utf-8")
print("\n--- RESULT ---")
print("Video Title:", res.get("video_title"))
print("Summary:", repr(res.get("summary")))
print("Chapters:", json.dumps(res.get("chapters"), ensure_ascii=False, indent=2))
print("Key Takeaways:", json.dumps(res.get("key_takeaways"), ensure_ascii=False, indent=2))
print("Model Used:", res.get("model_used"))
print("Fallback Used:", res.get("fallback_used"))
