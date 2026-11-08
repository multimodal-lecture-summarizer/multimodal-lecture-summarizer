import json

with open("storage/checkpoints/1b3580b4-56d7-4e94-b1a2-347a865a8d48.json", "r", encoding="utf-8") as f:
    ckpt = json.load(f)

print("Checkpoint keys:", ckpt.keys())
print("Stage:", ckpt.get("stage"))
print("Status:", ckpt.get("status"))
data = ckpt.get("data", {})
print("Data keys:", data.keys())
print("Chapters count:", len(data.get("chapters", [])))
print("Keyframes count:", len(data.get("keyframes", [])))
print("Video file url:", data.get("video_file_url"))
