import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")
sys.path.insert(0, "backend")
from ai_workers.core.checkpoint import CheckpointManager
from ai_workers.modules.summarization.llm_client import LLMClient
from ai_workers.core.config import worker_settings

ckpt_mgr = CheckpointManager()
ckpt = ckpt_mgr.load_checkpoint('4ea966d0-a2ca-4c6b-bdd4-b82093a50e65', expected_stage='quality_complete')
data = ckpt['data']

lines = []
for u in data.get('utterances', [])[:10]:
    st = u.get('start', 0)
    tx = u.get('text', '')
    lines.append(f"[{st:.1f}s] {tx}")
transcript = "\n".join(lines)

prompt = f"""You are an expert video title and summarization specialist.
Analyze the transcript below and produce an authoritative, informative, and specific title in Vietnamese.

TRANSCRIPT:
{transcript}

CRITICAL RULES FOR "video_title":
1. SPECIFICITY IS MANDATORY: The title MUST explicitly name the specific core subject, skill, tool, sport, or technology taught/demonstrated in the video (e.g., if the video is about Parkour training with Charlotte Parkour, the title MUST contain 'Parkour' or 'Charlotte Parkour').
2. INFORMATIVE & CLEAR: Anyone reading the title must know immediately and precisely what the video covers.
3. NATURAL NOUN PHRASE FORMAT: 4 to 8 words in Vietnamese. E.g.: "Kỹ Thuật Luyện Tập Parkour Cơ Bản (Charlotte Parkour)" or "Hướng Dẫn Thực Hành Parkour & Vượt Chướng Ngại Vật".
4. STRICT PROHIBITION: NEVER generate vague, generic titles like "Tổng Quan Bài Học", "Kỹ Thuật Thực Hành", "Hướng Dẫn Kỹ Thuật", "Giới Thiệu Nội Dung", or "Thực Hành Cơ Bản". Titles without the specific subject name are completely unacceptable.

Return ONLY valid JSON:
{{"video_title": "Specific Natural Video Title"}}
"""

from ai_workers.modules.summarization.summarizer import Summarizer
summarizer = Summarizer()
client = summarizer.openrouter_client
raw_response = client.generate_chat_completion(
    model="qwen/qwen-2.5-7b-instruct",
    prompt=prompt,
    temperature=0.2,
    max_tokens=100,
    response_format={"type": "json_object"},
)
print("Response:", raw_response)
