"""LLM Summarizer — LangChain, LLM Prompts, ChromaDB RAG.

Migrated from: src/mls/modules/text.py
NGƯỜI 3: LangChain, LLM Prompts, ChromaDB RAG
"""

from __future__ import annotations

import os
import json
import re
from typing import Any
from openai import OpenAI


class Summarizer:
    """Grounded LLM summarization with auto fallback models (OpenRouter -> Groq)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        from ai_workers.core.config import worker_settings
        self.openrouter_key = worker_settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        self.groq_key = worker_settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        
        # Primary API Key
        self.api_key = self.openrouter_key or self.groq_key

        self.openrouter_models = [
            worker_settings.OPENROUTER_MODEL or "qwen/qwen-2.5-7b-instruct",
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ]
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192"
        ]

    def build_rag_index(self, video_id: str, utterances: list[dict], slides: list[dict]) -> bool:
        """Build ChromaDB vector index from transcript and slide content."""
        try:
            from app.services.chromadb import chromadb_service
        except ImportError:
            import sys
            backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backend"))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            try:
                from app.services.chromadb import chromadb_service
            except Exception as e:
                print(f"[RAG Index] Could not import chromadb_service: {e}")
                return False

        def format_time(secs: float) -> str:
            m = int(secs) // 60
            s = int(secs) % 60
            return f"{m:02d}:{s:02d}"

        chunks = []
        metadatas = []

        if utterances:
            # Chunk speech utterances in ~25s windows to preserve granular timestamps
            window_size = 25.0
            max_time = float(utterances[-1].get("end", 0.0))
            curr_start = float(utterances[0].get("start", 0.0))
            
            while curr_start < max_time:
                curr_end = curr_start + window_size
                
                # Find all utterances starting or overlapping with [curr_start, curr_end]
                matched_utts = [
                    u for u in utterances
                    if max(float(u.get("start", 0.0)), curr_start) < min(float(u.get("end", 0.0)), curr_end)
                    or (curr_start <= float(u.get("start", 0.0)) < curr_end)
                ]
                
                speech_text = " ".join([u.get("text", "").strip() for u in matched_utts if u.get("text", "").strip()]).strip()
                
                if speech_text:
                    # Determine exact start second of first utterance in this chunk
                    actual_start = float(matched_utts[0].get("start", curr_start))
                    timecode = format_time(actual_start)
                    
                    doc_parts = [f"[{timecode}] Lời giảng: {speech_text}"]
                    
                    # Match any slides overlapping this time window
                    keyframe_url = ""
                    if slides:
                        for slide in slides:
                            s_start = float(slide.get("start_seconds", 0.0))
                            s_end = float(slide.get("end_seconds", 0.0))
                            if max(actual_start, s_start) < min(curr_end, s_end):
                                ocr = slide.get("ocr_text", "").strip()
                                cap = slide.get("caption", "").strip()
                                if ocr:
                                    doc_parts.append(f"Văn bản trên slide: {ocr}")
                                if cap and cap != "[Nhạc nền / Im lặng]":
                                    doc_parts.append(f"Mô tả hình ảnh: {cap}")
                                if not keyframe_url:
                                    keyframe_url = slide.get("keyframe_url", "")
                    
                    full_doc = " | ".join(doc_parts)
                    chunks.append(full_doc)
                    metadatas.append({
                        "video_id": str(video_id),
                        "start_seconds": actual_start,
                        "end_seconds": min(curr_end, max_time),
                        "timecode": timecode,
                        "keyframe_url": keyframe_url
                    })
                
                curr_start += window_size
        elif slides:
            # Fallback if no audio transcript available, chunk by slides
            for slide in slides:
                s_start = float(slide.get("start_seconds", 0.0))
                s_end = float(slide.get("end_seconds", 0.0))
                timecode = slide.get("start_timecode") or format_time(s_start)
                ocr_str = slide.get("ocr_text", "").strip()
                caption_str = slide.get("caption", "").strip()

                doc_parts = [f"[{timecode}]"]
                if ocr_str:
                    doc_parts.append(f"Văn bản trên slide: {ocr_str}")
                if caption_str and caption_str != "[Nhạc nền / Im lặng]":
                    doc_parts.append(f"Mô tả hình ảnh: {caption_str}")

                full_doc = " | ".join(doc_parts)
                if full_doc.strip():
                    chunks.append(full_doc)
                    metadatas.append({
                        "video_id": str(video_id),
                        "start_seconds": s_start,
                        "end_seconds": s_end,
                        "timecode": timecode,
                        "keyframe_url": slide.get("keyframe_url", "")
                    })
                    curr_start += window_size

        if chunks:
            print(f"[RAG Index] Building index for video {video_id} with {len(chunks)} multimodal chunks...")
            return chromadb_service.add_transcript_chunks(str(video_id), chunks, metadatas)
        return False

    def summarize(
        self,
        utterances: list[dict],
        slides: list[dict],
        chapters: list[dict],
    ) -> dict[str, Any]:
        """Generate grounded summary using LLM.

        Args:
            utterances: ASR transcript segments.
            slides: OCR + caption data per keyframe.
            chapters: Auto-detected chapter boundaries.

        Returns:
            Structured summary with chapters.
        """
        # Convert seconds to MM:SS format for better LLM context
        def format_time(secs: float) -> str:
            m = int(secs) // 60
            s = int(secs) % 60
            return f"{m:02d}:{s:02d}"

        transcript_lines = []
        for u in utterances:
            start_ts = format_time(u.get("start", 0))
            text = u.get("text", "").strip()
            if text:
                transcript_lines.append(f"[{start_ts}] {text}")
        
        transcript = "\n".join(transcript_lines)
        
        # Calculate max duration
        duration = utterances[-1]["end"] if utterances else (slides[-1]["end_seconds"] if slides else 0.0)
        formatted_duration = format_time(duration)
        # Format visual descriptions from slides
        visual_descriptions = []
        for slide in slides:
            # Handle both 'slides' format and 'keyframes' format
            ts = slide.get("start_seconds") if "start_seconds" in slide else slide.get("timestamp", 0.0)
            timecode = slide.get("start_timecode") or format_time(ts)
            caption = slide.get("caption") if "caption" in slide else slide.get("description", "")
            caption = caption.strip() if caption else ""
            ocr_text = slide.get("ocr_text", "").strip()

            
            desc_parts = []
            if caption and caption != "[Nhạc nền / Im lặng]":
                desc_parts.append(f"Image: {caption}")
            if ocr_text:
                desc_parts.append(f"Text on slide: {ocr_text}")
                
            if desc_parts:
                visual_descriptions.append(f"[{timecode}] " + " | ".join(desc_parts))
        visual_text = "\n".join(visual_descriptions)

        # Bail out only if BOTH transcript and visual data are empty
        if not transcript.strip() and not visual_text.strip():
            return {
                "video_title": "Video không có nội dung để tóm tắt",
                "summary": "No content (audio or visual) to summarize.",
                "chapters": []
            }

        # Fallback if API key is not configured
        if not self.api_key:
            duration = utterances[-1]["end"] if utterances else (slides[-1]["end_seconds"] if slides else 10.0)
            return {
                "video_title": "Video chưa đặt tên (Chưa cấu hình API Key)",
                "summary": "Nội dung video đã được trích xuất thành công. Vui lòng cấu hình OPENROUTER_API_KEY hoặc GROQ_API_KEY trong file .env để tạo tóm tắt AI chi tiết.",
                "chapters": [
                    {
                        "title": "Nội dung chính",
                        "startTime": 0.0,
                        "endTime": duration,
                        "summary": "Tổng hợp nội dung xuyên suốt video."
                    }
                ],
                "model_used": "None"
            }

        chapter_constraints = ""
        if chapters:
            chapter_intervals = "\n        ".join([f"Chapter {i+1}: {format_time(c['startTime'])} ({c['startTime']:.2f}s) -> {format_time(c['endTime'])} ({c['endTime']:.2f}s)" for i, c in enumerate(chapters)])
            chapter_constraints = f"""
        CRITICAL CONSTRAINTS FOR CHAPTERS:
        - We have pre-calculated the exact chapter boundaries using a deterministic algorithm.
        - You MUST use exactly these {len(chapters)} chapter intervals. Do NOT add, remove, or modify the timestamps.
        - Your ONLY task is to read the transcript within each specific interval and write a concise title and summary for it.
        - CRITICAL: In the summary text and any timestamp references, use the exact timecode [MM:SS] corresponding to each chapter or topic's actual start time (e.g., [00:00], [03:15], [07:30]). NEVER output [00:00] for all items!
        
        PREDEFINED CHAPTER BOUNDARIES:
        {chapter_intervals}
"""
        else:
            chapter_constraints = f"""
        CRITICAL CONSTRAINTS FOR CHAPTERS:
        - The total duration of this video is {duration} seconds ({formatted_duration}).
        - You MUST NOT generate any chapters with an endTime greater than {duration}.
        - The last chapter's endTime MUST be exactly {duration}.
        - Base your chapters strictly on the slide/scene transition timestamps and topic shifts in the transcript.
        - CRITICAL: In the summary text and any timestamp references, use the exact timecode [MM:SS] corresponding to each topic's actual start time. NEVER output [00:00] for all items!
"""

        prompt = f"""
        You are a professional AI assistant tasked with summarizing lecture or informational videos.
        Analyze the following audio transcript AND visual keyframe descriptions. 
        
        Before generating the final summary, you MUST perform a step-by-step analysis (Chain of Thought).
        1. Step 1: Map the speaker's main points to the visual text (OCR) and images shown on screen.
        2. Step 2: Identify the core topics and how they transition over time.
        
        After your analysis, generate:
        1. A detailed video summary in plain text format (around 300-500 words).
        2. A concise, descriptive title for this video (video_title - 5-10 words, in the same language as the content).
        3. Logical chapters for the video.
{chapter_constraints}
        Audio Transcript (with timestamps):
        {transcript if transcript.strip() else "[No speech detected]"}

        Visual Context (Descriptions & OCR of key scenes at specific timestamps):
        {visual_text if visual_text else "[No visual descriptions available]"}

        You must return ONLY valid JSON.
        
        Required schema:
        {{
            "analysis": "Keep this concise (max 150 words). Step-by-step reasoning...",
            "video_title": "Concise Descriptive Video Title Here",
            "summary": "Plain text summary content here...",
            "chapters": [
                {{
                    "title": "Chapter title",
                    "startTime": 0.0,
                    "endTime": 80.0,
                    "summary": "Chapter summary"
                }}
            ]
        }}
        
        Rules:
        - The chapters array must contain exactly {len(chapters) if chapters else 1} items.
        - Preserve the supplied chapter start and end times.
        - Do not add markdown fences like ```json.
        - Do not add commentary before or after the JSON.
        - Every chapter must have a non-empty title and summary.
        """

        # Build candidate client & model target list
        targets = []
        if self.openrouter_key:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/multimodal-lecture-summarizer",
                    "X-Title": "Multimodal Lecture Summarizer"
                }
            )
            for m in self.openrouter_models:
                targets.append(("OpenRouter", m, or_client))

        if self.groq_key:
            groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_key
            )
            for m in self.groq_models:
                targets.append(("Groq", m, groq_client))

        last_error = None
        for provider, model, client in targets:
            print(f"Trying {provider} summarization with model: {model}...")
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                response_text = chat_completion.choices[0].message.content.strip()
                
                print(f"[LLM RAW RESPONSE]\n{response_text}")
                
                # Clean up markdown codeblocks if LLM wraps the JSON
                if response_text.startswith("```"):
                    response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                    response_text = re.sub(r"\n```$", "", response_text)
                    
                data = json.loads(response_text)
                
                print(f"[LLM PARSED] type={type(data).__name__} keys={list(data.keys()) if isinstance(data, dict) else None}")
                
                # Validation
                if not isinstance(data, dict):
                    raise ValueError("LLM response is not a JSON object")

                video_title = data.get("video_title") or data.get("title")
                llm_chapters = data.get("chapters", [])
                
                if not isinstance(video_title, str) or not video_title.strip():
                    raise ValueError("Missing or empty video_title")

                if not isinstance(llm_chapters, list):
                    raise ValueError("chapters must be a list")

                print(f"[LLM CHAPTERS] expected={len(chapters) if chapters else 1} actual={len(llm_chapters)}")

                if chapters:
                    if len(llm_chapters) != len(chapters):
                        raise ValueError(f"Expected {len(chapters)} chapters, got {len(llm_chapters)}")
                        
                    for index, chapter in enumerate(llm_chapters):
                        if not isinstance(chapter, dict):
                            raise ValueError(f"Chapter {index} is not an object")
                        if not chapter.get("title"):
                            raise ValueError(f"Chapter {index} missing title")
                        if not chapter.get("summary"):
                            raise ValueError(f"Chapter {index} missing summary")
                        
                    validated_chapters = []
                    for i, orig_c in enumerate(chapters):
                        validated_chapters.append({
                            "title": llm_chapters[i]["title"],
                            "startTime": orig_c["startTime"],
                            "endTime": orig_c["endTime"],
                            "summary": llm_chapters[i]["summary"]
                        })
                    data["chapters"] = validated_chapters
                
                print(f"[OK] Successfully generated summary with {provider} ({model})")
                return {
                    "video_title": data.get("video_title") or data.get("title") or "Bài giảng chưa đặt tên",
                    "summary": data.get("summary", "Tóm tắt bài giảng."),
                    "chapters": data.get("chapters", []),
                    "model_used": f"{provider} ({model})"
                }
            except Exception as e:
                print(f"{provider} model {model} failed: {e}. Trying next fallback target...")
                last_error = e

        # Fallback if all models fail
        print("All LLM providers/models failed. Using offline fallback summary.")
        duration = utterances[-1]["end"] if utterances else 10.0
        return {
            "video_title": "Bài giảng chưa đặt tên (Lỗi AI)",
            "summary": f"### Tóm tắt bài giảng\n\nNội dung bài giảng đã dịch thành công. (Gặp lỗi khi tạo tóm tắt bằng AI: {last_error})",
            "chapters": [
                {
                    "title": "Nội dung chính bài học",
                    "startTime": 0.0,
                    "endTime": duration,
                    "summary": "Tổng hợp nội dung đã nói trong toàn bộ bài học."
                }
            ],
            "model_used": f"Offline Fallback (Error: {last_error})"
        }

    def answer_question(self, question: str, context_chunks: list[str]) -> dict[str, Any]:
        """RAG-based Q&A: retrieve relevant chunks and generate answer."""
        return {"answer": "", "references": [], "confidence": 0.0}

    def process(self, utterances: list[dict], slides: list[dict], chapters: list[dict]) -> dict[str, Any]:
        """Full text pipeline: build index → summarize."""
        result = self.summarize(utterances, slides, chapters)
        return result
