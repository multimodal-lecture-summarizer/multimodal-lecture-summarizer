"""LLM Summarizer Service — Coordinates prompt formulation, RAG vector indexing, and resilient multi-provider LLM summarization."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from ai_workers.modules.summarization.errors import (
    LLMBaseError,
    LLMResponseParsingError,
    map_exception_to_llm_error,
)
from ai_workers.modules.summarization.llm_client import LLMClient

logger = logging.getLogger("ai_workers.summarization.summarizer")


class Summarizer:
    """Grounded LLM summarization with multi-provider fallback (OpenRouter -> Groq) and Tenacity resilience."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        llm_timeout: float = 20.0,
        max_attempts: int = 3,
        custom_wait: Any = None,
        overall_timeout: float = 90.0,
    ):
        self.config = config or {}
        self.overall_timeout = float(overall_timeout)
        from ai_workers.core.config import worker_settings

        self.openrouter_key = worker_settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        self.groq_key = worker_settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
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
            "llama3-70b-8192",
        ]

        # Initialize dedicated LLM clients with resilience settings
        self.openrouter_client = LLMClient(
            provider="OpenRouter",
            api_key=self.openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=llm_timeout,
            max_attempts=max_attempts,
            custom_wait=custom_wait,
            default_headers={
                "HTTP-Referer": "https://github.com/multimodal-lecture-summarizer",
                "X-Title": "Multimodal Lecture Summarizer",
            },
        )

        self.groq_client = LLMClient(
            provider="Groq",
            api_key=self.groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=llm_timeout,
            max_attempts=max_attempts,
            custom_wait=custom_wait,
        )

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
                logger.error(f"[RAG Index] Could not import chromadb_service: {e}")
                return False

        def format_time(secs: float) -> str:
            m = int(secs) // 60
            s = int(secs) % 60
            return f"{m:02d}:{s:02d}"

        chunks = []
        metadatas = []

        if utterances:
            window_size = 25.0
            max_time = float(utterances[-1].get("end", 0.0))
            curr_start = float(utterances[0].get("start", 0.0))

            while curr_start < max_time:
                curr_end = curr_start + window_size
                matched_utts = [
                    u for u in utterances
                    if max(float(u.get("start", 0.0)), curr_start) < min(float(u.get("end", 0.0)), curr_end)
                    or (curr_start <= float(u.get("start", 0.0)) < curr_end)
                ]

                speech_text = " ".join([u.get("text", "").strip() for u in matched_utts if u.get("text", "").strip()]).strip()

                if speech_text:
                    actual_start = float(matched_utts[0].get("start", curr_start))
                    timecode = format_time(actual_start)

                    doc_parts = [f"[{timecode}] Lời giảng: {speech_text}"]

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
                        "keyframe_url": keyframe_url,
                    })

                curr_start += window_size
        elif slides:
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
                        "keyframe_url": slide.get("keyframe_url", ""),
                    })

        if chunks:
            logger.info(f"[RAG Index] Building index for video {video_id} with {len(chunks)} multimodal chunks...")
            return chromadb_service.add_transcript_chunks(str(video_id), chunks, metadatas)
        return False

    def summarize(
        self,
        utterances: list[dict],
        slides: list[dict],
        chapters: list[dict],
        job_id: str = "unknown",
    ) -> dict[str, Any]:
        """Generate grounded summary using LLM with resilience layer.

        Args:
            utterances: ASR transcript segments.
            slides: OCR + caption data per keyframe.
            chapters: Auto-detected chapter boundaries.
            job_id: Celery / pipeline job identifier.

        Returns:
            Structured summary with chapters and model_used.
        """
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

        duration = utterances[-1]["end"] if utterances else (slides[-1]["end_seconds"] if slides else 0.0)
        formatted_duration = format_time(duration)

        visual_descriptions = []
        for slide in slides:
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
                "chapters": [],
                "model_used": "None",
            }

        # Fallback if neither API key is configured
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
                        "summary": "Tổng hợp nội dung xuyên suốt video.",
                    }
                ],
                "model_used": "None",
            }

        chapter_constraints = ""
        if chapters:
            chapter_intervals = "\n        ".join([
                f"Chapter {i+1}: {format_time(c['startTime'])} ({c['startTime']:.2f}s) -> {format_time(c['endTime'])} ({c['endTime']:.2f}s)"
                for i, c in enumerate(chapters)
            ])
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
        targets: list[tuple[str, str, LLMClient]] = []
        if self.openrouter_client.is_configured:
            for m in self.openrouter_models:
                targets.append(("OpenRouter", m, self.openrouter_client))

        if self.groq_client.is_configured:
            for m in self.groq_models:
                targets.append(("Groq", m, self.groq_client))

        last_error: Optional[LLMBaseError] = None
        import time
        overall_start_time = time.time()

        for provider, model, client in targets:
            elapsed_total = time.time() - overall_start_time
            if elapsed_total >= self.overall_timeout:
                logger.warning(
                    f"Overall summarization budget of {self.overall_timeout:.1f}s exceeded "
                    f"after {elapsed_total:.2f}s (job_id={job_id}). Stopping remaining candidate attempts."
                )
                if last_error is None:
                    from ai_workers.modules.summarization.errors import LLMTimeoutError
                    last_error = LLMTimeoutError(
                        message=f"Overall summarization budget ({self.overall_timeout:.1f}s) exceeded.",
                        provider=provider,
                        model=model,
                    )
                break

            print(f"Trying {provider} summarization with model: {model} (job_id={job_id}, elapsed={elapsed_total:.1f}s/{self.overall_timeout:.1f}s)...")
            try:
                raw_response = client.generate_chat_completion(
                    model=model,
                    prompt=prompt,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    job_id=job_id,
                )

                # Clean up markdown codeblocks if LLM wraps the JSON
                response_text = raw_response
                if response_text.startswith("```"):
                    response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                    response_text = re.sub(r"\n```$", "", response_text)

                try:
                    data = json.loads(response_text)
                except Exception as json_err:
                    raise LLMResponseParsingError(
                        message=f"Failed to parse LLM JSON: {json_err}",
                        provider=provider,
                        model=model,
                        raw_error=json_err,
                    )

                if not isinstance(data, dict):
                    raise LLMResponseParsingError(
                        message="LLM response is not a JSON object",
                        provider=provider,
                        model=model,
                    )

                video_title = data.get("video_title") or data.get("title")
                llm_chapters = data.get("chapters", [])

                if not isinstance(video_title, str) or not video_title.strip():
                    raise LLMResponseParsingError(
                        message="Missing or empty video_title in LLM response",
                        provider=provider,
                        model=model,
                    )

                if not isinstance(llm_chapters, list):
                    raise LLMResponseParsingError(
                        message="chapters must be a list in LLM response",
                        provider=provider,
                        model=model,
                    )

                if chapters:
                    if len(llm_chapters) != len(chapters):
                        raise LLMResponseParsingError(
                            message=f"Expected {len(chapters)} chapters, got {len(llm_chapters)}",
                            provider=provider,
                            model=model,
                        )

                    for index, chapter in enumerate(llm_chapters):
                        if not isinstance(chapter, dict):
                            raise LLMResponseParsingError(
                                message=f"Chapter {index} is not an object",
                                provider=provider,
                                model=model,
                            )
                        if not chapter.get("title"):
                            raise LLMResponseParsingError(
                                message=f"Chapter {index} missing title",
                                provider=provider,
                                model=model,
                            )
                        if not chapter.get("summary"):
                            raise LLMResponseParsingError(
                                message=f"Chapter {index} missing summary",
                                provider=provider,
                                model=model,
                            )

                    validated_chapters = []
                    for i, orig_c in enumerate(chapters):
                        validated_chapters.append({
                            "title": llm_chapters[i]["title"],
                            "startTime": orig_c["startTime"],
                            "endTime": orig_c["endTime"],
                            "summary": llm_chapters[i]["summary"],
                        })
                    data["chapters"] = validated_chapters

                print(f"[OK] Successfully generated summary with {provider} ({model})")
                return {
                    "video_title": data.get("video_title") or data.get("title") or "Bài giảng chưa đặt tên",
                    "summary": data.get("summary", "Tóm tắt bài giảng."),
                    "key_takeaways": data.get("key_takeaways", []),
                    "chapters": data.get("chapters", []),
                    "model_used": f"{provider} ({model})",
                    "fallback_used": False,
                    "summary_method": "llm",
                }

            except Exception as e:
                llm_err = map_exception_to_llm_error(e, provider=provider, model=model)
                logger.warning(f"{provider} model {model} failed: {llm_err}. Trying next fallback target...")
                last_error = llm_err

        # Extractive fallback if all providers/models fail
        logger.error(f"All LLM providers/models failed for job_id={job_id}. Last error: {last_error}. Activating Extractive Fallback...")
        from ai_workers.modules.summarization.extractive_fallback import ExtractiveSummarizer
        extractive = ExtractiveSummarizer()
        return extractive.generate_fallback(
            utterances=utterances,
            slides=slides,
            chapters=chapters,
            llm_error=last_error,
            job_id=job_id,
        )

    def answer_question(self, question: str, context_chunks: list[str]) -> dict[str, Any]:
        """RAG-based Q&A: retrieve relevant chunks and generate answer."""
        return {"answer": "", "references": [], "confidence": 0.0}

    def process(
        self,
        utterances: list[dict],
        slides: list[dict],
        chapters: list[dict],
        job_id: str = "unknown",
    ) -> dict[str, Any]:
        """Full text pipeline: build index -> summarize."""
        return self.summarize(utterances, slides, chapters, job_id=job_id)
