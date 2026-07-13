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
    """Grounded LLM summarization with auto fallback models."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        # Get primary API Key (Groq)
        from ai_workers.core.config import worker_settings
        self.api_key = worker_settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        # Fallback models in priority order
        self.models_list = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192"
        ]

    def build_rag_index(self, utterances: list[dict], slides: list[dict]) -> None:
        """Build ChromaDB vector index from transcript and slide content."""
        pass

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
            timecode = slide.get("start_timecode", "00:00:00")
            caption = slide.get("caption", "").strip()
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
                "summary": "Nội dung video đã được trích xuất thành công. Vui lòng cấu hình GROQ_API_KEY trong file .env để tạo tóm tắt AI chi tiết.",
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

        # Initialize Groq client
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

        prompt = f"""
        You are a professional AI assistant tasked with summarizing lecture or informational videos.
        Analyze the following audio transcript AND visual keyframe descriptions. 
        
        Before generating the final summary, you MUST perform a step-by-step analysis (Chain of Thought).
        1. Step 1: Map the speaker's main points to the visual text (OCR) and images shown on screen.
        2. Step 2: Identify the core topics and how they transition over time.
        
        After your analysis, generate:
        1. A detailed video summary in plain text format (around 300-500 words).
        2. A concise, descriptive title for this video (video_title - 5-10 words, in the same language as the content).
        3. Logical chapters segmented based on slide changes (Visual Context timestamps) and semantic topic transitions. Each chapter must contain:
           - Chapter title (title)
           - Start time (startTime - in seconds)
           - End time (endTime - in seconds)
           - Brief chapter summary (summary - 1-2 sentences)

        CRITICAL CONSTRAINTS FOR CHAPTERS:
        - The total duration of this video is {duration} seconds ({formatted_duration}).
        - You MUST NOT generate any chapters with an endTime greater than {duration}.
        - The last chapter's endTime MUST be exactly {duration}.
        - Base your chapters strictly on the slide/scene transition timestamps and topic shifts in the transcript. Do not invent non-existent topics for silent parts.

        Audio Transcript (with timestamps):
        {transcript if transcript.strip() else "[No speech detected]"}

        Visual Context (Descriptions & OCR of key scenes at specific timestamps):
        {visual_text if visual_text else "[No visual descriptions available]"}

        Please return the result in the following STRICT JSON format:
        {{
            "analysis": "Keep this concise (max 150 words). Step-by-step reasoning and mapping of audio to visual concepts...",
            "video_title": "Concise Descriptive Video Title Here",
            "summary": "Plain text summary content here...",
            "chapters": [
                {{
                    "title": "Chapter 1 Title",
                    "startTime": 0.0,
                    "endTime": 60.0,
                    "summary": "Brief summary of chapter 1..."
                }}
            ]
        }}
        """

        last_error = None
        for model in self.models_list:
            print(f"Trying Groq summarization with model: {model}...")
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                response_text = chat_completion.choices[0].message.content.strip()
                
                # Clean up markdown codeblocks if LLM wraps the JSON
                if response_text.startswith("```"):
                    response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                    response_text = re.sub(r"\n```$", "", response_text)
                    
                data = json.loads(response_text)
                print(f"[OK] Successfully generated summary with model: {model}")
                return {
                    "video_title": data.get("video_title") or data.get("title") or "Bài giảng chưa đặt tên",
                    "summary": data.get("summary", "Tóm tắt bài giảng."),
                    "chapters": data.get("chapters", []),
                    "model_used": f"Groq ({model})"
                }
            except Exception as e:
                print(f"⚠️ Model {model} failed: {e}. Trying fallback model...")
                last_error = e

        # Fallback if all models fail
        print("❌ All Groq models failed. Using offline fallback summary.")
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
