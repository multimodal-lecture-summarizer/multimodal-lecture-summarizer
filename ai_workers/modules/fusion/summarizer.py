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
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "llama-3.3-70b-versatile",
            "llama3-8b-8192"
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
        transcript = " ".join([u.get("text", "") for u in utterances])
        if not transcript.strip():
            return {
                "summary": "No lecture content to summarize.",
                "chapters": []
            }

        # Fallback if API key is not configured
        if not self.api_key:
            duration = utterances[-1]["end"] if utterances else 10.0
            return {
                "summary": "Lecture Summary\n\nLecture transcribed successfully. Please configure GROQ_API_KEY in the .env file to generate a detailed AI summary.",
                "chapters": [
                    {
                        "title": "Main Lecture Content",
                        "startTime": 0.0,
                        "endTime": duration,
                        "summary": "Summary of the content spoken throughout the lecture."
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
        You are a professional AI assistant tasked with summarizing lecture videos.
        Analyze the following transcript to:
        1. Create a detailed lecture summary in plain text format (do NOT use markdown syntax like #, ##, **, -, *, etc. Use standard paragraphs and clear spacing, around 300-500 words).
        2. Automatically divide the lecture into logical chapters according to timestamps. Each chapter must contain:
           - Chapter title (title)
           - Start time (startTime - in seconds, as float/int)
           - End time (endTime - in seconds, as float/int)
           - Brief chapter summary (summary - about 1-2 sentences)

        Lecture Transcript:
        {transcript}

        Please return the result in the following STRICT JSON format (with no other explanations outside the JSON):
        {{
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
