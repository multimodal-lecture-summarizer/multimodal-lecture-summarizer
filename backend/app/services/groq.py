import logging
import requests
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled = bool(self.api_key and not self.api_key.startswith("gsk_your"))

        if self.enabled:
            logger.info("Groq API service initialized successfully.")
        else:
            logger.warning(
                "Groq API key is missing or default. Backend will run in LLM Mock mode."
            )

    def generate_chat_completion(
        self, prompt: str, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """
        Sends a request to Groq API chat completion endpoint.
        Falls back to generating a mock response if service is disabled or fails.
        """
        if self.enabled:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            try:
                response = requests.post(
                    self.url, json=payload, headers=headers, timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(
                        f"Groq API returned error {response.status_code}: {response.text}"
                    )
            except Exception as e:
                logger.error(f"Error calling Groq API: {e}")

        # Fallback Mock logic
        return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generates mock LLM summary or Q&A answers based on keywords in prompt."""
        prompt_lower = prompt.lower()
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return (
                "## EXECUTIVE SUMMARY\n"
                "This video lecture presents a comprehensive analysis of Artificial Intelligence "
                "concepts, specifically focusing on Audio processing, Computer Vision (CLIP/BLIP-2), "
                "and LLM orchestration using LangChain. The speaker details the integration of "
                "these technologies to construct a fully automated, multimodal summarization pipeline.\n\n"
                "### KEY TAKEAWAYS\n"
                "- **Audio ASR**: Using WhisperX enables word-level timestamp alignment with less than 10% WER.\n"
                "- **Visual Keyframes**: PySceneDetect combined with CLIP embeddings clusters keyframes efficiently, "
                "yielding a semantic coverage score (F-score) higher than 0.45.\n"
                "- **Multimodal Fusion**: The blending of textual transcriptions and keyframe captions "
                "allows LLMs to generate structured, context-rich chapter summaries."
            )
        elif "question" in prompt_lower or "q&a" in prompt_lower or "?" in prompt_lower:
            return (
                "Based on the processed video transcript, the speaker explains that **WhisperX** "
                "is selected over original Whisper because it incorporates a forced alignment "
                "step using phoneme models. This provides precise word-level timestamps (needed for "
                "syncing video frames) and speeds up processing by 4x to 8x using the fast-whisper backend."
            )
        else:
            return (
                "This is an automated mock response simulating Groq's LLM generation. "
                "The requested prompt analyzed context related to video summarization and successfully completed."
            )


groq_service = GroqService()
