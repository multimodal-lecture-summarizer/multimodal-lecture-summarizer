import logging
import re
import requests
from typing import Optional, Dict, Any, List
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Unified LLM service with OpenRouter, Groq, and Mock fallback support."""

    def __init__(self):
        self.openrouter_key = settings.OPENROUTER_API_KEY
        self.openrouter_model = settings.OPENROUTER_MODEL or "qwen/qwen-2.5-7b-instruct"
        self.openrouter_url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        self.openrouter_enabled = bool(
            self.openrouter_key and not self.openrouter_key.startswith("sk-or-v1-your")
        )

        self.groq_key = settings.GROQ_API_KEY
        self.groq_model = settings.GROQ_MODEL or "llama-3.1-8b-instant"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_enabled = bool(self.groq_key and not self.groq_key.startswith("gsk_your"))

        # Fallback models list for OpenRouter
        self.openrouter_models = [
            self.openrouter_model,
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "mistralai/mistral-small-24b-instruct-2501"
        ]

        if self.openrouter_enabled:
            logger.info(f"OpenRouter API initialized (Primary model: {self.openrouter_model}).")
        elif self.groq_enabled:
            logger.info(f"Groq API initialized (Model: {self.groq_model}).")
        else:
            logger.warning("No LLM API keys provided. Backend running in LLM Mock mode.")

    def generate_chat_completion(
        self, prompt: str, system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """Sends chat completion request with OpenRouter -> Groq -> Mock fallback hierarchy."""
        if not isinstance(system_prompt, str):
            system_prompt = "".join(system_prompt) if isinstance(system_prompt, (list, tuple)) else str(system_prompt)
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # 1. Try OpenRouter API first
        if self.openrouter_enabled:
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multimodal-lecture-summarizer",
                "X-Title": "Multimodal Lecture Summarizer",
            }
            
            for model in self.openrouter_models:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                }
                try:
                    logger.info(f"Attempting OpenRouter RAG Q&A with model: {model}...")
                    response = requests.post(
                        self.openrouter_url, json=payload, headers=headers, timeout=45
                    )
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        logger.info(f"Successfully generated answer with OpenRouter ({model})")
                        return content
                    else:
                        logger.warning(
                            f"OpenRouter model {model} returned status {response.status_code}: {response.text[:200]}"
                        )
                except Exception as e:
                    logger.error(f"Error calling OpenRouter with model {model}: {e}")

        # 2. Fallback to Groq API if OpenRouter fails or is disabled
        if self.groq_enabled:
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            try:
                logger.info(f"Falling back to Groq API with model: {self.groq_model}...")
                response = requests.post(
                    self.groq_url, json=payload, headers=headers, timeout=30
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

        # 3. Fallback Mock response
        logger.warning("All LLM providers failed or disabled. Returning mock response.")
        return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Last-resort answer from retrieved lecture context, never project boilerplate."""
        summary_match = re.search(
            r"Overall Lecture Summary:\s*(.+?)(?:\n\nLecture Chapter Outline:|\n\nRetrieved Specific|\Z)",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            summary = " ".join(summary_match.group(1).split())
            if summary:
                return summary[:1200]
        return (
            "I could not reach the LLM provider. Please retry; the lecture summary "
            "and transcript are available on the result page."
        )


llm_service = LLMService()
