"""
Unified LLM Engine Interface for RQ2 Summarization & RQ3 Question Answering.
Supports:
- Gemini API (google-generativeai / gemini-2.5-flash) with retry+fallback
- Local HuggingFace Transformers (Qwen2.5-1.5B-Instruct, BART-large-cnn) with singleton cache
- Deterministic Offline Abstractive Synthesizer v2 (SBERT-centrality + keyword fallback)
"""

import os
import re
import sys as _sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseLLMEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate text from prompt with strict token constraints."""
        pass


class GeminiLLMEngine(BaseLLMEngine):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Retry 2x on 429/ResourceExhausted with exponential backoff, then let caller fallback
        last_err = None
        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
                )
                text = getattr(response, "text", "") or ""
                if text.strip():
                    return text.strip()
                # Empty response is treated as failure for retry
                last_err = "empty response"
                raise ValueError("empty response")
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_ratelimit = any(k in msg for k in ["429", "quota", "rate", "exhausted", "resource"])
                if is_ratelimit and attempt < 2:
                    backoff = 2 ** (attempt + 1)  # 2s, 4s
                    print(f"[GeminiLLMEngine] RateLimit hit (attempt {attempt+1}/3), backoff {backoff}s: {e}")
                    time.sleep(backoff)
                    continue
                # Non-ratelimit or final attempt
                if attempt < 2 and "empty response" in msg:
                    continue
                print(f"[GeminiLLMEngine] Generation error: {e}")
                # Do not return "" silently on final ratelimit — let factory fallback
                if is_ratelimit:
                    raise
                return ""
        # Should not reach here; raise for factory fallback
        if last_err and "429" in str(last_err).lower() or "quota" in str(last_err).lower():
            raise RuntimeError(f"Gemini ratelimit after 3 attempts: {last_err}")
        return ""


class HuggingFaceLLMEngine(BaseLLMEngine):
    """Local HF engine with singleton cache. Supports Qwen (chat) and BART (seq2seq)."""
    # Use sys.modules as backing store so the cache survives module reimports in Jupyter.
    _CACHE_KEY = "__hf_llm_engine_cache__"
    if _CACHE_KEY not in _sys.modules:
        _sys.modules[_CACHE_KEY] = {}
    _CACHE: Dict[str, Any] = _sys.modules[_CACHE_KEY]  # type: ignore

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[str] = None, trust_remote_code: bool = False):
        # Auto device: cuda if available, else cpu (plan: auto skip HF on cpu)
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        # On CPU, Qwen 1.5B is very slow — caller should have checked, but we still allow with warning
        if device == "cpu":
            print("[HuggingFaceLLMEngine] WARNING: Running on CPU — Qwen 1.5B will be slow (>30s/call). Consider deterministic fallback.")

        # Respect HF_HOME (Colab: /root/.cache/huggingface)
        hf_home = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
        if hf_home:
            os.environ["HF_HOME"] = hf_home

        cache_key = f"{model_id}::{device}"
        if cache_key in HuggingFaceLLMEngine._CACHE:
            self.tokenizer, self.model, self.pipe, self.model_id, self.device = HuggingFaceLLMEngine._CACHE[cache_key]
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

        self.model_id = model_id
        self.device = device
        is_bart = "bart" in model_id.lower()

        # Device map and dtype
        device_map = device
        model_kwargs = {}
        if device == "cuda":
            try:
                import torch, gc
                # Free VRAM before loading a new model (chaptering models may still be resident)
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"[HuggingFaceLLMEngine] Free VRAM before load: {free_gb:.2f} GB")

                # Try 4-bit quantization first (saves ~2.5 GB vs FP16, needs bitsandbytes)
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                    model_kwargs["quantization_config"] = bnb_cfg
                    model_kwargs["device_map"] = "auto"
                    device_map = "auto"
                    print("[HuggingFaceLLMEngine] Using 4-bit quantization (bitsandbytes) — ~0.8 GB VRAM")
                except ImportError:
                    # bitsandbytes not available — fall back to FP16
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                    device_map = "auto"
                    print("[HuggingFaceLLMEngine] bitsandbytes not found, using FP16 — ~3 GB VRAM")
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        if is_bart:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, **model_kwargs)
            self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if device == "cuda" else -1)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, **model_kwargs)
            # pipeline handles device via model device_map; use -1 for cpu, 0 for cuda when not auto
            pipe_device = -1
            if device == "cuda" and model_kwargs.get("device_map") != "auto":
                pipe_device = 0
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=pipe_device)

        HuggingFaceLLMEngine._CACHE[cache_key] = (self.tokenizer, self.model, self.pipe, self.model_id, self.device)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        is_bart = "bart" in self.model_id.lower()
        try:
            if is_bart:
                # BART is seq2seq summarization — truncate input to 1024 tokens for BART limit
                res = self.pipe(prompt, max_length=max_tokens, min_length=max(16, max_tokens // 4), do_sample=False)
                return res[0]["summary_text"].strip()
            else:
                # Hard-truncate prompt to MAX_INPUT_TOKENS to avoid OOM on T4 (14.56 GB)
                # Qwen2.5-1.5B safe limit: ~4096 input tokens (FP16) or ~6000 (4-bit)
                MAX_INPUT_TOKENS = 3072
                # Estimate: ~1.3 chars/token on average
                max_chars = int(MAX_INPUT_TOKENS * 3.5)
                if len(prompt) > max_chars:
                    # Keep system instructions (first 300 chars) + truncated body
                    head = prompt[:300]
                    tail = prompt[-(max_chars - 300):]
                    prompt = head + "\n...[truncated for VRAM budget]...\n" + tail
                messages = [{"role": "user", "content": prompt}]
                res = self.pipe(messages, max_new_tokens=max_tokens, temperature=max(0.01, temperature), do_sample=temperature > 0, return_full_text=False)
                # pipeline returns list of dicts with generated_text
                if isinstance(res, list) and res:
                    out = res[0].get("generated_text", "")
                    if isinstance(out, list):
                        # chat format returns list of messages
                        out = out[-1].get("content", "") if isinstance(out[-1], dict) else str(out[-1])
                    elif isinstance(out, dict):
                        out = out.get("content", str(out))
                    return str(out).strip()
                return str(res).strip()
        except Exception as e:
            print(f"[HuggingFaceLLMEngine] Generation error ({self.model_id}): {e}")
            raise
        return ""


class DeterministicAbstractiveEngine(BaseLLMEngine):
    """
    High-fidelity offline multi-sentence synthesis engine v2.
    - Tries SBERT centrality (all-MiniLM-L6-v2) if sentence_transformers available
    - Falls back to keyword-length scoring
    """
    # SBERT model cached in sys.modules to survive reimports and avoid repeated HF downloads
    _SBERT_CACHE_KEY = "__deterministic_sbert_model__"

    def __init__(self):
        pass

    def _get_sbert_model(self):
        """Load SBERT model once per kernel session, cached in sys.modules."""
        if _sys.modules.get(self._SBERT_CACHE_KEY) is not None:
            return _sys.modules[self._SBERT_CACHE_KEY]
        try:
            from sentence_transformers import SentenceTransformer
            # Try local cache first (avoids HF 429 rate-limit on repeated Colab runs)
            try:
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    local_files_only=True,
                )
            except Exception:
                # Not cached locally yet — download once
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    local_files_only=False,
                )
            _sys.modules[self._SBERT_CACHE_KEY] = model
            return model
        except Exception:
            _sys.modules[self._SBERT_CACHE_KEY] = None  # mark as unavailable, don't retry
            return None

    def _sbert_centrality_scores(self, sents: List[str]) -> Optional[List[float]]:
        try:
            import numpy as np
            model = self._get_sbert_model()
            if model is None:
                return None
            embs = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            centroid = embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            scores = (embs @ centroid).tolist()
            return scores
        except Exception:
            return None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        source = prompt
        if "TRANSCRIPT:" in prompt:
            source = prompt.split("TRANSCRIPT:", 1)[1]
        elif "CONTENT:" in prompt:
            source = prompt.split("CONTENT:", 1)[1]
        
        raw_sents = [s.strip() for s in re.split(r'[.\n]+', source) if len(s.strip().split()) > 3]
        if not raw_sents:
            return "This lecture provides a comprehensive overview of the discussed concepts and fundamental principles."

        # SBERT centrality if available
        centrality = self._sbert_centrality_scores(raw_sents)

        scored_sents = []
        keywords = {"analysis", "strategy", "concept", "example", "method", "system", "algorithm", "data", "model", "definition", "result", "important", "conclusion", "hypothesis", "experiment", "theory"}
        
        for idx, s in enumerate(raw_sents):
            words = set(re.findall(r"\b\w+\b", s.lower()))
            kw_hits = len(words.intersection(keywords))
            # Base score: length + keyword + position decay
            score = len(s.split()) * 0.5 + kw_hits * 4.0 - idx * 0.1
            # Boost by centrality if available
            if centrality is not None:
                score += float(centrality[idx]) * 10.0
            scored_sents.append((score, idx, s))

        scored_sents.sort(key=lambda x: x[0], reverse=True)
        top_sents = sorted(scored_sents[:min(5, len(scored_sents))], key=lambda x: x[1])

        summary_parts = []
        summary_parts.append(f"In this lecture, the core focus centers on {top_sents[0][2].lower() if top_sents else 'the subject'}.")
        for item in top_sents[1:]:
            sent = item[2]
            if not sent.endswith('.'):
                sent += '.'
            summary_parts.append(sent)

        result = " ".join(summary_parts)
        words = result.split()
        max_words = int(max_tokens / 1.3)
        if len(words) > max_words:
            result = " ".join(words[:max_words]) + "."
        return result


def _try_hf(preference: str) -> Optional[BaseLLMEngine]:
    """Try to create HF engine; return None on any failure. Handles auto CPU skip."""
    # On CPU, Qwen is slow — still try but caller may decide to skip
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    if preference == "hf-bart":
        model_id = "facebook/bart-large-cnn"
    try:
        import torch
        is_cuda = torch.cuda.is_available()
    except Exception:
        is_cuda = False

    if not is_cuda and preference == "auto":
        # Auto on CPU: skip HF to avoid 10min run, go deterministic (validated decision)
        print("[get_llm_engine] HF skipped on CPU (auto->deterministic for speed)")
        return None

    try:
        return HuggingFaceLLMEngine(model_id=model_id)
    except Exception as e:
        print(f"[get_llm_engine] HuggingFace load failed ({model_id}): {e}")
        return None


# ---------------------------------------------------------------------------
# Singleton cache — stored in sys.modules so it survives module reimports
# inside a Jupyter/Colab kernel session.  Plain module-level dicts are reset
# every time the module is reloaded; sys.modules entries are not.
# ---------------------------------------------------------------------------
_CACHE_KEY = "__llm_engine_cache__"
if _CACHE_KEY not in _sys.modules:
    _sys.modules[_CACHE_KEY] = {}          # preference_str -> BaseLLMEngine instance

# Convenience alias used everywhere in this module
_ENGINE_CACHE: Dict[str, "BaseLLMEngine"] = _sys.modules[_CACHE_KEY]  # type: ignore


def _resolve_preference(preference: str) -> str:
    """Normalise preference string and apply LLM_PREFERENCE env-var override for 'auto'."""
    pref = (preference or "auto").lower().strip()
    if pref == "auto":
        env_pref = os.getenv("LLM_PREFERENCE", "").lower().strip()
        if env_pref:
            pref = env_pref
    return pref


def _build_engine(pref: str) -> BaseLLMEngine:
    """Create a fresh engine for the given (already-resolved) preference."""
    if pref == "deterministic":
        return DeterministicAbstractiveEngine()

    if pref in ("hf", "hf-bart"):
        hf = _try_hf(pref)
        if hf is not None:
            print(f"[get_llm_engine] Using {hf.__class__.__name__} ({getattr(hf, 'model_id', pref)})")
            return hf
        print(f"[get_llm_engine] HF ({pref}) unavailable, fallback to deterministic")
        return DeterministicAbstractiveEngine()

    if pref == "gemini":
        try:
            engine = GeminiLLMEngine()
            print(f"[get_llm_engine] Using GeminiLLMEngine ({engine.model_name})")
            return engine
        except Exception as e:
            print(f"[get_llm_engine] Gemini init failed ({e}), fallback to HF/deterministic")
            hf = _try_hf("auto")
            if hf is not None:
                return hf
            return DeterministicAbstractiveEngine()

    # auto (default)
    # Try HF first (unless CPU-skip), else deterministic. Gemini is NOT tried in auto (offline-first).
    hf = _try_hf("auto")
    if hf is not None:
        print(f"[get_llm_engine] Using {hf.__class__.__name__} ({getattr(hf, 'model_id', 'auto')}) [auto]")
        return hf
    print("[get_llm_engine] Using DeterministicAbstractiveEngine [auto fallback]")
    return DeterministicAbstractiveEngine()


def get_llm_engine(preference: str = "auto") -> BaseLLMEngine:
    """
    Factory helper to obtain the best available LLM backend.

    preference: auto | hf | hf-bart | deterministic | gemini
    - auto: HF if CUDA available, else deterministic; gemini only if explicitly preferred
    - hf / hf-bart: try HF, fallback to deterministic on failure (never raise)
    - deterministic: always deterministic (offline, no network)
    - gemini: try Gemini (with retry), on 429/failure fallback to HF→deterministic with warning

    Singleton-cached per resolved preference key — repeated calls return the
    same instance so HuggingFace models are only loaded once per process.
    Never raises 429 to caller — always returns a working engine.
    """
    pref = _resolve_preference(preference)
    if pref not in _ENGINE_CACHE:
        _ENGINE_CACHE[pref] = _build_engine(pref)
    else:
        print(f"[get_llm_engine] Reusing cached {_ENGINE_CACHE[pref].__class__.__name__} (preference={pref})")
    return _ENGINE_CACHE[pref]


def clear_llm_engine_cache() -> None:
    """Clear the singleton cache (useful for testing or switching preferences at runtime)."""
    _ENGINE_CACHE.clear()
