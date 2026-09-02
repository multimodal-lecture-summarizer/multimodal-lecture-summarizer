"""Extractive Fallback Summarizer for Multimodal Lecture Processing.

Provides deterministic, lightweight extractive summarization using TF-IDF,
multimodal alignment (OCR text, slide captions, visual importance scores),
cue-phrase heuristics, diversity filtering (MMR/redundancy penalty), and
chronological timeline reordering.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


VI_EN_STOPWORDS: Set[str] = {
    # Vietnamese common stopwords & fillers
    "và", "là", "của", "trong", "có", "các", "những", "cho", "với", "được",
    "này", "đó", "khi", "để", "thì", "mà", "từ", "ra", "vào", "lại", "qua",
    "đã", "sẽ", "đang", "rất", "nhiều", "như", "hay", "hoặc", "nếu", "bởi",
    "vì", "tại", "theo", "về", "lên", "xuống", "đến", "ở", "tôi", "chúng_ta",
    "chúng_tôi", "các_bạn", "thầy", "cô", "anh", "chị", "em", "mình", "người",
    "cái", "việc", "sự", "điều", "thế", "nào", "gì", "sao", "đâu", "nơi",
    "lúc", "khi_nào", "như_thế", "như_vậy", "ờ", "à", "ừm", "ừ", "hả", "nhé",
    "nha", "đấy", "rồi", "thôi", "luôn", "ngay", "chỉ", "cũng", "đều", "hơn",
    # English common stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}

CUE_PHRASES = [
    # Vietnamese cue phrases
    "tóm lại", "kết luận", "quan trọng là", "cần nhớ", "chú ý", "định nghĩa",
    "nguyên lý", "bản chất", "mục tiêu", "khái niệm", "đặc điểm", "ưu điểm",
    "nhược điểm", "quy tắc", "công thức", "kết quả", "ví dụ tiêu biểu", "nói tóm lại",
    # English cue phrases
    "in summary", "in conclusion", "importantly", "key point", "to summarize",
    "the main idea", "crucial", "essential", "definition", "principle", "formula",
]


class ExtractiveSummarizer:
    """Lightweight deterministic extractive summarizer for fallback operation."""

    def __init__(
        self,
        min_sentence_words: int = 4,
        max_sentence_words: int = 60,
        similarity_threshold: float = 0.55,
    ):
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    @staticmethod
    def _clean_sentence_text(text: str) -> str:
        """Strip filler words and extraneous whitespace."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        # Remove typical speech hesitation prefixes
        cleaned = re.sub(r"^(?:ờ|à|ừm|ừ|thì|là|như_vậy|thì_là)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences by punctuation and line breaks."""
        raw_parts = re.split(r"(?<=[.?!;:\n])\s+", text)
        sentences = []
        for p in raw_parts:
            cleaned = self._clean_sentence_text(p)
            words = self._tokenize(cleaned)
            if len(words) >= self.min_sentence_words:
                # Ensure it ends with proper punctuation
                if not cleaned.endswith((".", "!", "?")):
                    cleaned += "."
                sentences.append(cleaned)
        return sentences

    def _extract_candidates(
        self,
        utterances: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract candidate sentences with source timestamp metadata."""
        candidates: List[Dict[str, Any]] = []

        # 1. From Utterances (Speech)
        for u in utterances:
            text = (u.get("text") or "").strip()
            if not text:
                continue
            start = float(u.get("start", 0.0))
            end = float(u.get("end", start + 2.0))
            speaker = u.get("speaker")

            parts = self._split_into_sentences(text)
            if not parts:
                cleaned = self._clean_sentence_text(text)
                words = self._tokenize(cleaned)
                if len(words) >= 2:
                    parts = [cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."]

            duration_per_part = (end - start) / max(len(parts), 1)
            for idx, sent in enumerate(parts):
                p_start = start + (idx * duration_per_part)
                p_end = p_start + duration_per_part
                candidates.append({
                    "text": sent,
                    "start": p_start,
                    "end": p_end,
                    "source": "speech",
                    "speaker": speaker,
                    "importance_boost": 0.0,
                })

        # 2. From Slides (OCR text and captions if speech is sparse or for reinforcement)
        for s in slides:
            s_start = float(s.get("start_seconds", s.get("timestamp", 0.0)))
            s_end = float(s.get("end_seconds", s_start + 5.0))
            ocr_text = (s.get("ocr_text") or "").strip()
            caption = (s.get("caption") or s.get("description") or "").strip()
            imp_score = float(s.get("importanceScore", s.get("importance_score", 0.8)))

            # If OCR text contains substantive bullet points
            if ocr_text:
                for line in ocr_text.split("\n"):
                    line_clean = self._clean_sentence_text(line)
                    words = self._tokenize(line_clean)
                    if len(words) >= 3 and len(words) <= self.max_sentence_words:
                        if not line_clean.endswith((".", "!", "?")):
                            line_clean += "."
                        candidates.append({
                            "text": line_clean,
                            "start": s_start,
                            "end": s_end,
                            "source": "ocr",
                            "importance_boost": imp_score * 0.2,
                        })

            # If caption is meaningful (not generic music/silence)
            if caption and caption not in ["[Nhạc nền / Im lặng]", "Slide"]:
                words = self._tokenize(caption)
                if len(words) >= 4:
                    if not caption.endswith((".", "!", "?")):
                        caption += "."
                    candidates.append({
                        "text": caption,
                        "start": s_start,
                        "end": s_end,
                        "source": "caption",
                        "importance_boost": imp_score * 0.15,
                    })

        return candidates

    def _compute_tfidf_scores(
        self,
        candidates: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
    ) -> List[float]:
        """Compute TF-IDF and multimodal relevance scores for each candidate."""
        if not candidates:
            return []

        # Document frequencies
        doc_count = len(candidates)
        tokenized_candidates = [self._tokenize(c["text"]) for c in candidates]
        df: Dict[str, int] = {}
        for tokens in tokenized_candidates:
            unique_tokens = set(tokens) - VI_EN_STOPWORDS
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        # Calculate IDF
        idf: Dict[str, float] = {}
        for token, count in df.items():
            idf[token] = math.log((1.0 + doc_count) / (1.0 + count)) + 1.0

        # Build OCR & Caption token sets with timestamp intervals for multimodal boost
        visual_windows: List[Tuple[float, float, Set[str], float]] = []
        for s in slides:
            s_start = float(s.get("start_seconds", s.get("timestamp", 0.0)))
            s_end = float(s.get("end_seconds", s_start + 10.0))
            ocr_text = s.get("ocr_text", "")
            caption = s.get("caption", "") or s.get("description", "")
            imp = float(s.get("importanceScore", 0.8))
            v_tokens = set(self._tokenize(f"{ocr_text} {caption}")) - VI_EN_STOPWORDS
            if v_tokens:
                visual_windows.append((s_start, s_end, v_tokens, imp))

        scores: List[float] = []
        for idx, c in enumerate(candidates):
            tokens = tokenized_candidates[idx]
            meaningful_tokens = [t for t in tokens if t not in VI_EN_STOPWORDS]
            if not meaningful_tokens:
                scores.append(0.01)
                continue

            # 1. Base TF-IDF score
            raw_tfidf = sum(idf.get(t, 1.0) for t in meaningful_tokens)
            # Length normalization (damped by length^0.75)
            tfidf_score = raw_tfidf / (len(tokens) ** 0.75)

            # 2. Multimodal OCR & Visual overlap boost
            c_start = c["start"]
            c_end = c["end"]
            multimodal_boost = 0.0
            for v_start, v_end, v_tokens, v_imp in visual_windows:
                # Check timestamp overlap or adjacency within 5s
                if max(c_start, v_start) <= min(c_end, v_end) or abs(c_start - v_start) <= 5.0:
                    overlap_count = len(set(meaningful_tokens) & v_tokens)
                    if overlap_count > 0:
                        multimodal_boost += 0.25 * overlap_count * v_imp

            # 3. Cue phrases boost
            cue_boost = 0.0
            lower_text = c["text"].lower()
            for cue in CUE_PHRASES:
                if cue in lower_text:
                    cue_boost += 0.35
                    break

            # 4. Source & importance boost
            source_boost = c.get("importance_boost", 0.0)
            if c.get("source") == "speech":
                source_boost += 0.1  # Prefer spoken explanations over raw OCR bullets

            total_score = tfidf_score + multimodal_boost + cue_boost + source_boost
            scores.append(total_score)

        return scores

    @staticmethod
    def _jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Calculate Jaccard similarity between two token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    def _select_diverse_sentences(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        max_sentences: int = 5,
    ) -> List[Dict[str, Any]]:
        """Select top scoring sentences while penalizing redundancy (MMR approach)."""
        if not candidates:
            return []

        indexed_candidates = list(enumerate(candidates))
        # Sort by raw score descending
        indexed_candidates.sort(key=lambda x: scores[x[0]], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_token_sets: List[Set[str]] = []

        for orig_idx, cand in indexed_candidates:
            if len(selected) >= max_sentences:
                break

            cand_tokens = set(self._tokenize(cand["text"])) - VI_EN_STOPWORDS
            if not cand_tokens:
                continue

            # Check similarity with already selected sentences
            is_redundant = False
            for prev_tokens in selected_token_sets:
                sim = self._jaccard_similarity(cand_tokens, prev_tokens)
                if sim >= self.similarity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(cand)
                selected_token_sets.append(cand_tokens)

        # Fallback: if similarity pruned too aggressively, take highest scoring remaining
        if not selected and indexed_candidates:
            selected.append(indexed_candidates[0][1])

        # RE-SORT CHRONOLOGICALLY by timeline start seconds
        selected.sort(key=lambda c: c["start"])
        return selected

    def _extract_chapter_summary(
        self,
        chapter: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        scores: List[float],
        slides: List[Dict[str, Any]],
    ) -> str:
        """Extract a 1-3 sentence summary bounded strictly within the chapter time window."""
        c_start = float(chapter.get("startTime", 0.0))
        c_end = float(chapter.get("endTime", c_start + 60.0))

        # Filter candidates strictly within [c_start - 1.0, c_end + 1.0]
        chapter_cands: List[Tuple[int, Dict[str, Any]]] = []
        for idx, c in enumerate(candidates):
            if c["start"] >= (c_start - 1.0) and c["start"] <= (c_end + 1.0):
                chapter_cands.append((idx, c))

        if chapter_cands:
            cand_items = [item[1] for item in chapter_cands]
            cand_scores = [scores[item[0]] for item in chapter_cands]
            selected = self._select_diverse_sentences(cand_items, cand_scores, max_sentences=3)
            if selected:
                return " ".join([s["text"] for s in selected])

        # If no candidates in window, look for slides/OCR in that timeframe
        for s in slides:
            s_start = float(s.get("start_seconds", s.get("timestamp", 0.0)))
            if s_start >= c_start and s_start <= c_end:
                ocr = (s.get("ocr_text") or "").strip()
                cap = (s.get("caption") or s.get("description") or "").strip()
                if ocr:
                    first_line = ocr.split("\n")[0].strip()
                    if first_line:
                        return f"Nội dung slide chính: {first_line}."
                if cap and cap not in ["[Nhạc nền / Im lặng]", "Slide"]:
                    return f"Mô tả phân đoạn: {cap}."

        return f"Tóm tắt nội dung phần {chapter.get('title', 'bài học')} (khoảng thời gian {int(c_start)}s - {int(c_end)}s)."

    def _extract_key_takeaways(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        max_takeaways: int = 5,
    ) -> List[str]:
        """Extract distinct key takeaway bullet points."""
        selected = self._select_diverse_sentences(candidates, scores, max_sentences=max_takeaways)
        takeaways = []
        for s in selected:
            text = s["text"].strip()
            if text.endswith("."):
                text = text[:-1]
            takeaways.append(text)
        return takeaways

    def _infer_title(
        self,
        candidates: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        chapters: List[Dict[str, Any]],
    ) -> str:
        """Infer a descriptive lecture title from slide OCR, chapter names, or top keywords."""
        # 1. Check title from first slide (slide 0 OCR header is often the lecture title)
        if slides:
            first_slide = slides[0]
            ocr = (first_slide.get("ocr_text") or "").strip()
            if ocr:
                first_line = ocr.split("\n")[0].strip()
                words = self._tokenize(first_line)
                if 2 <= len(words) <= 12 and not first_line.startswith(("http", "www")):
                    return first_line

        # 2. Check first chapter title
        if chapters and chapters[0].get("title"):
            first_chap = chapters[0]["title"].strip()
            if len(first_chap) > 3 and not first_chap.lower().startswith("chương 1: mở đầu"):
                return f"Bài giảng: {first_chap}"

        # 3. Frequent significant terms
        all_tokens: List[str] = []
        for c in candidates:
            all_tokens.extend([t for t in self._tokenize(c["text"]) if t not in VI_EN_STOPWORDS and len(t) > 2])

        if all_tokens:
            from collections import Counter
            top_words = [w.capitalize() for w, _ in Counter(all_tokens).most_common(3)]
            if top_words:
                return f"Tóm tắt bài giảng: {' - '.join(top_words)}"

        return "Bản tóm tắt bài giảng (Extractive Summary)"

    def generate_fallback(
        self,
        utterances: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        chapters: List[Dict[str, Any]],
        llm_error: Optional[Any] = None,
        job_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Generate a complete, structured extractive fallback summary matching LLM output schema."""
        candidates = self._extract_candidates(utterances, slides)

        # Edge case: Absolutely no transcript or slide content
        if not candidates:
            duration = 0.0
            if chapters:
                duration = float(chapters[-1].get("endTime", 0.0))
            return {
                "status": "failed",
                "video_title": "Không có dữ liệu bài giảng",
                "summary": "Không thể trích xuất nội dung bài giảng do thiếu bản ghi âm thanh và hình ảnh.",
                "key_takeaways": [],
                "chapters": chapters or [],
                "model_used": "Extractive Fallback (Failed: No Content)",
                "fallback_used": True,
                "summary_method": "extractive_fallback",
                "llm_error": llm_error if isinstance(llm_error, dict) else (llm_error.to_dict() if hasattr(llm_error, "to_dict") else {"error_code": "NO_CONTENT", "message": "No utterances or slides found"}),
            }

        # 1. Score candidates
        scores = self._compute_tfidf_scores(candidates, slides)

        # 2. Executive summary (3-5 top sentences, sorted chronologically)
        executive_sents = self._select_diverse_sentences(candidates, scores, max_sentences=5)
        exec_summary_text = " ".join([s["text"] for s in executive_sents])

        # Markdown formatted executive summary
        formatted_summary = (
            f"### Tổng quan bài giảng (Tóm tắt trích xuất tự động)\n\n"
            f"{exec_summary_text}\n\n"
            f"> *Ghi chú: Bản tóm tắt được tổng hợp tự động từ các luận điểm then chốt trong lời giảng và nội dung slide.*"
        )

        # 3. Key takeaways (3-7 bullets)
        key_takeaways = self._extract_key_takeaways(candidates, scores, max_takeaways=5)

        # 4. Update chapters with chapter-scoped summaries
        updated_chapters: List[Dict[str, Any]] = []
        if chapters:
            for idx, c in enumerate(chapters):
                chap_summary = self._extract_chapter_summary(c, candidates, scores, slides)
                chap_title = c.get("title") or f"Chương {idx + 1}"
                updated_chapters.append({
                    "title": chap_title,
                    "startTime": float(c.get("startTime", 0.0)),
                    "endTime": float(c.get("endTime", 0.0)),
                    "summary": chap_summary,
                })
        else:
            # If chapters list was empty, create a single overarching chapter
            duration = candidates[-1]["end"] if candidates else 60.0
            updated_chapters.append({
                "title": "Nội dung bài học",
                "startTime": 0.0,
                "endTime": duration,
                "summary": exec_summary_text,
            })

        # 5. Title inference
        video_title = self._infer_title(candidates, slides, chapters)

        # 6. Error metadata packaging
        err_dict = None
        if llm_error:
            if hasattr(llm_error, "to_dict"):
                err_dict = llm_error.to_dict()
            elif isinstance(llm_error, dict):
                err_dict = llm_error
            else:
                err_dict = {"error_code": "LLM_FALLBACK", "message": str(llm_error)}

        return {
            "status": "done",
            "video_title": video_title,
            "summary": formatted_summary,
            "executive_summary": exec_summary_text,
            "key_takeaways": key_takeaways,
            "chapters": updated_chapters,
            "model_used": "Extractive Fallback (TF-IDF + Multimodal)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": err_dict,
        }
