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

            # If scene is an action demonstration or has detected objects
            det = s.get("detected_objects", {})
            tracks = det.get("tracks_summary", []) if isinstance(det, dict) else []
            counts = s.get("object_counts", {})
            if tracks or (isinstance(counts, dict) and any(c > 0 for c in counts.values())):
                obj_parts = [f"{c} {lbl}" for lbl, c in sorted(counts.items()) if c > 0] if counts else [t.get("label") for t in tracks if t.get("label")]
                if obj_parts:
                    obj_str = ", ".join(obj_parts[:3])
                    m = int(s_start) // 60
                    sec = int(s_start) % 60
                    candidates.append({
                        "text": f"Khoảnh khắc thực hành kỹ thuật trực quan với {obj_str} ở mốc {m:02d}:{sec:02d}.",
                        "start": s_start,
                        "end": s_end,
                        "source": "action",
                        "importance_boost": imp_score * 0.35,
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
        max_takeaways: int = 6,
    ) -> List[Dict[str, Any]]:
        """Extract 4-6 distinct key takeaway events with actual timestamps and concrete titles."""
        def format_time(secs: float) -> str:
            m = int(secs) // 60
            s = int(secs) % 60
            return f"{m:02d}:{s:02d}"

        n_takeaways = max(4, min(max_takeaways, len(candidates)))
        selected = self._select_diverse_sentences(candidates, scores, max_sentences=n_takeaways)
        takeaways = []
        for s in selected:
            text = s["text"].strip()
            if text.endswith("."):
                text = text[:-1]
            start_ts = float(s.get("start", 0.0))
            time_str = format_time(start_ts)

            words = text.split()
            if len(words) > 7:
                title = " ".join(words[:5])
            else:
                title = text

            takeaways.append({
                "timestamp": round(start_ts, 2),
                "time_str": time_str,
                "title": title,
                "description": text,
            })
        return takeaways

    def _infer_title(
        self,
        candidates: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        chapters: List[Dict[str, Any]],
        output_language: str = "vi",
    ) -> str:
        """Infer a descriptive lecture title from slide OCR, chapter names, or top keywords."""
        is_en = (output_language or "vi").strip().lower() == "en"

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
            if len(first_chap) > 3 and not first_chap.lower().startswith(("chương 1: mở đầu", "chapter 1")):
                prefix = "Lecture" if is_en else "Bài giảng"
                return f"{prefix}: {first_chap}"

        # 3. Frequent significant terms
        all_tokens: List[str] = []
        for c in candidates:
            all_tokens.extend([t for t in self._tokenize(c["text"]) if t not in VI_EN_STOPWORDS and len(t) > 2])

        if all_tokens:
            from collections import Counter
            top_words = [w.capitalize() for w, _ in Counter(all_tokens).most_common(3)]
            if top_words:
                prefix = "Lecture Summary" if is_en else "Tóm tắt bài giảng"
                return f"{prefix}: {' - '.join(top_words)}"

        return "Lecture Summary (Extractive Summary)" if is_en else "Bản tóm tắt bài giảng (Extractive Summary)"

    def generate_fallback(
        self,
        utterances: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        chapters: List[Dict[str, Any]],
        llm_error: Optional[Any] = None,
        job_id: str = "unknown",
        output_language: str = "vi",
        summary_length: str = "medium",
    ) -> Dict[str, Any]:
        """Generate a complete, structured extractive fallback summary matching LLM output schema."""
        import datetime as dt

        length_key = (summary_length or "medium").strip().lower()
        if length_key not in ("short", "medium", "detailed"):
            length_key = "medium"

        candidates = self._extract_candidates(utterances, slides)
        is_en = (output_language or "vi").strip().lower() == "en"

        # Edge case: Absolutely no transcript or slide content
        if not candidates:
            duration = 0.0
            if chapters:
                duration = float(chapters[-1].get("endTime", 0.0))
            empty_text = "Unable to extract lecture content due to lack of audio and visual data." if is_en else "Không thể trích xuất nội dung bài giảng do thiếu bản ghi âm thanh và hình ảnh."
            return {
                "status": "failed",
                "video_title": "No Lecture Content Available" if is_en else "Không có dữ liệu bài giảng",
                "summary": empty_text,
                "key_takeaways": [],
                "chapters": chapters or [],
                "model_used": "Extractive Fallback (Failed: No Content)",
                "fallback_used": True,
                "summary_method": "extractive_fallback",
                "summary_length": length_key,
                "word_count": 0,
                "reading_time_sec": 0,
                "summary_variants": {
                    length_key: {
                        "summary_text": empty_text,
                        "key_takeaways_json": [],
                        "chapters_json": chapters or [],
                        "word_count": 0,
                        "reading_time_sec": 0,
                        "model_used": "Extractive Fallback (Failed: No Content)",
                        "summary_length": length_key,
                        "generated_at": dt.datetime.utcnow().isoformat(),
                    }
                },
                "llm_error": llm_error if isinstance(llm_error, dict) else (llm_error.to_dict() if hasattr(llm_error, "to_dict") else {"error_code": "NO_CONTENT", "message": "No utterances or slides found"}),
            }

        # 1. Score candidates
        scores = self._compute_tfidf_scores(candidates, slides)

        # 2. Sentence extraction & structural formatting based on cognitive depth
        if length_key == "short":
            # Short: 3-4 key sentences, 1 thesis + 3 bullet guidelines
            selected_sents = self._select_diverse_sentences(candidates, scores, max_sentences=4)
            if selected_sents:
                thesis = selected_sents[0]["text"]
                bullets = selected_sents[1:4]
            else:
                thesis = candidates[0]["text"]
                bullets = []

            bullet_lines = "\n".join([f"- {b['text']}" for b in bullets])
            if is_en:
                formatted_summary = (
                    f"### Executive Summary (TL;DR)\n\n"
                    f"{thesis}\n\n"
                    f"**Core Actionable Guidelines:**\n"
                    f"{bullet_lines if bullet_lines else '- Focus on core fundamentals.'}\n\n"
                    f"> *Note: Automated concise executive summary extracted from key lecture moments.*"
                )
            else:
                formatted_summary = (
                    f"### Tóm tắt Lướt nhanh (Executive TL;DR)\n\n"
                    f"{thesis}\n\n"
                    f"**Kim chỉ nam cốt lõi:**\n"
                    f"{bullet_lines if bullet_lines else '- Nắm vững các thao tác kỹ thuật cơ bản.'}\n\n"
                    f"> *Ghi chú: Bản tóm tắt cô đọng được trích xuất tự động từ các điểm trọng tâm nhất.*"
                )
            exec_summary_text = f"{thesis} " + " ".join([b["text"] for b in bullets])
            max_takeaways_target = 3

        elif length_key == "detailed":
            # Detailed: 15-20 sentences organized by chapter/stage with multimodal grounding
            selected_sents = self._select_diverse_sentences(candidates, scores, max_sentences=18)
            exec_summary_text = " ".join([s["text"] for s in selected_sents])

            section_blocks = []
            if chapters:
                for idx, chap in enumerate(chapters):
                    c_start = float(chap.get("startTime", 0.0))
                    c_end = float(chap.get("endTime", c_start + 60.0))
                    title = chap.get("title") or (f"Chapter {idx+1}" if is_en else f"Chương {idx+1}")
                    chap_sents = [
                        s["text"] for s in selected_sents
                        if float(s.get("start", 0.0)) >= c_start and float(s.get("start", 0.0)) <= c_end
                    ]
                    if not chap_sents:
                        # Fallback candidate within this chapter
                        for c in candidates:
                            if c_start <= float(c.get("start", 0.0)) <= c_end:
                                chap_sents.append(c["text"])
                                if len(chap_sents) >= 2:
                                    break

                    body = " ".join(chap_sents) if chap_sents else ("Chi tiết kỹ thuật đang được cập nhật." if not is_en else "Technical details in progress.")
                    # Find matching slide evidence
                    evidences = []
                    for s in slides:
                        s_st = float(s.get("start_seconds", s.get("timestamp", 0.0)))
                        if c_start <= s_st <= c_end:
                            ocr = (s.get("ocr_text") or "").strip()
                            cap = (s.get("caption") or "").strip()
                            if ocr:
                                ocr_snippet = ocr.split("\n")[0][:80]
                                evidences.append(f"Chữ trên slide: '{ocr_snippet}'" if not is_en else f"Slide text: '{ocr_snippet}'")
                            if cap and cap not in ("[Nhạc nền / Im lặng]", "Slide"):
                                evidences.append(f"Hành động trực quan: {cap}" if not is_en else f"Visual action: {cap}")
                            if len(evidences) >= 2:
                                break
                    evidence_str = f"\n*Bằng chứng đa phương thức*: {'; '.join(evidences)}" if evidences else ""
                    section_blocks.append(f"### {idx+1}. {title}\n{body}{evidence_str}")
            else:
                # Chunk selected sentences into 3 structured sections
                chunk_size = max(1, len(selected_sents) // 3)
                p1 = " ".join([s["text"] for s in selected_sents[:chunk_size]])
                p2 = " ".join([s["text"] for s in selected_sents[chunk_size:chunk_size*2]])
                p3 = " ".join([s["text"] for s in selected_sents[chunk_size*2:]])
                section_blocks.append(f"### 1. {'Phần mở đầu & Nguyên lý' if not is_en else 'Introduction & Principles'}\n{p1}")
                section_blocks.append(f"### 2. {'Phân tích Kỹ thuật & Thực hành' if not is_en else 'Technical Analysis & Execution'}\n{p2}")
                section_blocks.append(f"### 3. {'Tổng kết & Lưu ý an toàn' if not is_en else 'Summary & Safety Notes'}\n{p3}")

            intro_header = "### Hồ sơ Phân tích Chuyên sâu (Comprehensive Study Guide)" if not is_en else "### Comprehensive Multimodal Study Guide"
            formatted_summary = (
                f"{intro_header}\n\n"
                f"{chr(10).join(section_blocks)}\n\n"
                f"> *{'Tài liệu phân tích chuyên sâu được tổng hợp từ giọng nói, slide OCR và chuỗi hành động thị giác.' if not is_en else 'Comprehensive study guide synthesized from speech transcript, slide OCR, and visual action timelines.'}*"
            )
            max_takeaways_target = 7

        else:  # medium (default)
            # Medium: 7-9 sentences in 2-3 coherent paragraphs
            selected_sents = self._select_diverse_sentences(candidates, scores, max_sentences=8)
            exec_summary_text = " ".join([s["text"] for s in selected_sents])
            mid_point = max(1, len(selected_sents) // 2)
            p1 = " ".join([s["text"] for s in selected_sents[:mid_point]])
            p2 = " ".join([s["text"] for s in selected_sents[mid_point:]])

            if is_en:
                formatted_summary = (
                    f"### Lecture Overview (Automated Extractive Summary)\n\n"
                    f"{p1}\n\n"
                    f"{p2}\n\n"
                    f"> *Note: This summary was automatically generated from speech transcript and visual slide content.*"
                )
            else:
                formatted_summary = (
                    f"### Tổng quan bài giảng (Tóm tắt trích xuất tiêu chuẩn)\n\n"
                    f"{p1}\n\n"
                    f"{p2}\n\n"
                    f"> *Ghi chú: Bản tóm tắt tiêu chuẩn được tổng hợp cân đối từ lời giảng và phân cảnh trực quan.*"
                )
            max_takeaways_target = 5

        # 3. Key takeaways (scaled according to length tier)
        key_takeaways = self._extract_key_takeaways(candidates, scores, max_takeaways=max_takeaways_target)

        # 4. Update chapters with chapter-scoped summaries
        updated_chapters: List[Dict[str, Any]] = []
        if chapters:
            for idx, c in enumerate(chapters):
                chap_summary = self._extract_chapter_summary(c, candidates, scores, slides)
                default_title = f"Chapter {idx + 1}" if is_en else f"Chương {idx + 1}"
                chap_title = c.get("title") or default_title
                updated_chapters.append({
                    "title": chap_title,
                    "startTime": float(c.get("startTime", 0.0)),
                    "endTime": float(c.get("endTime", 0.0)),
                    "summary": chap_summary,
                })
        else:
            duration = candidates[-1]["end"] if candidates else 60.0
            updated_chapters.append({
                "title": "Lecture Content" if is_en else "Nội dung bài học",
                "startTime": 0.0,
                "endTime": duration,
                "summary": exec_summary_text,
            })

        # 5. Title inference
        from ai_workers.modules.summarization.summarizer import clean_video_title
        raw_inferred_title = self._infer_title(candidates, slides, chapters, output_language=output_language)
        title_evidence_lines = []
        for slide in slides:
            title_evidence_lines.extend(
                line.strip()
                for line in (slide.get("ocr_text") or "").splitlines()
                if line.strip()
            )
        title_evidence_lines.extend(
            (utterance.get("text") or "").strip()
            for utterance in utterances
            if (utterance.get("text") or "").strip()
        )
        title_evidence_lines.extend(
            (slide.get("caption") or slide.get("description") or "").strip()
            for slide in slides
            if (slide.get("caption") or slide.get("description") or "").strip()
        )
        video_title = clean_video_title(
            raw_inferred_title,
            summary_text=exec_summary_text,
            lang_code=output_language,
            chapters=updated_chapters,
            takeaways=key_takeaways,
            source_evidence="\n".join(title_evidence_lines),
        )

        # 6. Error metadata packaging
        err_dict = None
        if llm_error:
            if hasattr(llm_error, "to_dict"):
                err_dict = llm_error.to_dict()
            elif isinstance(llm_error, dict):
                err_dict = llm_error
            else:
                err_dict = {"error_code": "LLM_FALLBACK", "message": str(llm_error)}

        words = formatted_summary.split()
        word_count = len(words)
        reading_time_sec = max(10, int((word_count / 200) * 60))

        variant_data = {
            "summary_text": formatted_summary,
            "key_takeaways_json": key_takeaways,
            "chapters_json": updated_chapters,
            "word_count": word_count,
            "reading_time_sec": reading_time_sec,
            "model_used": "Extractive Fallback (TF-IDF + Multimodal)",
            "summary_length": length_key,
            "generated_at": dt.datetime.utcnow().isoformat(),
        }

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
            "summary_length": length_key,
            "word_count": word_count,
            "reading_time_sec": reading_time_sec,
            "summary_variants": {
                length_key: variant_data
            },
            "llm_error": err_dict,
        }
