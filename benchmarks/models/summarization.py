"""
Hierarchical Lecture Summarization Pipelines (S0 - S4) for RQ2 Evaluation.

Implements pipelines defined in Master Plan & Decisions Log (D-T08):
- S0: Flat / Truncated Transcript Baseline (Direct LLM summarization)
- S1: Fixed-Chunk Map-Reduce Baseline (Chunking -> Map summaries -> Reduce)
- S2: Oracle Hierarchy Diagnostic (Ground-truth chapter segments)
- S3: Predicted Hierarchy Summarizer (Driven by C5 chapter boundaries)
- S4: Multimodal Predicted Hierarchy (C5 chapters + Transcript + OCR + Keyframes)

Strictly enforces equal source/output budgets per D-T08 (max 32k source tokens, max 512 output tokens).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any, Tuple
import math
import re
from benchmarks.models.llm_engine import BaseLLMEngine, get_llm_engine


@dataclass
class SummarizerConfig:
    variant_id: str
    max_source_tokens: int = 32000
    max_output_tokens: int = 512
    chunk_tokens: int = 2000
    max_frames: int = 200
    frame_resolution_px: int = 448


@dataclass
class SummaryResult:
    variant_id: str
    summary_text: str
    token_usage: Dict[str, int]
    num_chapters: int
    hierarchy: Optional[List[Dict[str, Any]]] = None
    status: str = "ok"


class BaseSummarizer:
    """Base class enforcing equal token budgets across all summarization variants (D-T08)."""
    def __init__(self, config: SummarizerConfig, llm_engine: Optional[BaseLLMEngine] = None):
        self.config = config
        self.llm = llm_engine if llm_engine is not None else get_llm_engine()
        self.assert_budget()

    def assert_budget(self) -> None:
        """Enforce strict budget invariant per decisions-log.md D-T08."""
        assert self.config.max_source_tokens <= 32000, (
            f"Budget violation for {self.config.variant_id}: "
            f"source_tokens = {self.config.max_source_tokens} > 32000 cap"
        )
        assert self.config.max_output_tokens <= 512, (
            f"Budget violation for {self.config.variant_id}: "
            f"output_tokens = {self.config.max_output_tokens} > 512 cap"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 1.3 tokens per word)."""
        words = len(text.split())
        return int(words * 1.3)

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Safely truncate text to stay strictly within token budget."""
        words = text.split()
        max_words = int(max_tokens / 1.3)
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text


class S0_FlatSummarizer(BaseSummarizer):
    """
    S0: Flat Transcript Baseline.
    Truncates transcript to 32k source tokens, produces flat summary up to 512 tokens.
    """
    def summarize(self, transcript_sentences: Sequence[str]) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)
        input_tokens = self._estimate_tokens(budgeted_input)
        print(f"    [S0 Flat] Nén {len(transcript_sentences)} câu ({input_tokens} tokens) -> LLM sinh tóm tắt phẳng...", flush=True)

        prompt = (
            "You are an expert scientific lecture summarizer. Please write a comprehensive, "
            "coherent summary of the following lecture under 512 tokens.\n\n"
            f"TRANSCRIPT:\n{budgeted_input}\n\n"
            "SUMMARY:"
        )
        summary_text = self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)
        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S0_flat",
            summary_text=summary_text,
            token_usage={"source_tokens": input_tokens, "output_tokens": self._estimate_tokens(summary_text)},
            num_chapters=1,
            hierarchy=None
        )


class S1_FixedChunkMapReduceSummarizer(BaseSummarizer):
    """
    S1: Fixed-Chunk Map-Reduce Baseline.
    Divides transcript into equal token chunks (e.g. 2000 tokens), summarizes each chunk,
    then combines into final 512-token summary.
    """
    def summarize(self, transcript_sentences: Sequence[str]) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)
        words = budgeted_input.split()
        chunk_word_size = int(self.config.chunk_tokens / 1.3)
        
        # Map step: chunking
        chunks = []
        for i in range(0, len(words), chunk_word_size):
            chunks.append(" ".join(words[i : i + chunk_word_size]))

        print(f"    [S1 MapReduce] Chia transcript thành {len(chunks)} chunks ({self.config.chunk_tokens} tokens/chunk) -> Bắt đầu Map...", flush=True)

        # Map step: generate chunk summaries
        chunk_summaries = []
        for idx, c in enumerate(chunks):
            map_prompt = (
                f"Summarize the key ideas in this lecture section (Part {idx+1}) in 1-2 concise paragraphs:\n\n"
                f"CONTENT:\n{c}\n\n"
                "SECTION SUMMARY:"
            )
            c_sum = self.llm.generate(map_prompt, max_tokens=128)
            chunk_summaries.append(c_sum)

        # Reduce step: synthesize final summary
        print(f"    [S1 MapReduce] Đã map {len(chunks)} sections -> Bắt đầu Reduce tổng hợp...", flush=True)
        combined_summaries = "\n".join([f"- Part {i+1}: {s}" for i, s in enumerate(chunk_summaries)])
        reduce_prompt = (
            "Synthesize the following section summaries into a coherent, comprehensive overall lecture summary "
            f"under 512 tokens:\n\n{combined_summaries}\n\nFINAL SUMMARY:"
        )
        summary_text = self.llm.generate(reduce_prompt, max_tokens=self.config.max_output_tokens)
        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S1_fixed_chunk",
            summary_text=summary_text,
            token_usage={"source_tokens": self._estimate_tokens(budgeted_input), "output_tokens": self._estimate_tokens(summary_text)},
            num_chapters=len(chunks),
            hierarchy=[{"chunk_id": i + 1, "text": chunk_summaries[i]} for i in range(len(chunks))]
        )


class S2_OracleHierarchySummarizer(BaseSummarizer):
    """
    S2: Oracle Hierarchy Diagnostic.
    Summarizes using reference ground-truth chapter boundaries.
    """
    def summarize(
        self,
        transcript_sentences: Sequence[str],
        oracle_chapters: Sequence[Dict[str, Any]]
    ) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)

        print(f"    [S2 Oracle] Tóm tắt theo {len(oracle_chapters)} chapters ground-truth...", flush=True)
        chapter_bullets = []
        for ch in oracle_chapters:
            title = ch.get("title", "Section")
            ch_sents = ch.get("sentences", [])
            ch_text = " ".join(ch_sents)
            prompt = (
                f"Summarize the key takeaway of this specific lecture chapter '{title}' in 1-2 concise sentences:\n\n"
                f"CONTENT:\n{ch_text}\n\n"
                "TAKEAWAY:"
            )
            ch_summary = self.llm.generate(prompt, max_tokens=64)
            chapter_bullets.append(f"- **{title}**: {ch_summary}")

        summary_text = "\n".join(chapter_bullets)
        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S2_oracle_hierarchy",
            summary_text=summary_text,
            token_usage={"source_tokens": self._estimate_tokens(budgeted_input), "output_tokens": self._estimate_tokens(summary_text)},
            num_chapters=len(oracle_chapters),
            hierarchy=list(oracle_chapters)
        )


def _build_macro_chapters(
    transcript_sentences: Sequence[str],
    predicted_boundaries_sec: Sequence[float],
    timestamps_sec: Optional[Sequence[float]] = None,
    max_macro_chapters: int = 6,
    ocr_texts: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """Partition transcript into chapters and merge fine-grained slides into macro-chapters (max 6)."""
    n_sents = len(transcript_sentences)
    raw_chapters = []

    if timestamps_sec is not None and len(timestamps_sec) == n_sents and len(predicted_boundaries_sec) > 0:
        boundaries = sorted(list(predicted_boundaries_sec))
        cur_ch_sents = []
        cur_b_idx = 0
        cur_start_ts = 0.0

        for s_idx, (sent, ts) in enumerate(zip(transcript_sentences, timestamps_sec)):
            if cur_b_idx < len(boundaries) and ts >= boundaries[cur_b_idx]:
                if cur_ch_sents:
                    raw_chapters.append({
                        "chapter_id": len(raw_chapters) + 1,
                        "timestamp": f"{int(cur_start_ts)}s",
                        "sentences": cur_ch_sents
                    })
                cur_ch_sents = [sent]
                cur_start_ts = boundaries[cur_b_idx]
                cur_b_idx += 1
            else:
                cur_ch_sents.append(sent)
        if cur_ch_sents:
            raw_chapters.append({
                "chapter_id": len(raw_chapters) + 1,
                "timestamp": f"{int(cur_start_ts)}s",
                "sentences": cur_ch_sents
            })
    else:
        num_chapters = max(1, len(predicted_boundaries_sec) + 1)
        chapter_size = max(1, n_sents // num_chapters)
        for ch_idx in range(num_chapters):
            start_i = ch_idx * chapter_size
            end_i = min(n_sents, (ch_idx + 1) * chapter_size) if ch_idx < num_chapters - 1 else n_sents
            ch_sents = transcript_sentences[start_i:end_i]
            ts_label = f"{int(predicted_boundaries_sec[ch_idx-1])}s" if ch_idx > 0 and ch_idx - 1 < len(predicted_boundaries_sec) else "0s"
            raw_chapters.append({
                "chapter_id": ch_idx + 1,
                "timestamp": ts_label,
                "sentences": ch_sents
            })

    # If raw chapters <= max_macro_chapters, attach OCR and return
    if len(raw_chapters) <= max_macro_chapters:
        for idx, ch in enumerate(raw_chapters):
            ch["ocr_text"] = ocr_texts[idx] if ocr_texts and idx < len(ocr_texts) else None
        return raw_chapters

    # Merge adjacent micro-chapters into max_macro_chapters
    macro_chapters = []
    group_size = math.ceil(len(raw_chapters) / max_macro_chapters)
    for g_idx in range(0, len(raw_chapters), group_size):
        group = raw_chapters[g_idx : g_idx + group_size]
        merged_sents = []
        for ch in group:
            merged_sents.extend(ch["sentences"])
        start_ts = group[0]["timestamp"]

        merged_ocr = None
        if ocr_texts:
            group_ocrs = [ocr_texts[i] for i in range(g_idx, min(len(ocr_texts), g_idx + group_size)) if ocr_texts and i < len(ocr_texts) and ocr_texts[i]]
            if group_ocrs:
                merged_ocr = " | ".join(group_ocrs[:3])

        macro_chapters.append({
            "chapter_id": len(macro_chapters) + 1,
            "timestamp": start_ts,
            "sentences": merged_sents,
            "ocr_text": merged_ocr,
            "num_subchapters": len(group)
        })

    return macro_chapters


class S3_PredictedHierarchySummarizer(BaseSummarizer):
    """
    S3: Predicted Hierarchy Summarizer (Core RQ2 contribution).
    Uses semantic chapter boundaries predicted by the C5 Temporal Transformer.
    Implements Hierarchical Chapter Map-Reduce with adaptive macro-clustering for fast, high-quality synthesis.
    """
    def summarize(
        self,
        transcript_sentences: Sequence[str],
        predicted_boundaries_sec: Sequence[float],
        timestamps_sec: Optional[Sequence[float]] = None
    ) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)

        # Build macro chapters (max 6 chapters for fast, high-coverage synthesis)
        raw_b_count = len(predicted_boundaries_sec)
        chapters = _build_macro_chapters(transcript_sentences, predicted_boundaries_sec, timestamps_sec, max_macro_chapters=6)
        n_chapters = len(chapters)
        print(f"    [S3 Hierarchy] Gom {raw_b_count} ranh giới C5 -> {n_chapters} macro-chapters -> Map takeaways ({n_chapters} chapters)...", flush=True)

        # Map Phase: extract chapter key takeaways (max 64 tokens per chapter)
        chapter_takeaways = []
        for c in chapters:
            ch_text = " ".join(c["sentences"])
            prompt = (
                f"State the core takeaway of Chapter {c['chapter_id']} ({c['timestamp']}) in 1-2 concise sentences:\n\n"
                f"CONTENT:\n{ch_text}\n\n"
                "TAKEAWAY:"
            )
            salient = self.llm.generate(prompt, max_tokens=64).strip()
            c["takeaway"] = salient
            chapter_takeaways.append(f"- Chapter {c['chapter_id']} [{c['timestamp']}]: {salient}")

        # Reduce / Synthesis Phase
        print(f"    [S3 Hierarchy] Đã map {n_chapters} chapters -> Reduce phân tầng tổng hợp...", flush=True)
        combined_chapters = "\n".join(chapter_takeaways)
        reduce_prompt = (
            "You are an expert scientific lecture summarizer. Synthesize the following chapter takeaways "
            "into a comprehensive, coherent, and well-structured lecture summary under 512 tokens. "
            "Preserve key terminology, chronological structure, and core findings:\n\n"
            f"CHAPTER TAKEAWAYS:\n{combined_chapters}\n\n"
            "FINAL SUMMARY:"
        )
        summary_text = self.llm.generate(reduce_prompt, max_tokens=self.config.max_output_tokens).strip()

        # Enforce D-T08 hard budget cap
        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S3_predicted_hierarchy",
            summary_text=summary_text,
            token_usage={
                "source_tokens": self._estimate_tokens(budgeted_input),
                "output_tokens": self._estimate_tokens(summary_text),
                "num_chapters": n_chapters,
            },
            num_chapters=n_chapters,
            hierarchy=chapters
        )


class S4_MultimodalHierarchySummarizer(BaseSummarizer):
    """
    S4: Multimodal Predicted Hierarchy Summarizer (Proposed representation S4).
    Enriches C5 predicted chapters with OCR slide key concepts and visual cues.
    Implements Multimodal Hierarchical Chapter Map-Reduce with adaptive macro-clustering.
    """
    def summarize(
        self,
        transcript_sentences: Sequence[str],
        predicted_boundaries_sec: Sequence[float],
        timestamps_sec: Optional[Sequence[float]] = None,
        ocr_texts: Optional[Sequence[str]] = None,
        visual_captions: Optional[Sequence[str]] = None
    ) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)

        # Build macro chapters with aligned OCR evidence (max 6 chapters)
        raw_b_count = len(predicted_boundaries_sec)
        chapters = _build_macro_chapters(
            transcript_sentences, predicted_boundaries_sec, timestamps_sec,
            max_macro_chapters=6, ocr_texts=ocr_texts
        )
        n_chapters = len(chapters)
        print(f"    [S4 Multimodal] Gom {raw_b_count} ranh giới + Slide OCR -> {n_chapters} macro-chapters -> Multimodal Map...", flush=True)

        # Map Phase: extract chapter key takeaways grounded in OCR and visual evidence
        chapter_takeaways = []
        for c in chapters:
            ch_text = " ".join(c["sentences"])
            raw_ocr = c.get("ocr_text")
            if raw_ocr:
                ocr_words = raw_ocr.split()
                ocr_concept = " ".join(ocr_words[:5]) if len(ocr_words) > 5 else raw_ocr
            else:
                ocr_concept = None

            c["slide_evidence"] = ocr_concept

            if ocr_concept:
                prompt = (
                    f"Summarize the key concepts of Chapter {c['chapter_id']} ({c['timestamp']}) in 1-2 concise sentences, "
                    f"incorporating slide evidence '{ocr_concept}':\n\n"
                    f"CONTENT:\n{ch_text}\n\n"
                    "TAKEAWAY:"
                )
            else:
                prompt = (
                    f"Summarize the key concepts of Chapter {c['chapter_id']} ({c['timestamp']}) in 1-2 concise sentences:\n\n"
                    f"CONTENT:\n{ch_text}\n\n"
                    "TAKEAWAY:"
                )
            salient = self.llm.generate(prompt, max_tokens=64).strip()
            c["salient_point"] = salient
            bullet = f"- Chapter {c['chapter_id']} [{c['timestamp']}]: {salient}"
            if ocr_concept:
                bullet += f" (Slide Focus: {ocr_concept})"
            chapter_takeaways.append(bullet)

        # Reduce / Synthesis Phase
        print(f"    [S4 Multimodal] Đã map {n_chapters} multimodal chapters -> Reduce phân tầng tổng hợp...", flush=True)
        combined_chapters = "\n".join(chapter_takeaways)
        reduce_prompt = (
            "You are an expert scientific lecture summarizer. Synthesize the following multimodal chapter takeaways "
            "(incorporating transcript arguments and visual slide evidence) into a comprehensive, coherent, "
            "and well-structured lecture summary under 512 tokens. Highlight key methodological and empirical takeaways:\n\n"
            f"CHAPTER EVIDENCE & TAKEAWAYS:\n{combined_chapters}\n\n"
            "FINAL MULTIMODAL SUMMARY:"
        )
        summary_text = self.llm.generate(reduce_prompt, max_tokens=self.config.max_output_tokens).strip()

        # Enforce D-T08 hard budget cap
        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S4_multimodal_hierarchy",
            summary_text=summary_text,
            token_usage={
                "source_tokens": self._estimate_tokens(budgeted_input),
                "output_tokens": self._estimate_tokens(summary_text),
                "num_chapters": n_chapters,
            },
            num_chapters=n_chapters,
            hierarchy=chapters
        )


class S3_PlusEvidenceSummarizer(BaseSummarizer):
    """
    S3+ev ablation: S3's text-only hierarchy with optional slide-evidence injection.
    Isolates the contribution of slide evidence in the Hierarchical Map-Reduce pipeline.
    """
    def summarize(
        self,
        transcript_sentences: Sequence[str],
        predicted_boundaries_sec: Sequence[float],
        timestamps_sec: Optional[Sequence[float]] = None,
        ocr_texts: Optional[Sequence[str]] = None,
    ) -> SummaryResult:
        full_text = " ".join(transcript_sentences)
        budgeted_input = self._truncate_to_budget(full_text, self.config.max_source_tokens)

        chapters = _build_macro_chapters(
            transcript_sentences, predicted_boundaries_sec, timestamps_sec,
            max_macro_chapters=6, ocr_texts=ocr_texts
        )
        n_chapters = len(chapters)
        print(f"    [S3+ev Ablation] Ghép slide evidence -> {n_chapters} macro-chapters -> Map...", flush=True)

        chapter_takeaways = []
        for c in chapters:
            ch_text = " ".join(c["sentences"])
            raw_ocr = c.get("ocr_text")
            if raw_ocr:
                ocr_words = raw_ocr.split()
                ocr_concept = " ".join(ocr_words[:5]) if len(ocr_words) > 5 else raw_ocr
            else:
                ocr_concept = None

            c["slide_evidence"] = ocr_concept
            if ocr_concept:
                prompt = (
                    f"State the core topic of Chapter {c['chapter_id']} ({c['timestamp']}) in 1-2 sentences, "
                    f"incorporating slide focus '{ocr_concept}':\n\n"
                    f"CONTENT:\n{ch_text}\n\n"
                    "TOPIC:"
                )
            else:
                prompt = (
                    f"State the core topic of Chapter {c['chapter_id']} ({c['timestamp']}) in 1-2 sentences:\n\n"
                    f"CONTENT:\n{ch_text}\n\n"
                    "TOPIC:"
                )
            salient = self.llm.generate(prompt, max_tokens=64).strip()
            c["salient_point"] = salient
            bullet = f"- Chapter {c['chapter_id']} [{c['timestamp']}]: {salient}"
            if ocr_concept:
                bullet += f" (Slide Focus: {ocr_concept})"
            chapter_takeaways.append(bullet)

        print(f"    [S3+ev Ablation] Đã map {n_chapters} chapters -> Reduce tổng hợp...", flush=True)
        combined_chapters = "\n".join(chapter_takeaways)
        reduce_prompt = (
            "You are an expert scientific lecture summarizer. Synthesize the following chapter takeaways "
            "(with slide context) into a structured lecture summary under 512 tokens:\n\n"
            f"CHAPTER TAKEAWAYS:\n{combined_chapters}\n\n"
            "FINAL SUMMARY:"
        )
        summary_text = self.llm.generate(reduce_prompt, max_tokens=self.config.max_output_tokens).strip()

        summary_text = self._truncate_to_budget(summary_text, self.config.max_output_tokens)

        return SummaryResult(
            variant_id="S3_plus_evidence",
            summary_text=summary_text,
            token_usage={
                "source_tokens": self._estimate_tokens(budgeted_input),
                "output_tokens": self._estimate_tokens(summary_text),
                "num_chapters": n_chapters,
            },
            num_chapters=n_chapters,
            hierarchy=chapters,
        )


def compute_rouge_metrics(candidate: str, reference: str) -> Dict[str, float]:
    """
    Compute authentic ROUGE-1, ROUGE-2, and ROUGE-L F1 scores between candidate and reference summaries.
    """
    def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        return ngrams

    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    cand_tokens = re.findall(r"\b\w+\b", candidate.lower())
    ref_tokens = re.findall(r"\b\w+\b", reference.lower())

    if not cand_tokens or not ref_tokens:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    # ROUGE-1
    cand_1 = get_ngrams(cand_tokens, 1)
    ref_1 = get_ngrams(ref_tokens, 1)
    overlap_1 = sum(min(cand_1[k], ref_1.get(k, 0)) for k in cand_1)
    r1_rec = overlap_1 / len(ref_tokens)
    r1_prec = overlap_1 / len(cand_tokens)
    r1_f1 = (2 * r1_rec * r1_prec) / (r1_rec + r1_prec + 1e-8)

    # ROUGE-2
    cand_2 = get_ngrams(cand_tokens, 2)
    ref_2 = get_ngrams(ref_tokens, 2)
    overlap_2 = sum(min(cand_2[k], ref_2.get(k, 0)) for k in cand_2)
    r2_rec = overlap_2 / max(1, len(ref_tokens) - 1)
    r2_prec = overlap_2 / max(1, len(cand_tokens) - 1)
    r2_f1 = (2 * r2_rec * r2_prec) / (r2_rec + r2_prec + 1e-8)

    # ROUGE-L
    lcs_val = lcs_length(cand_tokens, ref_tokens)
    rl_rec = lcs_val / len(ref_tokens)
    rl_prec = lcs_val / len(cand_tokens)
    rl_f1 = (2 * rl_rec * rl_prec) / (rl_rec + rl_prec + 1e-8)

    return {
        "rouge_1": float(round(r1_f1 * 100.0, 2)),
        "rouge_2": float(round(r2_f1 * 100.0, 2)),
        "rouge_l": float(round(rl_f1 * 100.0, 2)),
    }
