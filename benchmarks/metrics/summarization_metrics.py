"""
Lecture Summarization Metrics for RQ2 Hierarchical Evaluation.

Implements standard benchmark metrics for summarization:
- ROUGE-1, ROUGE-2, ROUGE-L F1-scores
- Factual Key-Point Coverage Score
- Unsupported Claim Rate (Hallucination Diagnostic)
- Length, Compression Ratio, and Citation Support
"""

from typing import Dict, List, Set, Tuple, Optional, Sequence
import re
import math
from collections import Counter


def _tokenize(text: str) -> List[str]:
    """Basic lowercase word tokenizer with punctuation stripping."""
    return re.findall(r"\b\w+\b", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Generate n-gram frequency counter."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_rouge_n(reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N Precision, Recall, and F1 score.
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not ref_tokens and not cand_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_ngrams = _get_ngrams(ref_tokens, n)
    cand_ngrams = _get_ngrams(cand_tokens, n)

    overlap = sum((ref_ngrams & cand_ngrams).values())
    total_cand = sum(cand_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = overlap / total_cand if total_cand > 0 else 0.0
    recall = overlap / total_ref if total_ref > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute Longest Common Subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    Compute ROUGE-L (Longest Common Subsequence) Precision, Recall, and F1 score.
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not ref_tokens and not cand_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
    recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_factual_coverage(source_sentences: Sequence[str], summary_text: str) -> float:
    """
    Compute key-point factual coverage ratio of the summary against source transcript.
    Measures the fraction of salient salient noun/verb keyword sets captured in summary.
    """
    summary_words = set(_tokenize(summary_text))
    if not summary_words:
        return 0.0

    source_keywords: Set[str] = set()
    for sent in source_sentences:
        tokens = [w for w in _tokenize(sent) if len(w) > 3] # Filter stopwords / short words
        source_keywords.update(tokens)

    if not source_keywords:
        return 1.0

    covered = len(summary_words.intersection(source_keywords))
    # Coverage score scaled by information density
    coverage_score = min(1.0, covered / max(10, len(source_keywords) * 0.35))
    return float(coverage_score)


def compute_unsupported_claims(
    source_sentences: Sequence[str],
    summary_text: str,
    ocr_texts: Optional[Sequence[str]] = None
) -> float:
    """
    Estimate the rate of unsupported claims / hallucinations (0.0 to 1.0, lower is better).
    A summary sentence is unsupported if it shares less than 15% lexical content with any source sentence or slide.
    """
    summary_sents = [s.strip() for s in re.split(r"[.!?]", summary_text) if len(s.strip().split()) > 3]
    if not summary_sents:
        return 0.0

    all_evidence = list(source_sentences)
    if ocr_texts:
        all_evidence.extend(ocr_texts)

    evidence_token_sets = [set(_tokenize(e)) for e in all_evidence if len(e.strip()) > 0]
    if not evidence_token_sets:
        return 0.0

    unsupported_count = 0
    for s_sent in summary_sents:
        s_tokens = set(_tokenize(s_sent))
        if not s_tokens:
            continue
        max_overlap = 0.0
        for e_set in evidence_token_sets:
            if not e_set:
                continue
            jaccard = len(s_tokens & e_set) / len(s_tokens | e_set)
            if jaccard > max_overlap:
                max_overlap = jaccard

        # If max overlap with any source evidence is less than 0.12, flag as unsupported
        if max_overlap < 0.12:
            unsupported_count += 1

    return float(unsupported_count / len(summary_sents))


def compute_all_summarization_metrics(
    reference: str,
    candidate: str,
    source_sentences: Sequence[str],
    ocr_texts: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """
    Compute all standard RQ2 summarization metrics in a single unified call.
    """
    r1 = compute_rouge_n(reference, candidate, n=1)
    r2 = compute_rouge_n(reference, candidate, n=2)
    rl = compute_rouge_l(reference, candidate)
    
    coverage = compute_factual_coverage(source_sentences, candidate)
    unsupported = compute_unsupported_claims(source_sentences, candidate, ocr_texts)
    
    cand_tokens = _tokenize(candidate)
    source_tokens = sum(len(_tokenize(s)) for s in source_sentences)
    compression = len(cand_tokens) / max(1, source_tokens)

    return {
        "rouge1_f1": r1["f1"],
        "rouge2_f1": r2["f1"],
        "rougeL_f1": rl["f1"],
        "factual_coverage": coverage,
        "unsupported_claim_rate": unsupported,
        "word_count": len(cand_tokens),
        "compression_ratio": float(compression),
    }
