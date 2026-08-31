"""
QA and Evidence Localization Metrics for RQ3 Evaluation.

Implements standard benchmark metrics for question answering & retrieval:
- Retrieval: Recall@1, Recall@3, Mean Reciprocal Rank (MRR)
- Answer Quality: Token F1, Exact Match (EM)
- Grounding: Evidence Time-Interval IoU (Intersection over Union)
- Citation Precision: Fraction of predictions with valid temporal evidence
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any
import re


def _tokenize(text: str) -> List[str]:
    """Basic lowercase word tokenizer with punctuation stripping."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_retrieval_metrics(
    retrieved_chunk_ids: Sequence[Any],
    ground_truth_chunk_ids: Sequence[Any]
) -> Dict[str, float]:
    """
    Compute Recall@1, Recall@3, and Mean Reciprocal Rank (MRR).
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0}

    gt_set = set(ground_truth_chunk_ids)
    
    # Recall @ 1
    r_at_1 = 1.0 if retrieved_chunk_ids[0] in gt_set else 0.0
    
    # Recall @ 3
    top_3 = retrieved_chunk_ids[:3]
    r_at_3 = 1.0 if any(cid in gt_set for cid in top_3) else 0.0
    
    # MRR (Reciprocal Rank)
    rr = 0.0
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in gt_set:
            rr = 1.0 / rank
            break

    return {
        "recall_at_1": float(r_at_1),
        "recall_at_3": float(r_at_3),
        "mrr": float(rr)
    }


def compute_token_f1_and_em(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute Answer Token F1 score and Exact Match (EM) boolean.
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return {"f1": 1.0, "exact_match": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"f1": 0.0, "exact_match": 0.0}

    em = 1.0 if " ".join(pred_tokens) == " ".join(gt_tokens) else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return {"f1": 0.0, "exact_match": em}

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"f1": float(f1), "exact_match": float(em)}


def compute_evidence_iou(
    pred_interval: Tuple[float, float],
    true_interval: Tuple[float, float]
) -> float:
    """
    Compute 1D temporal Intersection over Union (IoU) between predicted evidence timestamp
    and ground truth evidence timestamp range.
    """
    p_start, p_end = min(pred_interval), max(pred_interval)
    t_start, t_end = min(true_interval), max(true_interval)

    inter_start = max(p_start, t_start)
    inter_end = min(p_end, t_end)
    intersection = max(0.0, inter_end - inter_start)

    union = (p_end - p_start) + (t_end - t_start) - intersection
    if union <= 0.0:
        return 1.0 if intersection > 0 else 0.0

    return float(intersection / union)


def compute_all_qa_metrics(
    retrieved_ids: Sequence[Any],
    gt_ids: Sequence[Any],
    predicted_answer: str,
    ground_truth_answer: str,
    pred_timestamp_range: Optional[Tuple[float, float]] = None,
    true_timestamp_range: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Compute all standard RQ3 QA and Evidence Localization metrics in a single unified call.
    """
    ret_m = compute_retrieval_metrics(retrieved_ids, gt_ids)
    ans_m = compute_token_f1_and_em(predicted_answer, ground_truth_answer)
    
    iou = 0.0
    if pred_timestamp_range is not None and true_timestamp_range is not None:
        iou = compute_evidence_iou(pred_timestamp_range, true_timestamp_range)

    has_grounding = 1.0 if pred_timestamp_range is not None and pred_timestamp_range[1] > pred_timestamp_range[0] else 0.0

    return {
        "recall_at_1": ret_m["recall_at_1"],
        "recall_at_3": ret_m["recall_at_3"],
        "mrr": ret_m["mrr"],
        "answer_f1": ans_m["f1"],
        "exact_match": ans_m["exact_match"],
        "evidence_iou": float(iou),
        "grounding_rate": float(has_grounding)
    }
