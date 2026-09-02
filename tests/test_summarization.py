"""
Unit Tests for Hierarchical Summarization Pipelines (S0 - S4) and Metrics.
"""

import pytest

from benchmarks.models.summarization import (
    SummarizerConfig,
    S0_FlatSummarizer,
    S1_FixedChunkMapReduceSummarizer,
    S2_OracleHierarchySummarizer,
    S3_PredictedHierarchySummarizer,
    S4_MultimodalHierarchySummarizer,
    S3_PlusEvidenceSummarizer,
)
from benchmarks.metrics.summarization_metrics import (
    compute_rouge_n,
    compute_rouge_l,
    compute_factual_coverage,
    compute_unsupported_claims,
    compute_all_summarization_metrics,
)


@pytest.fixture
def sample_lecture_sentences():
    return [
        "Welcome everyone to this scientific lecture on neuroimaging applications.",
        "Today we focus specifically on multiple hypothesis testing corrections.",
        "The standard Family-Wise Error Rate controls the probability of false positives.",
        "The Bonferroni correction divides the significance threshold alpha by m tests.",
        "However the Bonferroni procedure is overly conservative under positive dependence.",
        "Holm sequentially rejects hypotheses from the smallest p-value to the largest.",
        "False Discovery Rate control offers greater statistical power for voxel-wise fMRI.",
        "Benjamini and Hochberg introduced the step-up procedure in nineteen ninety five.",
        "In conclusion controlling FDR provides the optimal trade-off for discovery."
    ]


@pytest.fixture
def sample_reference_summary():
    return (
        "This lecture reviews multiple hypothesis testing in neuroimaging. "
        "It compares Family-Wise Error Rate methods like Bonferroni and Holm with False Discovery Rate. "
        "FDR provides greater statistical power for large-scale voxel analysis."
    )


def test_budget_enforcement():
    valid_cfg = SummarizerConfig(variant_id="S3_valid", max_source_tokens=32000, max_output_tokens=512)
    s3 = S3_PredictedHierarchySummarizer(valid_cfg)
    assert s3.config.max_source_tokens == 32000

    with pytest.raises(AssertionError):
        invalid_cfg = SummarizerConfig(variant_id="S3_invalid", max_source_tokens=64000)
        S3_PredictedHierarchySummarizer(invalid_cfg)


def test_s0_flat_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S0")
    summarizer = S0_FlatSummarizer(cfg)
    res = summarizer.summarize(sample_lecture_sentences)

    assert res.variant_id == "S0_flat"
    assert len(res.summary_text) > 0
    assert res.token_usage["output_tokens"] <= 512


def test_s1_fixed_chunk_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S1", chunk_tokens=50)
    summarizer = S1_FixedChunkMapReduceSummarizer(cfg)
    res = summarizer.summarize(sample_lecture_sentences)

    assert res.variant_id == "S1_fixed_chunk"
    assert res.num_chapters >= 1
    assert res.hierarchy is not None


def test_s2_oracle_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S2")
    summarizer = S2_OracleHierarchySummarizer(cfg)
    oracle_chapters = [
        {"title": "Introduction", "sentences": sample_lecture_sentences[:3]},
        {"title": "Methods", "sentences": sample_lecture_sentences[3:]}
    ]
    res = summarizer.summarize(sample_lecture_sentences, oracle_chapters)

    assert res.variant_id == "S2_oracle_hierarchy"
    assert res.num_chapters == 2


def test_s3_predicted_hierarchy_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S3")
    summarizer = S3_PredictedHierarchySummarizer(cfg)
    boundaries = [180.0, 360.0, 540.0]
    res = summarizer.summarize(sample_lecture_sentences, boundaries)

    assert res.variant_id == "S3_predicted_hierarchy"
    assert res.num_chapters == 4
    assert len(res.summary_text) > 0
    assert res.token_usage["output_tokens"] <= 512


def test_s4_multimodal_hierarchy_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S4")
    summarizer = S4_MultimodalHierarchySummarizer(cfg)
    boundaries = [180.0, 360.0]
    ocr_texts = ["FWER and Bonferroni Formula", "FDR Benjamini-Hochberg Equation", "Conclusion & Key Takeaways"]
    res = summarizer.summarize(sample_lecture_sentences, boundaries, ocr_texts=ocr_texts)

    assert res.variant_id == "S4_multimodal_hierarchy"
    assert res.num_chapters == 3
    assert len(res.summary_text) > 0
    assert res.token_usage["output_tokens"] <= 512


def test_s3_plus_evidence_summarizer(sample_lecture_sentences):
    cfg = SummarizerConfig(variant_id="S3_plus_evidence")
    summarizer = S3_PlusEvidenceSummarizer(cfg)
    boundaries = [180.0, 360.0]
    ocr_texts = ["FWER and Bonferroni Formula", "FDR Benjamini-Hochberg Equation", "Conclusion & Key Takeaways"]
    res = summarizer.summarize(sample_lecture_sentences, boundaries, ocr_texts=ocr_texts)

    assert res.variant_id == "S3_plus_evidence"
    assert res.num_chapters == 3
    assert len(res.summary_text) > 0
    assert res.token_usage["output_tokens"] <= 512


def test_summarization_metrics(sample_lecture_sentences, sample_reference_summary):
    candidate = (
        "This lecture discusses multiple hypothesis testing in neuroimaging. "
        "Controlling FDR offers greater statistical power than Bonferroni."
    )
    metrics = compute_all_summarization_metrics(
        reference=sample_reference_summary,
        candidate=candidate,
        source_sentences=sample_lecture_sentences,
        ocr_texts=["Multiple Hypothesis Testing in fMRI", "FDR vs FWER"]
    )

    assert 0.0 <= metrics["rouge1_f1"] <= 1.0
    assert 0.0 <= metrics["rouge2_f1"] <= 1.0
    assert 0.0 <= metrics["rougeL_f1"] <= 1.0
    assert 0.0 <= metrics["factual_coverage"] <= 1.0
    assert 0.0 <= metrics["unsupported_claim_rate"] <= 1.0
    assert metrics["word_count"] > 0
