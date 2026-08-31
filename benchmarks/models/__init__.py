"""
Benchmark Models Package for Multimodal Lecture Understanding, Chaptering, Summarization, and Retrieval QA.
"""

from .chaptering import (
    BaseChapteringModel,
    C1_TextOnlyChapterer,
    C2_AcousticChapterer,
    C3_VisualChapterer,
    C4_OCRChapterer,
    C5_TemporalCrossAttentionTransformer,
    C6_LateFusionChapterer,
    ChapteringBatch,
    ChapteringOutput,
)

from .summarization import (
    BaseSummarizer,
    SummarizerConfig,
    SummaryResult,
    S0_FlatSummarizer,
    S1_FixedChunkMapReduceSummarizer,
    S2_OracleHierarchySummarizer,
    S3_PredictedHierarchySummarizer,
    S4_MultimodalHierarchySummarizer,
)

from .retrieval_qa import (
    BaseRetrievalQA,
    QAConfig,
    QAResult,
    Q0_FlatRetrievalQA,
    Q1_OracleHierarchyRetrievalQA,
    Q2_PredictedHierarchyRetrievalQA,
    Q3_MultimodalHierarchyRetrievalQA,
)

__all__ = [
    "BaseChapteringModel",
    "C1_TextOnlyChapterer",
    "C2_AcousticChapterer",
    "C3_VisualChapterer",
    "C4_OCRChapterer",
    "C5_TemporalCrossAttentionTransformer",
    "C6_LateFusionChapterer",
    "ChapteringBatch",
    "ChapteringOutput",
    "BaseSummarizer",
    "SummarizerConfig",
    "SummaryResult",
    "S0_FlatSummarizer",
    "S1_FixedChunkMapReduceSummarizer",
    "S2_OracleHierarchySummarizer",
    "S3_PredictedHierarchySummarizer",
    "S4_MultimodalHierarchySummarizer",
    "BaseRetrievalQA",
    "QAConfig",
    "QAResult",
    "Q0_FlatRetrievalQA",
    "Q1_OracleHierarchyRetrievalQA",
    "Q2_PredictedHierarchyRetrievalQA",
    "Q3_MultimodalHierarchyRetrievalQA",
]
