"""
Benchmark Models Package for Multimodal Lecture Understanding, Chaptering, and Summarization.
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
]
