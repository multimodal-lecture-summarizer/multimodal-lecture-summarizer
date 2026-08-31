"""
Benchmark Models Package for Multimodal Lecture Understanding and Chaptering.
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
]
