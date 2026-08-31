"""Lecture Multimodal Dataset & DataLoader Package."""

from benchmarks.data.dataset import (
    LectureFeatureDataset,
    collate_lecture_batches,
    create_lecture_splits,
)

__all__ = [
    "LectureFeatureDataset",
    "collate_lecture_batches",
    "create_lecture_splits",
]
