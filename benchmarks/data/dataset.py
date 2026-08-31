"""
PyTorch Dataset and DataLoader abstractions for Multimodal Lecture Benchmarks.
Loads real extracted multimodal features (.pt files) for Chaptering, Summarization, and QA.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from benchmarks.models.chaptering import ChapteringBatch


class LectureFeatureDataset(Dataset):
    """
    Dataset for loading cached multimodal lecture tensors (.pt).
    """
    def __init__(
        self,
        data_dir: str = "benchmarks/data/cached_features",
        manifest_path: Optional[str] = None,
        lecture_ids: Optional[List[str]] = None,
        max_seq_len: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []

        if lecture_ids is not None:
            target_files = [self.data_dir / f"{lid}.pt" for lid in lecture_ids]
        else:
            target_files = sorted(list(self.data_dir.glob("*.pt")))

        for pt_file in target_files:
            if pt_file.exists():
                try:
                    data = torch.load(pt_file, weights_only=False)
                    self.samples.append(data)
                except Exception as e:
                    print(f"[Dataset] Warning: Failed to load {pt_file}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        T = item["timestamps"].shape[0]
        if self.max_seq_len is not None and T > self.max_seq_len:
            # Slice to max_seq_len
            return {
                "lecture_id": item["lecture_id"],
                "timestamps": item["timestamps"][:self.max_seq_len],
                "text_features": item["text_features"][:self.max_seq_len],
                "visual_features": item["visual_features"][:self.max_seq_len],
                "ocr_features": item["ocr_features"][:self.max_seq_len],
                "acoustic_features": item["acoustic_features"][:self.max_seq_len],
                "targets": item["targets"][:self.max_seq_len],
                "transcript_sentences": item["transcript_sentences"][:self.max_seq_len],
            }
        return item


def collate_lecture_batches(batch: List[Dict[str, Any]]) -> ChapteringBatch:
    """
    Pad sequences in batch to max sequence length in batch and return ChapteringBatch.
    """
    batch_size = len(batch)
    lengths = [b["timestamps"].shape[0] for b in batch]
    max_len = max(lengths)

    d_text = batch[0]["text_features"].shape[-1]
    d_vis = batch[0]["visual_features"].shape[-1]
    d_ocr = batch[0]["ocr_features"].shape[-1]
    d_ac = batch[0]["acoustic_features"].shape[-1]

    padded_ts = torch.zeros(batch_size, max_len, dtype=torch.float32)
    padded_text = torch.zeros(batch_size, max_len, d_text, dtype=torch.float32)
    padded_vis = torch.zeros(batch_size, max_len, d_vis, dtype=torch.float32)
    padded_ocr = torch.zeros(batch_size, max_len, d_ocr, dtype=torch.float32)
    padded_ac = torch.zeros(batch_size, max_len, d_ac, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = lengths[i]
        padded_ts[i, :L] = b["timestamps"]
        padded_text[i, :L] = b["text_features"]
        padded_vis[i, :L] = b["visual_features"]
        padded_ocr[i, :L] = b["ocr_features"]
        padded_ac[i, :L] = b["acoustic_features"]
        padded_targets[i, :L] = b["targets"]
        mask[i, :L] = True

    return ChapteringBatch(
        timestamps=padded_ts,
        text_features=padded_text,
        visual_features=padded_vis,
        ocr_features=padded_ocr,
        acoustic_features=padded_ac,
        mask=mask,
        targets=padded_targets
    )


def create_lecture_splits(
    data_dir: str = "benchmarks/data/cached_features",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[LectureFeatureDataset, LectureFeatureDataset, LectureFeatureDataset]:
    """Create reproducible train, validation, and test dataset splits."""
    full_ds = LectureFeatureDataset(data_dir=data_dir)
    total = len(full_ds)
    if total == 0:
        raise ValueError(f"No samples found in {data_dir}")

    torch.manual_seed(seed)
    indices = torch.randperm(total).tolist()

    n_train = max(1, int(total * train_ratio))
    n_val = max(1, int(total * val_ratio))
    n_test = total - n_train - n_val
    if n_test <= 0:
        n_test = max(1, total - n_train)
        n_val = 0

    train_ids = [full_ds.samples[i]["lecture_id"] for i in indices[:n_train]]
    val_ids = [full_ds.samples[i]["lecture_id"] for i in indices[n_train : n_train + n_val]]
    test_ids = [full_ds.samples[i]["lecture_id"] for i in indices[n_train + n_val :]]

    train_set = LectureFeatureDataset(data_dir=data_dir, lecture_ids=train_ids)
    val_set = LectureFeatureDataset(data_dir=data_dir, lecture_ids=val_ids)
    test_set = LectureFeatureDataset(data_dir=data_dir, lecture_ids=test_ids)

    return train_set, val_set, test_set
