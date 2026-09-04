"""
Multimodal Chaptering Models (C1 - C6) for Scientific Lecture Video Understanding.

Implements architectures defined in decisions-log.md (D-T02) and Phase 1 refactoring:
- C1: Text-only Baseline
- C2: Acoustic Ablation (d_ac = 32)
- C3: Visual Ablation (DINOv2 ViT-S/14)
- C4: OCR Ablation (PaddleOCR v3)
- C5: Proposed Decoupled Cross-Attention Transformer (Visual Query Anchor <-> Text/OCR/Acoustic Keys/Values)
      with Visual-Anchor Snapping (+/-45s) and Silent Intro Guard
- C6: Late Fusion Baseline (Concatenation-only, d_ac = 32)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChapteringBatch:
    """
    Standard input container for multimodal sequence chaptering.
    """
    timestamps: torch.Tensor             # [B, T] seconds
    text_features: Optional[torch.Tensor] = None      # [B, T, D_text]
    visual_features: Optional[torch.Tensor] = None    # [B, T, D_vis]
    ocr_features: Optional[torch.Tensor] = None       # [B, T, D_ocr]
    acoustic_features: Optional[torch.Tensor] = None  # [B, T, D_ac]
    mask: Optional[torch.Tensor] = None               # [B, T] True = valid token
    targets: Optional[torch.Tensor] = None            # [B, T] 1 = boundary, 0 = non-boundary

    def to(self, device) -> "ChapteringBatch":
        """Move all tensors to *device* and return a new ChapteringBatch.

        Mirrors the PyTorch convention so callers can write:
            batch = collate_lecture_batches(items).to(device)
        instead of manually moving each field.
        """
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return ChapteringBatch(
            timestamps=self.timestamps.to(device),
            text_features=_mv(self.text_features),
            visual_features=_mv(self.visual_features),
            ocr_features=_mv(self.ocr_features),
            acoustic_features=_mv(self.acoustic_features),
            mask=_mv(self.mask),
            targets=_mv(self.targets),
        )


@dataclass
class ChapteringOutput:
    """
    Standard output container for chaptering inference and training.
    """
    logits: torch.Tensor                         # [B, T]
    probabilities: torch.Tensor                  # [B, T]
    loss: Optional[torch.Tensor] = None          # Scalar loss if targets provided
    predicted_boundaries: List[List[float]] = field(default_factory=list) # List of boundary timestamps per sample


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with temporal support."""
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


def extract_visual_transitions(
    visual_features: torch.Tensor,
    timestamps: torch.Tensor,
    threshold: Optional[float] = None
) -> List[float]:
    """
    Extract visual transition timestamps from consecutive visual feature embeddings (D-T02, Red Team Finding 4).
    Computes cosine distance Delta v_t = 1 - cos(v_t, v_{t-1}).
    Adaptive threshold: tau_v = max(0.20, mean(Delta v) + 0.5 * std(Delta v)) if threshold is None.
    """
    if visual_features is None or len(visual_features) <= 1:
        return []

    if isinstance(visual_features, torch.Tensor):
        vf = visual_features.detach().cpu().float().numpy()
    else:
        vf = np.array(visual_features, dtype=np.float32)

    if isinstance(timestamps, torch.Tensor):
        ts = timestamps.detach().cpu().float().numpy()
    else:
        ts = np.array(timestamps, dtype=np.float32)

    # Normalize vectors
    norms = np.linalg.norm(vf, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-8
    normed_vf = vf / norms

    # Consecutive cosine similarities and distances
    cos_sims = np.sum(normed_vf[1:] * normed_vf[:-1], axis=-1)
    cos_dists = 1.0 - cos_sims  # Delta v_t in [0, 2]

    if len(cos_dists) == 0:
        return []

    if threshold is None:
        mean_d = float(np.mean(cos_dists))
        std_d = float(np.std(cos_dists))
        tau_v = max(0.20, mean_d + 0.5 * std_d)
    else:
        tau_v = float(threshold)

    transitions = []
    for idx, d in enumerate(cos_dists):
        if d > tau_v:
            transitions.append(float(ts[idx + 1]))

    return transitions


def apply_visual_snapping(
    predicted_boundaries: List[float],
    visual_transitions: List[float],
    window_sec: float = 45.0,
    first_sentence_time: float = 0.0
) -> List[float]:
    """
    Snap predicted boundaries to nearest visual transition (slide change) within +/- window_sec (default 45s).
    Silent Intro Guard: Never snap boundary 1 to 0.0s if first_sentence_time > 0.0s.
    """
    if not predicted_boundaries:
        return []
    if not visual_transitions:
        return sorted(list(set(predicted_boundaries)))

    snapped = []
    for idx, b in enumerate(predicted_boundaries):
        # Silent intro guard
        if b <= 0.0 and first_sentence_time > 0.0:
            b = first_sentence_time

        # Find closest visual transition within window_sec
        candidates = [v for v in visual_transitions if abs(b - v) <= window_sec]
        if candidates:
            best_v = min(candidates, key=lambda v: abs(b - v))
            # Protect intro: if best_v == 0.0 and first_sentence_time > 0.0, don't snap to 0.0
            if best_v <= 0.0 and first_sentence_time > 0.0:
                best_v = max(b, first_sentence_time)
            snapped.append(float(best_v))
        else:
            snapped.append(float(b))

    # Deduplicate and sort
    snapped = sorted(list(set(snapped)))
    filtered = []
    for s in snapped:
        if not filtered or (s - filtered[-1] >= 15.0):
            filtered.append(s)
    return filtered


class BaseChapteringModel(nn.Module):
    """
    Base class for all chaptering models with unified loss and boundary extraction.
    """
    def __init__(self, pos_weight: float = 4.0):
        super().__init__()
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Binary Cross Entropy with positive weight to handle class imbalance.
        """
        weight = torch.tensor([self.pos_weight], device=logits.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
        raw_loss = criterion(logits, targets)
        if mask is not None:
            raw_loss = raw_loss * mask.float()
            return raw_loss.sum() / (mask.float().sum() + 1e-8)
        return raw_loss.mean()

    def extract_boundaries(
        self,
        probabilities: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = 0.40,
        min_interval_sec: float = 30.0,
        smooth_sigma: float = 1.0,
        visual_transitions: Optional[List[List[float]]] = None,
        first_sentence_times: Optional[List[float]] = None,
        snap_window_sec: float = 45.0
    ) -> List[List[float]]:
        """
        Extract predicted boundary timestamps using 1D Non-Maximum Suppression (Peak Finding)
        with Gaussian kernel smoothing and optional Visual-Anchor Snapping (+/-45s).
        """
        batch_size, seq_len = probabilities.shape
        batch_boundaries: List[List[float]] = []

        for b in range(batch_size):
            raw_probs = probabilities[b].detach().cpu().numpy()
            times = timestamps[b].detach().cpu().numpy()
            valid_len = int(mask[b].sum().item()) if mask is not None else seq_len

            if valid_len <= 1:
                batch_boundaries.append([])
                continue

            p = raw_probs[:valid_len]
            t = times[:valid_len]

            # 1D Gaussian smoothing if enough points
            if smooth_sigma > 0 and len(p) >= 5:
                kernel_radius = int(np.ceil(2 * smooth_sigma))
                x_k = np.arange(-kernel_radius, kernel_radius + 1)
                kernel = np.exp(-0.5 * (x_k / smooth_sigma) ** 2)
                kernel /= kernel.sum()
                p_smooth = np.convolve(p, kernel, mode='same')
            else:
                p_smooth = p

            # Dynamic thresholding if threshold is None, else caller threshold
            if threshold is None:
                eff_threshold = max(0.20, float(np.mean(p_smooth) - 1.0 * np.std(p_smooth)))
            else:
                eff_threshold = float(threshold)

            # Detect local maxima
            candidate_peaks = []
            for i in range(len(p_smooth)):
                left = p_smooth[i - 1] if i > 0 else p_smooth[i]
                right = p_smooth[i + 1] if i < len(p_smooth) - 1 else p_smooth[i]

                # Check if local peak and exceeds threshold
                if p_smooth[i] >= eff_threshold and p_smooth[i] >= left and p_smooth[i] >= right:
                    candidate_peaks.append((float(p_smooth[i]), float(t[i]), i))

            # Zero-Boundary Guard (Red Team Finding 3): fallback to highest point if no peaks detected
            if not candidate_peaks and valid_len >= 3:
                inner_slice = p_smooth[1:valid_len - 1]
                if len(inner_slice) > 0:
                    max_idx = int(np.argmax(inner_slice)) + 1
                    candidate_peaks.append((float(p_smooth[max_idx]), float(t[max_idx]), max_idx))

            # 1D Greedy Non-Maximum Suppression (NMS)
            candidate_peaks.sort(key=lambda x: x[0], reverse=True)
            selected_times: List[float] = []

            for score, ts, idx in candidate_peaks:
                if all(abs(ts - sel_t) >= min_interval_sec for sel_t in selected_times):
                    selected_times.append(ts)

            # Sort chronologically
            selected_times.sort()

            # Apply Visual-Anchor Snapping if transitions provided
            if visual_transitions is not None and b < len(visual_transitions):
                vt = visual_transitions[b]
                fst = first_sentence_times[b] if first_sentence_times and b < len(first_sentence_times) else 0.0
                selected_times = apply_visual_snapping(
                    selected_times,
                    vt,
                    window_sec=snap_window_sec,
                    first_sentence_time=fst
                )

            batch_boundaries.append(selected_times)

        return batch_boundaries


class C1_TextOnlyChapterer(BaseChapteringModel):
    """
    C1: Text-only Baseline using sequence transcript embeddings + MLP boundary classifier.
    """
    def __init__(
        self,
        d_text: int = 384,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        pos_weight: float = 4.0
    ):
        super().__init__(pos_weight=pos_weight)
        self.proj = nn.Linear(d_text, d_hidden)
        self.pos_enc = PositionalEncoding(d_hidden, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=4,
            dim_feedforward=d_hidden * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        if batch.text_features is None:
            raise ValueError("C1 requires batch.text_features")

        x = self.proj(batch.text_features)
        x = self.pos_enc(x)
        src_key_padding_mask = ~batch.mask if batch.mask is not None else None
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(encoded).squeeze(-1)
        probs = torch.sigmoid(logits)

        loss = None
        if batch.targets is not None:
            loss = self.compute_loss(logits, batch.targets, batch.mask)

        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C2_AcousticChapterer(BaseChapteringModel):
    """C2: Acoustic Ablation Chapterer (Whisper log-mel features, d_ac = 32)."""
    def __init__(self, d_ac: int = 32, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 4.0):
        super().__init__(pos_weight=pos_weight)
        self.net = nn.Sequential(
            nn.Linear(d_ac, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        if batch.acoustic_features is None:
            raise ValueError("C2 requires batch.acoustic_features")
        logits = self.net(batch.acoustic_features).squeeze(-1)
        probs = torch.sigmoid(logits)
        loss = self.compute_loss(logits, batch.targets, batch.mask) if batch.targets is not None else None
        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C3_VisualChapterer(BaseChapteringModel):
    """C3: Visual Ablation Chapterer (DINOv2 ViT-S/14 embeddings)."""
    def __init__(self, d_vis: int = 384, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 4.0):
        super().__init__(pos_weight=pos_weight)
        self.net = nn.Sequential(
            nn.Linear(d_vis, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        if batch.visual_features is None:
            raise ValueError("C3 requires batch.visual_features")
        logits = self.net(batch.visual_features).squeeze(-1)
        probs = torch.sigmoid(logits)
        loss = self.compute_loss(logits, batch.targets, batch.mask) if batch.targets is not None else None
        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C4_OCRChapterer(BaseChapteringModel):
    """C4: OCR Slide Text Ablation Chapterer (PaddleOCR v3 embeddings)."""
    def __init__(self, d_ocr: int = 384, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 4.0):
        super().__init__(pos_weight=pos_weight)
        self.net = nn.Sequential(
            nn.Linear(d_ocr, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        if batch.ocr_features is None:
            raise ValueError("C4 requires batch.ocr_features")
        logits = self.net(batch.ocr_features).squeeze(-1)
        probs = torch.sigmoid(logits)
        loss = self.compute_loss(logits, batch.targets, batch.mask) if batch.targets is not None else None
        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C5_TemporalCrossAttentionTransformer(BaseChapteringModel):
    """
    C5: Proposed Temporal Multimodal Cross-Attention Transformer (Decisions Log D-T02).

    Architecture:
    - Shared projection to 256-dim space for Text, Visual, OCR, and Acoustic modalities (d_ac=32).
    - Decoupled Cross-Attention: Visual stream (DINOv2 ViT-S/14) acts as Query Anchor;
      Transcript, OCR, and Acoustic modalities act as Keys & Values.
    - 3 learned boundary tokens / queries.
    - 4-layer Transformer Encoder.
    - Boundary Classifier Head with Dynamic Moving-Avg NMS and Visual-Anchor Snapping (+/-45s).
    - BCEWithLogitsLoss with pos_weight = 4.0.
    """
    def __init__(
        self,
        d_text: int = 384,
        d_vis: int = 384,
        d_ocr: int = 384,
        d_ac: int = 32,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        num_boundary_tokens: int = 3,
        dropout: float = 0.1,
        pos_weight: float = 4.0
    ):
        super().__init__(pos_weight=pos_weight)
        self.d_model = d_model

        # 1. Modality Projection Layers to shared 256-dim space
        self.proj_text = nn.Linear(d_text, d_model)
        self.proj_vis = nn.Linear(d_vis, d_model)
        self.proj_ocr = nn.Linear(d_ocr, d_model)
        self.proj_ac = nn.Linear(d_ac, d_model)

        # 2. Modality Type Embeddings & LayerNorm
        self.modality_embed = nn.Embedding(4, d_model) # 0: text, 1: vis, 2: ocr, 3: ac
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 3. Learned Boundary Tokens (3 tokens per D-T02)
        self.boundary_tokens = nn.Parameter(torch.randn(1, num_boundary_tokens, d_model) * 0.02)

        # 4. Decoupled Multi-Head Cross-Attention (Visual Query Anchor <-> Context Keys & Values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_post_cross = nn.LayerNorm(d_model)

        # 5. Temporal Transformer Self-Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.multimodal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 6. Temporal Boundary Classifier Head
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        batch: ChapteringBatch,
        threshold: float = 0.40,
        snap_to_visual: bool = True
    ) -> ChapteringOutput:
        B, T = batch.timestamps.shape
        device = batch.timestamps.device

        # 1. Project available modalities
        v_feat = self.proj_vis(batch.visual_features) + self.modality_embed(torch.tensor(1, device=device)) if batch.visual_features is not None else None
        t_feat = self.proj_text(batch.text_features) + self.modality_embed(torch.tensor(0, device=device)) if batch.text_features is not None else None
        o_feat = self.proj_ocr(batch.ocr_features) + self.modality_embed(torch.tensor(2, device=device)) if batch.ocr_features is not None else None
        a_feat = self.proj_ac(batch.acoustic_features) + self.modality_embed(torch.tensor(3, device=device)) if batch.acoustic_features is not None else None

        # Query Anchor: Visual stream preferred, fallback to text/ocr/ac if visual missing
        if v_feat is not None:
            query = v_feat
        elif t_feat is not None:
            query = t_feat
        elif o_feat is not None:
            query = o_feat
        elif a_feat is not None:
            query = a_feat
        else:
            raise ValueError("At least one modality feature must be provided in ChapteringBatch")

        # Context streams for Keys & Values (Text, OCR, Acoustic)
        ctx_list = [f for f in [t_feat, o_feat, a_feat] if f is not None]
        if not ctx_list:
            ctx_list = [v_feat]

        # Key/Value sequence: concatenate across modalities along sequence dimension
        kv_seq = torch.cat(ctx_list, dim=1)  # [B, num_ctx * T, d_model]

        if batch.mask is not None:
            kv_mask = torch.cat([batch.mask] * len(ctx_list), dim=1) # [B, num_ctx * T]
            key_padding_mask = ~kv_mask
        else:
            key_padding_mask = None

        # Decoupled Cross-Attention
        q_norm = self.ln_q(query)
        kv_norm = self.ln_kv(kv_seq)
        cross_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask
        )
        fused = self.ln_post_cross(query + cross_out)
        fused = self.pos_enc(fused)

        # Prepend learned boundary query tokens
        b_tokens = self.boundary_tokens.expand(B, -1, -1) # [B, 3, d_model]
        full_seq = torch.cat([b_tokens, fused], dim=1)    # [B, 3 + T, d_model]

        # Temporal self-attention encoding
        if batch.mask is not None:
            b_mask = torch.ones((B, 3), dtype=torch.bool, device=device)
            full_mask = torch.cat([b_mask, batch.mask], dim=1) # [B, 3 + T]
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None

        encoded = self.multimodal_encoder(full_seq, src_key_padding_mask=src_key_padding_mask)

        # Extract temporal sequence tokens (skip boundary query tokens)
        temporal_encoded = encoded[:, 3:, :] # [B, T, d_model]
        logits = self.boundary_head(temporal_encoded).squeeze(-1) # [B, T]
        probs = torch.sigmoid(logits)

        loss = None
        if batch.targets is not None:
            loss = self.compute_loss(logits, batch.targets, batch.mask)

        # Extract visual transitions for snapping
        visual_trans_batch = None
        if snap_to_visual and batch.visual_features is not None:
            visual_trans_batch = []
            for b in range(B):
                v_len = int(batch.mask[b].sum().item()) if batch.mask is not None else T
                vt = extract_visual_transitions(
                    batch.visual_features[b, :v_len],
                    batch.timestamps[b, :v_len]
                )
                visual_trans_batch.append(vt)

        boundaries = self.extract_boundaries(
            probs,
            batch.timestamps,
            batch.mask,
            threshold=threshold,
            visual_transitions=visual_trans_batch
        )
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C6_LateFusionChapterer(BaseChapteringModel):
    """
    C6: Full Late Fusion Baseline (Ablation per decisions-log.md D-T02).

    Same modalities as C5, but uses direct concatenation-only without cross-attention transformer.
    """
    def __init__(
        self,
        d_text: int = 384,
        d_vis: int = 384,
        d_ocr: int = 384,
        d_ac: int = 32,
        d_hidden: int = 256,
        dropout: float = 0.1,
        pos_weight: float = 4.0
    ):
        super().__init__(pos_weight=pos_weight)
        self.proj_text = nn.Linear(d_text, d_hidden)
        self.proj_vis = nn.Linear(d_vis, d_hidden)
        self.proj_ocr = nn.Linear(d_ocr, d_hidden)
        self.proj_ac = nn.Linear(d_ac, d_hidden)

        # Concatenate 4 modalities: 4 * 256 = 1024-dim
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden * 4, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        B, T = batch.timestamps.shape
        device = batch.timestamps.device

        t_feat = self.proj_text(batch.text_features) if batch.text_features is not None else torch.zeros((B, T, 256), device=device)
        v_feat = self.proj_vis(batch.visual_features) if batch.visual_features is not None else torch.zeros((B, T, 256), device=device)
        o_feat = self.proj_ocr(batch.ocr_features) if batch.ocr_features is not None else torch.zeros((B, T, 256), device=device)
        a_feat = self.proj_ac(batch.acoustic_features) if batch.acoustic_features is not None else torch.zeros((B, T, 256), device=device)

        concat_feat = torch.cat([t_feat, v_feat, o_feat, a_feat], dim=-1) # [B, T, 1024]
        logits = self.mlp(concat_feat).squeeze(-1) # [B, T]
        probs = torch.sigmoid(logits)

        loss = None
        if batch.targets is not None:
            loss = self.compute_loss(logits, batch.targets, batch.mask)

        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)
