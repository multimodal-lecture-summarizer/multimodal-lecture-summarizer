"""
Multimodal Chaptering Models (C1 - C6) for Scientific Lecture Video Understanding.

Implements architectures defined in decisions-log.md (D-T02):
- C1: Text-only Baseline
- C2: Acoustic Ablation
- C3: Visual Ablation (DINOv2 ViT-S/14)
- C4: OCR Ablation (PaddleOCR v3)
- C5: Proposed Temporal Cross-Attention Transformer (4 layers, 256-dim, 3 boundary tokens)
- C6: Late Fusion Baseline (Concatenation-only)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
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


class BaseChapteringModel(nn.Module):
    """
    Base class for all chaptering models with unified loss and boundary extraction.
    """
    def __init__(self, pos_weight: float = 8.0):
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
        threshold: float = 0.5,
        min_interval_sec: float = 15.0
    ) -> List[List[float]]:
        """
        Extract predicted boundary timestamps from probability sequence with non-maximum suppression.
        """
        batch_size, seq_len = probabilities.shape
        batch_boundaries: List[List[float]] = []

        for b in range(batch_size):
            probs = probabilities[b].detach().cpu().numpy()
            times = timestamps[b].detach().cpu().numpy()
            valid_len = int(mask[b].sum().item()) if mask is not None else seq_len

            sample_boundaries = []
            last_ts = -min_interval_sec

            for t_idx in range(valid_len):
                if probs[t_idx] >= threshold:
                    cur_ts = float(times[t_idx])
                    if cur_ts - last_ts >= min_interval_sec:
                        sample_boundaries.append(cur_ts)
                        last_ts = cur_ts
            batch_boundaries.append(sample_boundaries)

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
        pos_weight: float = 8.0
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
            raise ValueError("C1_TextOnlyChapterer requires batch.text_features")
        
        x = self.proj(batch.text_features)
        x = self.pos_enc(x)
        
        src_key_padding_mask = ~batch.mask if batch.mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        logits = self.head(x).squeeze(-1) # [B, T]
        probs = torch.sigmoid(logits)

        loss = None
        if batch.targets is not None:
            loss = self.compute_loss(logits, batch.targets, batch.mask)

        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
        return ChapteringOutput(logits=logits, probabilities=probs, loss=loss, predicted_boundaries=boundaries)


class C2_AcousticChapterer(BaseChapteringModel):
    """C2: Acoustic Ablation Chapterer."""
    def __init__(self, d_ac: int = 64, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 8.0):
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
    def __init__(self, d_vis: int = 384, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 8.0):
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
    def __init__(self, d_ocr: int = 384, d_hidden: int = 256, dropout: float = 0.1, pos_weight: float = 8.0):
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
    C5: Proposed Temporal Multimodal Encoder (Decisions Log D-T02).
    
    Architecture:
    - Shared projection to 256-dim space for Text, Visual, OCR, and Acoustic modalities.
    - 3 learned boundary tokens / queries.
    - 4-layer Cross-Attention + Self-Attention blocks.
    - Binary Cross-Entropy loss with positive class weighting.
    """
    def __init__(
        self,
        d_text: int = 384,
        d_vis: int = 384,
        d_ocr: int = 384,
        d_ac: int = 64,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        num_boundary_tokens: int = 3,
        dropout: float = 0.1,
        pos_weight: float = 8.0
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
        
        # 4. 4-Layer Multimodal Cross-Attention & Self-Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.multimodal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. Temporal Boundary Classifier Head
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, batch: ChapteringBatch, threshold: float = 0.5) -> ChapteringOutput:
        B, T = batch.timestamps.shape
        device = batch.timestamps.device
        
        # Fuse input representations with modality embeddings
        feats = []
        if batch.text_features is not None:
            t_feat = self.proj_text(batch.text_features) + self.modality_embed(torch.tensor(0, device=device))
            feats.append(t_feat)
        if batch.visual_features is not None:
            v_feat = self.proj_vis(batch.visual_features) + self.modality_embed(torch.tensor(1, device=device))
            feats.append(v_feat)
        if batch.ocr_features is not None:
            o_feat = self.proj_ocr(batch.ocr_features) + self.modality_embed(torch.tensor(2, device=device))
            feats.append(o_feat)
        if batch.acoustic_features is not None:
            a_feat = self.proj_ac(batch.acoustic_features) + self.modality_embed(torch.tensor(3, device=device))
            feats.append(a_feat)
            
        if not feats:
            raise ValueError("At least one modality feature must be provided in ChapteringBatch")
            
        # Combine across modalities by elementwise summation / mean fusion at each timestep
        fused = torch.stack(feats, dim=0).mean(dim=0) # [B, T, d_model]
        fused = self.pos_enc(fused)
        
        # Prepend learned boundary query tokens
        b_tokens = self.boundary_tokens.expand(B, -1, -1) # [B, 3, d_model]
        full_seq = torch.cat([b_tokens, fused], dim=1)    # [B, 3 + T, d_model]
        
        # Build attention mask
        if batch.mask is not None:
            b_mask = torch.ones((B, 3), dtype=torch.bool, device=device)
            full_mask = torch.cat([b_mask, batch.mask], dim=1) # [B, 3 + T]
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None
            
        # Cross/Self attention encoding
        encoded = self.multimodal_encoder(full_seq, src_key_padding_mask=src_key_padding_mask)
        
        # Extract temporal sequence tokens (skip the 3 boundary queries)
        temporal_encoded = encoded[:, 3:, :] # [B, T, d_model]
        
        logits = self.boundary_head(temporal_encoded).squeeze(-1) # [B, T]
        probs = torch.sigmoid(logits)
        
        loss = None
        if batch.targets is not None:
            loss = self.compute_loss(logits, batch.targets, batch.mask)
            
        boundaries = self.extract_boundaries(probs, batch.timestamps, batch.mask, threshold)
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
        d_ac: int = 64,
        d_hidden: int = 256,
        dropout: float = 0.1,
        pos_weight: float = 8.0
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
