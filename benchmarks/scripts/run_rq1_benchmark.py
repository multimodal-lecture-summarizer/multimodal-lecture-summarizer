"""
RQ1 Chaptering & Temporal Representation Benchmark Runner (End-to-End Real Data).
Trains and evaluates C1-C6 models across 3 seeds on real cached multimodal lecture features.
Computes Collar F1@3s/5s/10s, Pk, WindowDiff, Paired Bootstrap 95% CIs, and Holm-Bonferroni p-values.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from benchmarks.models.chaptering import (
    ChapteringBatch,
    C1_TextOnlyChapterer,
    C2_AcousticChapterer,
    C3_VisualChapterer,
    C4_OCRChapterer,
    C5_TemporalCrossAttentionTransformer,
    C6_LateFusionChapterer,
)
from benchmarks.data.dataset import create_lecture_splits, collate_lecture_batches
from benchmarks.metrics.chapter_metrics import compute_all_chapter_metrics
from benchmarks.metrics.statistics import holm_bonferroni_family


def train_and_eval_model(
    model_fn,
    model_name: str,
    train_batch: ChapteringBatch,
    val_batch: ChapteringBatch,
    test_batch: ChapteringBatch,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> Dict[str, Any]:
    model = model_fn().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_batch)
        loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_out = model(val_batch)
            v_loss = v_out.loss.item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_out = model(test_batch, threshold=0.40)
        pred_boundaries = test_out.predicted_boundaries

    # Compute metrics per sample
    f1_3s_list, f1_5s_list, f1_10s_list, pk_list, wd_list = [], [], [], [], []
    num_samples = test_batch.timestamps.shape[0]

    for b in range(num_samples):
        times = test_batch.timestamps[b].cpu().numpy()
        targets = test_batch.targets[b].cpu().numpy()
        valid_len = int(test_batch.mask[b].sum().item()) if test_batch.mask is not None else len(times)
        
        gold_ts = [float(times[i]) for i in range(valid_len) if targets[i] == 1.0]
        preds = pred_boundaries[b]
        dur = float(times[valid_len - 1]) if valid_len > 0 else 600.0

        m = compute_all_chapter_metrics(gold_ts, preds, video_duration_sec=dur)
        f1_3s_list.append(m["collar_f1_3s"])
        f1_5s_list.append(m["collar_f1_5s"])
        f1_10s_list.append(m["collar_f1_10s"])
        pk_list.append(m["pk"])
        wd_list.append(m["window_diff"])

    return {
        "model_name": model_name,
        "f1_3s": f1_3s_list,
        "f1_5s": f1_5s_list,
        "f1_10s": f1_10s_list,
        "pk": pk_list,
        "wd": wd_list,
        "mean_f1_3s": float(np.mean(f1_3s_list)),
        "mean_f1_5s": float(np.mean(f1_5s_list)),
        "mean_pk": float(np.mean(pk_list)),
        "mean_wd": float(np.mean(wd_list)),
        "best_val_loss": best_val_loss,
        "predictions": pred_boundaries
    }


def run_rq1_benchmark(data_dir: str = "benchmarks/data/cached_features", seeds: List[int] = [42, 1337, 2026], device: str = "cpu"):
    print("=" * 80)
    print(f"STARTING RQ1 CHAPTERING BENCHMARK (Seeds: {seeds}, Device: {device})")
    print("=" * 80)

    model_factories = {
        "C1 (Text-only)": lambda: C1_TextOnlyChapterer(d_text=384, d_hidden=256, n_layers=2),
        "C2 (Acoustic)": lambda: C2_AcousticChapterer(d_ac=32, d_hidden=256),
        "C3 (Visual DINOv2)": lambda: C3_VisualChapterer(d_vis=384, d_hidden=256),
        "C4 (OCR Slide)": lambda: C4_OCRChapterer(d_ocr=384, d_hidden=256),
        "C5 (Cross-Attention Proposed)": lambda: C5_TemporalCrossAttentionTransformer(d_text=384, d_vis=384, d_ocr=384, d_ac=32, d_model=256, n_layers=4, n_heads=8, num_boundary_tokens=3),
        "C6 (Late Fusion Baseline)": lambda: C6_LateFusionChapterer(d_text=384, d_vis=384, d_ocr=384, d_ac=32, d_hidden=256),
    }

    all_seed_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_factories}

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        train_set, val_set, test_set = create_lecture_splits(data_dir=data_dir, train_ratio=0.6, val_ratio=0.2, seed=seed)
        
        train_b = collate_lecture_batches([train_set[i] for i in range(len(train_set))])
        val_b = collate_lecture_batches([val_set[i] for i in range(len(val_set))])
        test_b = collate_lecture_batches([test_set[i] for i in range(len(test_set))])

        for name, fn in model_factories.items():
            torch.manual_seed(seed)
            np.random.seed(seed)
            res = train_and_eval_model(fn, name, train_b, val_b, test_b, epochs=35, device=device)
            all_seed_results[name].append(res)
            print(f"  {name:30s} | Collar F1@3s: {res['mean_f1_3s']:.4f} | F1@5s: {res['mean_f1_5s']:.4f} | Pk: {res['mean_pk']:.4f} | WD: {res['mean_wd']:.4f}")

    # Aggregate across seeds
    summary_rows = []
    c5_f1_scores = []
    c1_f1_scores = []
    c2_f1_scores = []
    c3_f1_scores = []
    c4_f1_scores = []

    for name in model_factories:
        seed_f1_3s = [r["mean_f1_3s"] for r in all_seed_results[name]]
        seed_f1_5s = [r["mean_f1_5s"] for r in all_seed_results[name]]
        seed_pk = [r["mean_pk"] for r in all_seed_results[name]]
        seed_wd = [r["mean_wd"] for r in all_seed_results[name]]

        summary_rows.append({
            "Model Variant": name,
            "Collar F1@3s": f"{np.mean(seed_f1_3s):.4f} +/- {np.std(seed_f1_3s):.4f}",
            "Collar F1@5s": f"{np.mean(seed_f1_5s):.4f} +/- {np.std(seed_f1_5s):.4f}",
            "Pk Error": f"{np.mean(seed_pk):.4f}",
            "WindowDiff": f"{np.mean(seed_wd):.4f}",
        })

        # Collect paired sample-level scores across all seeds for bootstrap CI
        flat_f1 = [score for r in all_seed_results[name] for score in r["f1_3s"]]
        if "C5" in name:
            c5_f1_scores = flat_f1
        elif "C1" in name:
            c1_f1_scores = flat_f1
        elif "C2" in name:
            c2_f1_scores = flat_f1
        elif "C3" in name:
            c3_f1_scores = flat_f1
        elif "C4" in name:
            c4_f1_scores = flat_f1

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("RQ1 CHAPTERING BENCHMARK RESULTS (AGGREGATED ACROSS 3 SEEDS)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # Statistical Hypothesis Testing: Paired Bootstrap 95% CI & Holm-Bonferroni
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL HYPOTHESIS TESTING (D-T07 Family of 4 Comparisons)")
    print("=" * 80)

    deltas = {
        "C5 vs C1 (Text Baseline)": np.array(c5_f1_scores) - np.array(c1_f1_scores),
        "C5 vs C2 (Acoustic Ablation)": np.array(c5_f1_scores) - np.array(c2_f1_scores),
        "C5 vs C3 (Visual Ablation)": np.array(c5_f1_scores) - np.array(c3_f1_scores),
        "C5 vs C4 (OCR Ablation)": np.array(c5_f1_scores) - np.array(c4_f1_scores),
    }

    hb_results = holm_bonferroni_family(deltas, alpha=0.05, n_resamples=1000, seed=42)
    stat_rows = []
    for label, res in hb_results.items():
        stat_rows.append({
            "Comparison": label,
            "Delta (C5 - Baseline)": f"{res.mean_diff:+.4f}",
            "95% Bootstrap CI": f"[{res.ci_95[0]:+.4f}, {res.ci_95[1]:+.4f}]",
            "Cohen's d": f"{res.cohens_d:+.4f}",
            "Raw p-value": f"{res.raw_p_value:.4f}",
            "Holm-Adj p": f"{res.corrected_p_value:.4f}",
            "Significant (alpha=0.05)": "YES (Reject H0)" if res.reject_null else "NO (Fail to Reject)",
        })

    df_stats = pd.DataFrame(stat_rows)
    print(df_stats.to_string(index=False))

    # Save output artifacts
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_json(out_dir / "rq1_chaptering_summary.json", orient="records", indent=2)
    df_stats.to_json(out_dir / "rq1_chaptering_statistics.json", orient="records", indent=2)
    print(f"\n[OK] Saved benchmark artifacts to {out_dir}")

    return {"summary": df_summary.to_dict(orient="records"), "statistics": df_stats.to_dict(orient="records")}


if __name__ == "__main__":
    run_rq1_benchmark()
