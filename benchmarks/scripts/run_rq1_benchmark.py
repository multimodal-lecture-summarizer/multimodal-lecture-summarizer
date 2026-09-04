"""
RQ1 Chaptering & Temporal Representation Benchmark Runner (End-to-End Real Data).
Trains and evaluates C1-C6 models across 3 seeds on real cached multimodal lecture features.
Computes Collar F1@3s/5s/10s, Pk, WindowDiff, Paired Bootstrap 95% CIs, and Holm-Bonferroni p-values.
Outputs standardized reports/rq1_benchmark_results.json conforming to Phase 2 research specifications.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.models.chaptering import (
    ChapteringBatch,
    C1_TextOnlyChapterer,
    C2_AcousticChapterer,
    C3_VisualChapterer,
    C4_OCRChapterer,
    C5_TemporalCrossAttentionTransformer,
    C6_LateFusionChapterer,
    apply_visual_snapping,
    extract_visual_transitions,
)
from benchmarks.data.dataset import create_lecture_splits, collate_lecture_batches
from benchmarks.metrics.chapter_metrics import compute_all_chapter_metrics
from benchmarks.metrics.statistics import holm_bonferroni_family, paired_bootstrap_ci


def seed_everything(seed: int):
    """Ensure complete reproducibility and zero cross-seed leakage."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()


def train_and_eval_model(
    model_fn,
    model_name: str,
    train_batch: ChapteringBatch,
    val_batch: ChapteringBatch,
    test_batch: ChapteringBatch,
    epochs: int = 35,
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
        "best_state": best_state,
        "predictions": pred_boundaries
    }


def evaluate_c5_delta_ablation(
    c5_model: nn.Module,
    test_batch: ChapteringBatch,
    deltas: List[float] = [15.0, 30.0, 45.0, 60.0]
) -> Dict[str, Dict[str, float]]:
    """Ablation sensitivity sweep over visual-snapping radius delta."""
    results = {}
    B, T = test_batch.timestamps.shape
    c5_model.eval()

    with torch.no_grad():
        out = c5_model(test_batch, threshold=0.40, snap_to_visual=False)
        raw_boundaries = out.predicted_boundaries

    for delta in deltas:
        wd_list, f1_5s_list = [], []
        for b in range(B):
            times = test_batch.timestamps[b].cpu().numpy()
            targets = test_batch.targets[b].cpu().numpy()
            valid_len = int(test_batch.mask[b].sum().item()) if test_batch.mask is not None else len(times)

            gold_ts = [float(times[i]) for i in range(valid_len) if targets[i] == 1.0]
            raw_b = raw_boundaries[b]
            dur = float(times[valid_len - 1]) if valid_len > 0 else 600.0

            # Extract visual transitions for snapping
            if test_batch.visual_features is not None:
                vt = extract_visual_transitions(
                    test_batch.visual_features[b, :valid_len],
                    test_batch.timestamps[b, :valid_len]
                )
                snapped_b = apply_visual_snapping(raw_b, vt, window_sec=delta)
            else:
                snapped_b = raw_b

            m = compute_all_chapter_metrics(gold_ts, snapped_b, video_duration_sec=dur)
            wd_list.append(m["window_diff"])
            f1_5s_list.append(m["collar_f1_5s"])

        label = f"{int(delta)}s"
        results[label] = {
            "window_diff": float(round(np.mean(wd_list), 4)),
            "f1_5s": float(round(np.mean(f1_5s_list), 4))
        }
    return results


def run_rq1_benchmark(
    data_dir: str = "benchmarks/data/cached_features",
    seeds: List[int] = [42, 1337, 2026],
    device: str = "cpu"
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"STARTING RQ1 CHAPTERING BENCHMARK (Seeds: {seeds}, Device: {device})")
    print("=" * 80)

    model_factories = {
        "C1 (Text-only)": lambda: C1_TextOnlyChapterer(d_text=384, d_hidden=256, n_layers=2),
        "C2 (Acoustic)": lambda: C2_AcousticChapterer(d_ac=32, d_hidden=256),
        "C3 (Visual DINOv2)": lambda: C3_VisualChapterer(d_vis=384, d_hidden=256),
        "C4 (OCR Slide)": lambda: C4_OCRChapterer(d_ocr=384, d_hidden=256),
        "C5 (Cross-Attention Proposed)": lambda: C5_TemporalCrossAttentionTransformer(
            d_text=384, d_vis=384, d_ocr=384, d_ac=32, d_model=256,
            n_layers=4, n_heads=8, num_boundary_tokens=3, pos_weight=4.0
        ),
        "C6 (Late Fusion Baseline)": lambda: C6_LateFusionChapterer(d_text=384, d_vis=384, d_ocr=384, d_ac=32, d_hidden=256),
    }

    all_seed_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_factories}
    best_c5_state = None
    best_c5_val_loss = float("inf")
    delta_sensitivity_results = {}

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        seed_everything(seed)
        train_set, val_set, test_set = create_lecture_splits(data_dir=data_dir, train_ratio=0.6, val_ratio=0.2, seed=seed)

        train_b = collate_lecture_batches([train_set[i] for i in range(len(train_set))])
        val_b = collate_lecture_batches([val_set[i] for i in range(len(val_set))])
        test_b = collate_lecture_batches([test_set[i] for i in range(len(test_set))])

        for name, fn in model_factories.items():
            seed_everything(seed)
            res = train_and_eval_model(fn, name, train_b, val_b, test_b, epochs=35, device=device)
            all_seed_results[name].append(res)
            print(f"  {name:30s} | Collar F1@3s: {res['mean_f1_3s']:.4f} | F1@5s: {res['mean_f1_5s']:.4f} | Pk: {res['mean_pk']:.4f} | WD: {res['mean_wd']:.4f}")

            # Track best C5 model
            if "C5" in name and res["best_val_loss"] < best_c5_val_loss:
                best_c5_val_loss = res["best_val_loss"]
                best_c5_state = res["best_state"]

        # Run delta ablation on the first seed
        if seed == seeds[0]:
            c5_temp = model_factories["C5 (Cross-Attention Proposed)"]().to(device)
            if all_seed_results["C5 (Cross-Attention Proposed)"][-1]["best_state"] is not None:
                c5_temp.load_state_dict(all_seed_results["C5 (Cross-Attention Proposed)"][-1]["best_state"])
            delta_sensitivity_results = evaluate_c5_delta_ablation(c5_temp, test_b)

    # Save best C5 checkpoint
    if best_c5_state is not None:
        ckpt_path = Path("checkpoints/c5_real.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_c5_state, ckpt_path)
        print(f"\n[OK] Saved best C5 checkpoint to {ckpt_path}")

    # Build standardized JSON results
    metrics_by_model: Dict[str, Any] = {}
    summary_rows = []

    model_key_map = {
        "C1 (Text-only)": "C1",
        "C2 (Acoustic)": "C2",
        "C3 (Visual DINOv2)": "C3",
        "C4 (OCR Slide)": "C4",
        "C5 (Cross-Attention Proposed)": "C5",
        "C6 (Late Fusion Baseline)": "C6",
    }

    flat_scores: Dict[str, Dict[str, List[float]]] = {
        name: {"f1_3s": [], "f1_5s": [], "f1_10s": [], "pk": [], "wd": []}
        for name in model_factories
    }

    for name in model_factories:
        m_key = model_key_map[name]
        for r in all_seed_results[name]:
            flat_scores[name]["f1_3s"].extend(r["f1_3s"])
            flat_scores[name]["f1_5s"].extend(r["f1_5s"])
            flat_scores[name]["f1_10s"].extend(r["f1_10s"])
            flat_scores[name]["pk"].extend(r["pk"])
            flat_scores[name]["wd"].extend(r["wd"])

        def _stats(arr):
            ci = paired_bootstrap_ci(arr, n_resamples=1000, seed=42)
            return {
                "mean": float(round(np.mean(arr), 4)),
                "std": float(round(np.std(arr), 4)),
                "ci_95": [float(round(ci.ci_lower, 4)), float(round(ci.ci_upper, 4))]
            }

        metrics_by_model[m_key] = {
            "collar_f1_3s": _stats(flat_scores[name]["f1_3s"]),
            "collar_f1_5s": _stats(flat_scores[name]["f1_5s"]),
            "collar_f1_10s": _stats(flat_scores[name]["f1_10s"]),
            "pk": _stats(flat_scores[name]["pk"]),
            "window_diff": _stats(flat_scores[name]["wd"]),
        }

        summary_rows.append({
            "Model Variant": name,
            "Collar F1@3s": f"{metrics_by_model[m_key]['collar_f1_3s']['mean']:.4f} +/- {metrics_by_model[m_key]['collar_f1_3s']['std']:.4f}",
            "Collar F1@5s": f"{metrics_by_model[m_key]['collar_f1_5s']['mean']:.4f} +/- {metrics_by_model[m_key]['collar_f1_5s']['std']:.4f}",
            "Pk Error": f"{metrics_by_model[m_key]['pk']['mean']:.4f}",
            "WindowDiff": f"{metrics_by_model[m_key]['window_diff']['mean']:.4f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("RQ1 CHAPTERING BENCHMARK RESULTS (AGGREGATED ACROSS 3 SEEDS)")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # Statistical Hypothesis Testing
    c5_f1 = np.array(flat_scores["C5 (Cross-Attention Proposed)"]["f1_5s"])
    deltas = {
        "C5 vs C1 (Text Baseline)": c5_f1 - np.array(flat_scores["C1 (Text-only)"]["f1_5s"]),
        "C5 vs C2 (Acoustic Ablation)": c5_f1 - np.array(flat_scores["C2 (Acoustic)"]["f1_5s"]),
        "C5 vs C3 (Visual Ablation)": c5_f1 - np.array(flat_scores["C3 (Visual DINOv2)"]["f1_5s"]),
        "C5 vs C4 (OCR Ablation)": c5_f1 - np.array(flat_scores["C4 (OCR Slide)"]["f1_5s"]),
        "C5 vs C6 (Late Fusion Baseline)": c5_f1 - np.array(flat_scores["C6 (Late Fusion Baseline)"]["f1_5s"]),
    }

    hb_results = holm_bonferroni_family(deltas, alpha=0.05, n_resamples=1000, seed=42)
    stat_rows = []
    hypothesis_json = {}

    for label, res in hb_results.items():
        stat_rows.append({
            "Comparison": label,
            "Delta (F1@5s)": f"{res.mean_diff:+.4f}",
            "95% Bootstrap CI": f"[{res.ci_95[0]:+.4f}, {res.ci_95[1]:+.4f}]",
            "Cohen's d": f"{res.cohens_d:+.4f}",
            "Raw p-value": f"{res.raw_p_value:.4f}",
            "Holm-Adj p": f"{res.corrected_p_value:.4f}",
            "Significant (alpha=0.05)": "YES (Reject H0)" if res.reject_null else "NO (Fail to Reject)",
        })
        key_name = label.split()[0] + "_" + label.split()[2]
        hypothesis_json[key_name] = {
            "comparison": label,
            "mean_diff": float(round(res.mean_diff, 4)),
            "ci_95": [float(round(res.ci_95[0], 4)), float(round(res.ci_95[1], 4))],
            "cohen_d": float(round(res.cohens_d, 4)),
            "raw_p_value": float(round(res.raw_p_value, 4)),
            "corrected_p_value": float(round(res.corrected_p_value, 4)),
            "reject_null": bool(res.reject_null)
        }

    df_stats = pd.DataFrame(stat_rows)
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL HYPOTHESIS TESTING (Holm-Bonferroni Corrected)")
    print("=" * 80)
    print(df_stats.to_string(index=False))

    # Standardized Report Structure
    final_report = {
        "experiment": "RQ1_Chaptering",
        "seeds": seeds,
        "hardware_target": "Tesla T4 (16GB VRAM) / CPU Verification",
        "metrics": metrics_by_model,
        "hypothesis_tests": hypothesis_json,
        "ablation_delta_sensitivity": delta_sensitivity_results
    }

    # Save output artifacts
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "rq1_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    df_summary.to_json(out_dir / "rq1_chaptering_summary.json", orient="records", indent=2)
    df_stats.to_json(out_dir / "rq1_chaptering_statistics.json", orient="records", indent=2)
    print(f"\n[OK] Saved standardized report to {reports_dir / 'rq1_benchmark_results.json'}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RQ1 Chaptering Benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026], help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="benchmarks/data/cached_features", help="Path to cached features")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    run_rq1_benchmark(data_dir=args.data_dir, seeds=args.seeds, device=args.device)
