"""Aggregate evaluation rows and build production vs candidate comparisons."""

from __future__ import annotations

from typing import Any, Callable, Iterable


def mean_metric(rows: Iterable[dict[str, Any]], key: str) -> float:
    nums = [float(r[key]) for r in rows if key in r and r[key] is not None and r[key] == r[key]]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def group_mean(rows: Iterable[dict[str, Any]], group_key: str, metric_key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        g = str(r.get(group_key) or "")
        v = r.get(metric_key)
        if not g or v is None or v != v:
            continue
        buckets.setdefault(g, []).append(float(v))
    return {g: sum(vs) / len(vs) for g, vs in buckets.items() if vs}


def compare_models(
    rows: Iterable[dict[str, Any]],
    *,
    model_key: str,
    production_model: str,
    metrics: dict[str, Callable[[float, float], float]],
    higher_better: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Build one comparison row per (metric, candidate model).

    `metrics`: metric_name -> delta function(delta = candidate - production).
    """
    hb = higher_better or {}
    by_model: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        model = str(r.get(model_key) or "")
        if not model:
            continue
        by_model.setdefault(model, {})
        for metric in metrics:
            v = r.get(metric)
            if v is None or v != v:
                continue
            by_model[model].setdefault(metric, []).append(float(v))

    prod = by_model.get(production_model, {})
    out: list[dict[str, Any]] = []
    for model, metric_lists in sorted(by_model.items()):
        if model == production_model:
            continue
        for metric, delta_fn in metrics.items():
            prod_vals = prod.get(metric, [])
            cand_vals = metric_lists.get(metric, [])
            if not prod_vals or not cand_vals:
                continue
            prod_mean = sum(prod_vals) / len(prod_vals)
            cand_mean = sum(cand_vals) / len(cand_vals)
            delta = cand_mean - prod_mean
            better = delta_fn(delta)
            out.append(
                {
                    "model": model,
                    "production_model": production_model,
                    "metric": metric,
                    "production": prod_mean,
                    "candidate": cand_mean,
                    "delta": delta,
                    "better_than_production": better,
                    "higher_better": hb.get(metric, True),
                }
            )
    return out


def build_asr_model_comparison(
    asr_rows: list[dict[str, Any]],
    *,
    production_model: str = "faster-whisper-base.en",
) -> list[dict[str, Any]]:
    return compare_models(
        asr_rows,
        model_key="model",
        production_model=production_model,
        metrics={
            "wer": lambda d: d < 0,
            "cer": lambda d: d < 0,
            "rtf": lambda d: d < 0,
        },
        higher_better={"wer": False, "cer": False, "rtf": False},
    )


def summarize_dataset(rows: Iterable[dict[str, Any]], dataset: str = "TED") -> dict[str, Any]:
    rows = [r for r in rows if (r.get("dataset") or "TED") == dataset]
    return {
        "dataset": dataset,
        "n": len(rows),
        "wer_mean": mean_metric(rows, "wer"),
        "cer_mean": mean_metric(rows, "cer"),
        "rtf_mean": mean_metric(rows, "rtf"),
        "scene_f1_mean": mean_metric(rows, "f1") if rows and "n_ref" in rows[0] else float("nan"),
    }


def _dataset_label(rows: list[dict[str, Any]], default: str = "TED") -> str:
    if not rows:
        return default
    return str(rows[0].get("dataset") or default)


def aggregate_asr_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (dataset, model) — không tách theo talk."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        model = str(r.get("model") or "")
        if not model:
            continue
        buckets.setdefault((ds, model), []).append(r)
    out: list[dict[str, Any]] = []
    for (ds, model), items in sorted(buckets.items()):
        out.append(
            {
                "dataset": ds,
                "model": model,
                "n_clips": len(items),
                "wer": mean_metric(items, "wer"),
                "cer": mean_metric(items, "cer"),
                "rtf": mean_metric(items, "rtf"),
            }
        )
    return out


def aggregate_timeline_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gộp mọi talk trong cùng dataset thành một dòng."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        buckets.setdefault(ds, []).append(r)
    out: list[dict[str, Any]] = []
    for ds, items in sorted(buckets.items()):
        n_seg = sum(int(x.get("n_segments") or 0) for x in items)
        n_correct = sum(int(x.get("n_correct") or 0) for x in items)
        acc = (n_correct / n_seg) if n_seg else float("nan")
        mae_num = 0.0
        mae_den = 0
        for x in items:
            m = x.get("mae_sec")
            ns = int(x.get("n_segments") or 0)
            if m is not None and m == m and ns > 0:
                mae_num += float(m) * ns
                mae_den += ns
        out.append(
            {
                "dataset": ds,
                "n_talks": len(items),
                "n_segments": n_seg,
                "accuracy": acc,
                "mae_sec": (mae_num / mae_den) if mae_den else float("nan"),
            }
        )
    return out


def aggregate_chapter_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        buckets.setdefault(ds, []).append(r)
    out: list[dict[str, Any]] = []
    for ds, items in sorted(buckets.items()):
        out.append(
            {
                "dataset": ds,
                "n_talks": len(items),
                "n_ref": sum(int(x.get("n_ref") or 0) for x in items),
                "n_pred": sum(int(x.get("n_pred") or 0) for x in items),
                "precision": mean_metric(items, "precision"),
                "recall": mean_metric(items, "recall"),
                "f1": mean_metric(items, "f1"),
                "mae": mean_metric(items, "mae"),
            }
        )
    return out


def aggregate_ocr_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        buckets.setdefault(ds, []).append(r)
    return [
        {
            "dataset": ds,
            "n_keyframes": len(items),
            "cer": mean_metric(items, "cer"),
            "word_accuracy": mean_metric(items, "word_accuracy"),
        }
        for ds, items in sorted(buckets.items())
    ]


def aggregate_caption_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        buckets.setdefault(ds, []).append(r)
    out: list[dict[str, Any]] = []
    for ds, items in sorted(buckets.items()):
        n = len(items)
        hall = sum(1 for x in items if x.get("hallucinated")) / n if n else float("nan")
        ok = sum(1 for x in items if x.get("content_ok")) / n if n else float("nan")
        out.append(
            {
                "dataset": ds,
                "n_keyframes": n,
                "human_score": mean_metric(items, "human_score"),
                "content_ok_rate": ok,
                "hallucination_rate": hall,
            }
        )
    return out


def aggregate_summary_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TED")
        buckets.setdefault(ds, []).append(r)
    return [
        {
            "dataset": ds,
            "n_talks": len(items),
            "input": items[0].get("input", "Transcript + OCR + caption") if items else "",
            "rouge_l": mean_metric(items, "rouge_l"),
            "bertscore_f1": mean_metric(items, "bertscore_f1"),
            "factuality": mean_metric(items, "factuality"),
            "coverage": mean_metric(items, "coverage"),
        }
        for ds, items in sorted(buckets.items())
    ]


def aggregate_scene_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TVSum")
        buckets.setdefault(ds, []).append(r)
    return [
        {
            "dataset": ds,
            "n_videos": len(items),
            "n_ref": sum(int(x.get("n_ref") or 0) for x in items),
            "n_pred": sum(int(x.get("n_pred") or 0) for x in items),
            "precision": mean_metric(items, "precision"),
            "recall": mean_metric(items, "recall"),
            "f1": mean_metric(items, "f1"),
        }
        for ds, items in sorted(buckets.items())
    ]


def aggregate_keyframe_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset") or "TVSum")
        buckets.setdefault(ds, []).append(r)
    return [
        {
            "dataset": ds,
            "n_videos": len(items),
            "n_before": sum(int(x.get("n_before") or 0) for x in items),
            "n_after": sum(int(x.get("n_after") or 0) for x in items),
            "precision": mean_metric(items, "precision"),
            "recall": mean_metric(items, "recall"),
            "f1": mean_metric(items, "f1"),
            "compression_ratio": mean_metric(items, "compression_ratio"),
        }
        for ds, items in sorted(buckets.items())
    ]


def build_dataset_aggregates(results: dict[str, Any]) -> dict[str, Any]:
    """Gộp per-talk/per-video rows thành một dòng mỗi dataset (cho báo cáo luận văn)."""
    return {
        "asr": aggregate_asr_by_model(results.get("asr") or []),
        "scene": aggregate_scene_by_dataset(results.get("scene") or []),
        "keyframe": aggregate_keyframe_by_dataset(results.get("keyframe") or []),
        "ocr": aggregate_ocr_by_dataset(results.get("ocr") or []),
        "caption": aggregate_caption_by_dataset(results.get("caption") or []),
        "timeline": aggregate_timeline_by_dataset(results.get("timeline") or []),
        "chapter": aggregate_chapter_by_dataset(results.get("chapter") or []),
        "summary": aggregate_summary_by_dataset(results.get("summary") or []),
    }
