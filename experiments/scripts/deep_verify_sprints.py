"""Deep sprint verification: overlap with gated, drop audit, chapter alignment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.sprints import (
    FULL_STACK_S10,
    RECOMMENDED_STACK,
    RECOMMENDED_STACK_CONFIG,
    SprintContext,
    apply_sprint_stack,
    keyframe_evidence_score,
    sprint4_prune_dense_keyframes,
)
from experiments.scripts.offline_replay import pick_run
from experiments.scripts.offline_sprint_ablation import PRESETS
from experiments.scripts.run_gated_compare import compute_metrics


def _ts(kf: dict) -> float:
    return float(kf.get("timestamp") or 0.0)


def _chapter_bounds(ch: dict) -> tuple[float, float]:
    s = float(ch.get("startTime", ch.get("start_time", 0.0)) or 0.0)
    e = float(ch.get("endTime", ch.get("end_time", s)) or s)
    return s, e


def build_stack_run(baseline_run: dict, stack: list[str]) -> dict:
    ctx = SprintContext(
        chapters=[dict(c) for c in baseline_run.get("chapters", [])],
        keyframes=[dict(k) for k in baseline_run.get("keyframes", [])],
    )
    ctx = apply_sprint_stack(
        ctx,
        stack,
        sprint4_cfg=RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
    )
    run = dict(baseline_run)
    run["chapters"] = ctx.chapters
    run["keyframes"] = ctx.keyframes
    run["sprint_stack"] = stack
    run["sprint_stats"] = ctx.stats
    return run


def build_recommended(baseline_run: dict) -> dict:
    return build_stack_run(baseline_run, RECOMMENDED_STACK)


def build_sprint10(baseline_run: dict) -> dict:
    return build_stack_run(baseline_run, FULL_STACK_S10)


def jaccard_timestamps(a: list, b: list, tol: float = 2.0) -> dict:
    """Match keyframes if timestamps within tol seconds."""
    matched = 0
    pairs = []
    used_b = set()
    for ka in a:
        ta = _ts(ka)
        best = None
        best_gap = tol + 1
        for j, kb in enumerate(b):
            if j in used_b:
                continue
            gap = abs(ta - _ts(kb))
            if gap <= tol and gap < best_gap:
                best_gap = gap
                best = j
        if best is not None:
            matched += 1
            used_b.add(best)
            pairs.append((_ts(ka), _ts(b[best]), round(best_gap, 2)))
    union = len(a) + len(b) - matched
    return {
        "matched": matched,
        "jaccard": round(matched / union, 4) if union else 0.0,
        "only_a": len(a) - matched,
        "only_b": len(b) - matched,
        "pairs_sample": pairs[:8],
    }


def audit_sprint4_drops(baseline_kfs: list, sprint4_cfg: dict) -> list[dict]:
    """Explain each dropped keyframe vs kept in same temporal window."""
    cfg = sprint4_cfg or RECOMMENDED_STACK_CONFIG["sprint4_cfg"]
    kept, stats = sprint4_prune_dense_keyframes([dict(k) for k in baseline_kfs], **cfg)
    kept_ts = {_ts(k) for k in kept}
    window_sec = cfg.get("window_sec", 45.0)
    max_per = cfg.get("max_per_window", 2)

    sorted_kfs = sorted(baseline_kfs, key=lambda k: _ts(k))
    for k in sorted_kfs:
        k["evidence_score"] = keyframe_evidence_score(k)

    audits = []
    for kf in sorted_kfs:
        if _ts(kf) in kept_ts:
            continue
        ts = _ts(kf)
        window = [
            x for x in sorted_kfs
            if ts - window_sec <= _ts(x) <= ts + window_sec
        ]
        window_ranked = sorted(window, key=lambda x: x.get("evidence_score", 0), reverse=True)
        rank = next((i + 1 for i, x in enumerate(window_ranked) if _ts(x) == ts), -1)
        top_kept = [x for x in window_ranked if _ts(x) in kept_ts][:max_per]
        audits.append({
            "timestamp": ts,
            "evidence": kf.get("evidence_score"),
            "has_transcript": bool((kf.get("transcript") or "").strip()),
            "description": (kf.get("description") or "")[:70],
            "window_size": len(window),
            "rank_in_window": rank,
            "kept_in_window": [
                {"t": _ts(x), "evidence": x.get("evidence_score"), "desc": (x.get("description") or "")[:50]}
                for x in top_kept
            ],
            "drop_reason": (
                "no_transcript_low_evidence" if not (kf.get("transcript") or "").strip()
                else f"window_cap_rank_{rank}_gt_{max_per}"
            ),
        })
    return audits


def chapter_alignment_report(chapters: list, keyframes: list) -> list[dict]:
    rows = []
    n = len(chapters)
    for i, ch in enumerate(chapters):
        s, e = _chapter_bounds(ch)
        is_last = i == n - 1
        in_ch = []
        for k in keyframes:
            ts = _ts(k)
            if is_last:
                if s <= ts <= e:
                    in_ch.append(k)
            elif s <= ts < e:
                in_ch.append(k)
        with_tx = sum(1 for k in in_ch if (k.get("transcript") or "").strip())
        rows.append({
            "chapter": i + 1,
            "title": (ch.get("title") or "")[:50],
            "start": round(s, 1),
            "end": round(e, 1),
            "duration": round(e - s, 1),
            "keyframes": len(in_ch),
            "with_transcript": with_tx,
            "coverage": round(with_tx / max(1, len(in_ch)), 2),
            "boundary_note": (
                "KF at chapter end assigned to this chapter (half-open interval)"
                if in_ch and abs(_ts(in_ch[-1]) - e) < 0.1 and not is_last
                else None
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/ted_compare_20260806_162755/compare_report.json")
    parser.add_argument("--gated", default="outputs/gated_compare_20260807_143515/compare_report.json")
    parser.add_argument("--variant", default="sprint10", choices=("recommended", "sprint10"))
    args = parser.parse_args()

    baseline_payload = json.loads((project_root / args.baseline).read_text(encoding="utf-8"))
    baseline_run = pick_run(baseline_payload, "backend_parity", "backend_baseline")
    exp_run = pick_run(baseline_payload, "experimental_video")
    gated_run = None
    gated_path = project_root / args.gated
    if gated_path.is_file():
        gated_run = pick_run(json.loads(gated_path.read_text(encoding="utf-8")), "experimental_gated")

    recommended = build_sprint10(baseline_run) if args.variant == "sprint10" else build_recommended(baseline_run)
    variant_label = "Sprint 10" if args.variant == "sprint10" else "Recommended S6"
    rec_kfs = recommended["keyframes"]
    base_kfs = baseline_run.get("keyframes", [])

    export_meta = (recommended.get("sprint_stats") or {}).get("sprint10") or {}

    report = {
        "created_at": datetime.now().isoformat(),
        "variant": args.variant,
        "variant_label": variant_label,
        "variant_stack": recommended.get("sprint_stack", []),
        "recommended_metrics": compute_metrics(recommended),
        "baseline_metrics": compute_metrics(baseline_run),
        "sprint10_export_meta": export_meta,
        "sprint4_drop_audit": audit_sprint4_drops(base_kfs, RECOMMENDED_STACK_CONFIG["sprint4_cfg"]),
        "chapter_alignment": chapter_alignment_report(recommended["chapters"], rec_kfs),
    }

    if gated_run:
        report["gated_metrics"] = compute_metrics(gated_run)
        report["overlap_recommended_vs_gated"] = jaccard_timestamps(rec_kfs, gated_run.get("keyframes", []))
        report["overlap_baseline_vs_gated"] = jaccard_timestamps(base_kfs, gated_run.get("keyframes", []))

    if exp_run:
        report["experimental_old_metrics"] = compute_metrics(exp_run)
        report["overlap_recommended_vs_exp_old"] = jaccard_timestamps(rec_kfs, exp_run.get("keyframes", []))

    # Sanity: no duplicate timestamps in recommended
    ts_list = [_ts(k) for k in rec_kfs]
    report["duplicate_timestamps"] = len(ts_list) - len(set(ts_list))

    # All kept KF should have transcript after sprint4 on TED
    no_tx = [k for k in rec_kfs if not (k.get("transcript") or "").strip()]
    report["keyframes_without_transcript_after_s4"] = len(no_tx)

    out_dir = project_root / "outputs" / f"sprint_deep_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "deep_verify.json"
    out_md = out_dir / "deep_verify.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rm = report["recommended_metrics"]
    bm = report["baseline_metrics"]
    lines = [
        f"# Deep Sprint Verification — {variant_label}",
        "",
        "## Metrics",
        "",
        "| Variant | Ch | Min ch | KF | Coverage | Wall |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline | {bm['chapter_count']} | {bm['min_chapter_duration_sec']} | {bm['keyframe_count']} | {bm['keyframe_script_coverage']} | {bm['elapsed_wall_sec']} |",
        f"| {variant_label} | {rm['chapter_count']} | {rm['min_chapter_duration_sec']} | {rm['keyframe_count']} | {rm['keyframe_script_coverage']} | {rm['elapsed_wall_sec']} |",
    ]
    if gated_run:
        gm = report["gated_metrics"]
        lines.append(
            f"| Gated full | {gm['chapter_count']} | {gm['min_chapter_duration_sec']} | "
            f"{gm['keyframe_count']} | {gm['keyframe_script_coverage']} | {gm['elapsed_wall_sec']} |"
        )

    ov = report.get("overlap_recommended_vs_gated", {})
    lines += [
        "",
        "## Keyframe overlap (±2s)",
        f"- {variant_label} vs Gated: Jaccard **{ov.get('jaccard')}**, matched {ov.get('matched')}, only variant {ov.get('only_a')}, only gated {ov.get('only_b')}",
    ]
    if report.get("overlap_baseline_vs_gated"):
        ob = report["overlap_baseline_vs_gated"]
        lines.append(f"- Baseline vs Gated: Jaccard **{ob.get('jaccard')}**, matched {ob.get('matched')}")

    lines += ["", "## Sprint-4 drop audit", ""]
    for d in report["sprint4_drop_audit"]:
        lines.append(
            f"- t={d['timestamp']}s ev={d['evidence']} tx={d['has_transcript']} | "
            f"{d['drop_reason']} | {d['description']}"
        )
        for k in d.get("kept_in_window", []):
            lines.append(f"  - kept t={k['t']}s ev={k['evidence']} | {k['desc']}")

    lines += ["", "## Chapter ↔ Keyframe alignment", ""]
    lines.append("| Ch | Title | Dur(s) | KF | w/ transcript |")
    lines.append("|---|---|---:|---:|---:|")
    for row in report["chapter_alignment"]:
        lines.append(
            f"| {row['chapter']} | {row['title']} | {row['duration']} | {row['keyframes']} | {row['with_transcript']} |"
        )

    if export_meta:
        lines += [
            "",
            "## Sprint 10 export meta",
            f"- **pipeline_quality_score**: {export_meta.get('pipeline_quality_score')}",
            f"- **export_ready**: {export_meta.get('export_ready')}",
            f"- **generic_captions_remaining**: {export_meta.get('generic_captions_remaining')}",
            f"- **captions_enriched**: {export_meta.get('captions_enriched')}",
            f"- **chapters_with_visual_hints**: {export_meta.get('chapters_with_visual_hints')}",
        ]

    lines += [
        "",
        "## Sanity checks",
        f"- Duplicate timestamps: **{report['duplicate_timestamps']}** (expect 0)",
        f"- KF without transcript after S4: **{report['keyframes_without_transcript_after_s4']}** (expect 0 on TED)",
    ]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")
    print(f"VARIANT={args.variant} kf={rm['keyframe_count']} cov={rm['keyframe_script_coverage']}")
    if export_meta:
        print(f"QUALITY={export_meta.get('pipeline_quality_score')} export_ready={export_meta.get('export_ready')}")
    if ov:
        print(f"JACCARD vs gated={ov.get('jaccard')} matched={ov.get('matched')}")
    print(f"DROPPED by S4={len(report['sprint4_drop_audit'])} no_tx_after={report['keyframes_without_transcript_after_s4']}")


if __name__ == "__main__":
    main()
