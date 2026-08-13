"""Verify sprint presets: invariants, coverage, chapter gaps, KF audit."""

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
)
from experiments.scripts.offline_replay import offline_visual_gate, pick_run
from experiments.scripts.offline_sprint_ablation import PRESETS
from experiments.scripts.run_gated_compare import compute_metrics


def _chapter_bounds(ch: dict) -> tuple[float, float]:
    start = float(ch.get("startTime", ch.get("start_time", 0.0)) or 0.0)
    end = float(ch.get("endTime", ch.get("end_time", start)) or start)
    return start, end


def verify_run(run: dict, preset_name: str) -> dict:
    chapters = run.get("chapters", [])
    keyframes = run.get("keyframes", [])
    checks: list[dict] = []
    ok = True

    def add(name: str, passed: bool, detail: str):
        nonlocal ok
        if not passed:
            ok = False
        checks.append({"check": name, "passed": passed, "detail": detail})

    # Chapter invariants
    durs = []
    for i, ch in enumerate(chapters):
        s, e = _chapter_bounds(ch)
        dur = e - s
        durs.append(dur)
        if i > 0 and s < _chapter_bounds(chapters[i - 1])[1] - 0.01:
            add("chapter_monotonic", False, f"Chapter {i} starts before previous ends ({s})")
    if durs:
        add("min_chapter_45s", min(durs) >= 45.0 or preset_name == "baseline", f"min={min(durs):.2f}s")
        add("chapter_titles", all(ch.get("title") for ch in chapters), "all chapters have titles")

    # Keyframe invariants
    timestamps = [float(k.get("timestamp") or 0) for k in keyframes]
    add("kf_no_duplicate_ts", len(timestamps) == len(set(round(t, 3) for t in timestamps)), f"count={len(keyframes)}")
    add("kf_sorted", timestamps == sorted(timestamps), "timestamps monotonic")

    with_transcript = sum(1 for k in keyframes if (k.get("transcript") or "").strip())
    cov = with_transcript / max(1, len(keyframes))
    add("kf_coverage_100", cov == 1.0 or preset_name in ("baseline", "s1", "s1_s3", "s1_s2c_s3"), f"coverage={cov:.2%} ({with_transcript}/{len(keyframes)})")

    # Chapter-keyframe alignment
    gaps = []
    for i, ch in enumerate(chapters):
        s, e = _chapter_bounds(ch)
        is_last = i == len(chapters) - 1
        in_ch = []
        for k in keyframes:
            ts = float(k.get("timestamp") or 0)
            if is_last:
                if s <= ts <= e:
                    in_ch.append(k)
            elif s <= ts < e:
                in_ch.append(k)
        if not in_ch:
            gaps.append(ch.get("title", f"Chapter {i}"))
    add("chapter_kf_gaps", len(gaps) == 0, f"gaps={gaps}")

    # Evidence scores for sprint4+ presets
    stack = run.get("sprint_stack", [])
    if any(s.startswith("sprint4") for s in stack):
        scores = [keyframe_evidence_score(k) for k in keyframes]
        add("sprint4_evidence_positive", all(s > 0 for s in scores), f"min_score={min(scores) if scores else 0}")

    # Sprint 10 export metadata
    if preset_name == "sprint10" or "sprint10" in stack:
        export_meta = (run.get("sprint_stats") or {}).get("sprint10") or (run.get("sprint_stats") or {}).get("export_meta") or {}
        add("sprint10_export_ready", bool(export_meta.get("export_ready")), f"export_ready={export_meta.get('export_ready')}")
        q = export_meta.get("pipeline_quality_score")
        add("sprint10_quality_score", q is not None and float(q) >= 85.0, f"score={q}")
        add("sprint10_no_generic_captions", export_meta.get("generic_captions_remaining", 99) == 0, f"generic_left={export_meta.get('generic_captions_remaining')}")
        hints = export_meta.get("chapters_with_visual_hints", 0)
        add("sprint10_visual_hints", hints >= len(chapters), f"hints={hints}/{len(chapters)}")
        add("sprint10_transcript_coverage", float(export_meta.get("transcript_coverage") or 0) >= 0.85, f"coverage={export_meta.get('transcript_coverage')}")

    metrics = compute_metrics(run)
    return {
        "preset": preset_name,
        "all_passed": ok,
        "checks": checks,
        "metrics": metrics,
        "gaps": gaps,
        "keyframes_audit": [
            {
                "t": k.get("timestamp"),
                "evidence": keyframe_evidence_score(k),
                "has_transcript": bool((k.get("transcript") or "").strip()),
                "desc": (k.get("description") or "")[:60],
                "status": k.get("sprint4_status") or k.get("visual_gate_status") or "kept",
            }
            for k in keyframes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/ted_compare_20260806_162755/compare_report.json")
    parser.add_argument("--presets", default="sprint10,baseline")
    args = parser.parse_args()

    baseline_run = pick_run(
        json.loads((project_root / args.baseline).read_text(encoding="utf-8")),
        "backend_parity", "backend_baseline",
    )
    if not baseline_run:
        raise SystemExit("No baseline.")

    preset_names = [p.strip() for p in args.presets.split(",")]
    reports = []

    for name in preset_names:
        if name == "baseline":
            run = dict(baseline_run)
            run["sprint_stack"] = []
        else:
            preset = PRESETS.get(name)
            if not preset:
                # build recommended from constants
                if name == "recommended_s6":
                    preset = {"stack": RECOMMENDED_STACK, "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"]}
                elif name == "sprint10":
                    preset = {"stack": FULL_STACK_S10, "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"]}
                else:
                    print(f"SKIP unknown preset: {name}")
                    continue
            ctx = SprintContext(
                chapters=[dict(c) for c in baseline_run.get("chapters", [])],
                keyframes=[dict(k) for k in baseline_run.get("keyframes", [])],
            )
            ctx = apply_sprint_stack(
                ctx,
                preset.get("stack", []),
                visual_gate_fn=offline_visual_gate,
                project_root=project_root,
                visual_cfg=preset.get("visual_cfg"),
                sprint4_cfg=preset.get("sprint4_cfg"),
            )
            run = dict(baseline_run)
            run["chapters"] = ctx.chapters
            run["keyframes"] = ctx.keyframes
            run["sprint_stack"] = preset.get("stack", [])
            run["sprint_stats"] = ctx.stats

        reports.append(verify_run(run, name))

    out_dir = project_root / "outputs" / f"sprint_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "verify_report.json"
    out_md = out_dir / "verify_report.md"

    payload = {"created_at": datetime.now().isoformat(), "reports": reports}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Sprint Verification Report", ""]
    for r in reports:
        status = "PASS" if r["all_passed"] else "FAIL"
        m = r["metrics"]
        md += [
            f"## `{r['preset']}` — **{status}**",
            f"- Chapters: {m['chapter_count']} | Min ch: {m['min_chapter_duration_sec']}s | KF: {m['keyframe_count']} | Coverage: {m['keyframe_script_coverage']}",
            "",
            "| Check | Pass | Detail |",
            "|---|---|---|",
        ]
        for c in r["checks"]:
            md.append(f"| {c['check']} | {'✓' if c['passed'] else '✗'} | {c['detail']} |")
        if r.get("gaps"):
            md.append(f"\nChapter gaps: {r['gaps']}")
        md += ["", "### Keyframes kept", ""]
        for k in r.get("keyframes_audit", []):
            md.append(f"- t={k['t']}: ev={k['evidence']} tx={k['has_transcript']} | {k['desc']}")
        md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")

    passed = sum(1 for r in reports if r["all_passed"])
    print(f"OUT_DIR={out_dir}")
    print(f"PASSED={passed}/{len(reports)}")
    for r in reports:
        print(f"  {r['preset']}: {'PASS' if r['all_passed'] else 'FAIL'} kf={r['metrics']['keyframe_count']} cov={r['metrics']['keyframe_script_coverage']}")


if __name__ == "__main__":
    main()
