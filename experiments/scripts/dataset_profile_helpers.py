"""Pure helpers for deterministic local dataset profiling."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from experiments.scripts.benchmark_artifacts import RunnerContext, sha256_file


def split_for(path: str) -> str:
    tokens = {
        token
        for part in Path(path).parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }
    if "train" in tokens:
        return "train"
    if "test" in tokens:
        return "test"
    if "validation" in tokens or "val" in tokens or "dev" in tokens:
        return "validation"
    return "unknown"


def numeric_summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(value: float) -> int:
        return ordered[min(len(ordered) - 1, math.floor((len(ordered) - 1) * value))]

    return {
        "n": len(values),
        "missing_n": 0,
        "min": min(values),
        "p5": percentile(0.05),
        "p25": percentile(0.25),
        "median": median(values),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": max(values),
        "mean": mean(values),
        "standard_deviation": pstdev(values),
    }


def write_size_figure(
    context: RunnerContext, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    figure_dir = context.namespace_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    width, height = 720, 80 + 48 * len(rows)
    maximum = max(row["selected_bytes"] for row in rows) or 1
    bars = []
    for index, row in enumerate(rows):
        y = 50 + index * 48
        bar_width = max(1, int(500 * row["selected_bytes"] / maximum))
        dataset_id = html.escape(str(row["dataset_id"]))
        bars.append(
            f'<text x="10" y="{y + 16}" font-size="13">{dataset_id}</text>'
            f'<rect x="180" y="{y}" width="{bar_width}" height="22" fill="#315b7d" />'
            f'<text x="{190 + bar_width}" y="{y + 16}" font-size="12">{row["selected_bytes"]}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="24" font-size="16">Selected bytes by local dataset</text>'
        + "".join(bars)
        + "</svg>"
    )
    path = figure_dir / "selected-bytes-by-dataset.svg"
    path.write_text(svg, encoding="utf-8")
    source_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "figure_id": "selected_bytes_by_dataset",
        "path": path.relative_to(context.artifact_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "source_table_sha256": source_hash,
        "profile": context.profile,
        "unit": "bytes",
        "n": len(rows),
    }
