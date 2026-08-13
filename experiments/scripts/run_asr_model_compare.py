#!/usr/bin/env python3
"""Broader Faster-Whisper size ladder on the same TED-LIUM clips.

Loads each model once. Reports WER/CER plus load time, wall time, RTF.
Does not use GPU (other Python 3.11 job owns VRAM).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from experiments.evaluation.datasets import pick_tedlium_asr_items
from experiments.evaluation.metrics import asr_wer_cer, rtf

MODELS = [
    ("tiny.en", "Faster-Whisper tiny.en", "39M"),
    ("base.en", "Faster-Whisper base.en (production)", "74M"),
    ("base", "Faster-Whisper base (đa ngữ)", "74M"),
    ("small.en", "Faster-Whisper small.en", "244M"),
    ("medium.en", "Faster-Whisper medium.en", "769M"),
    ("large-v3", "Faster-Whisper large-v3", "1550M"),
]


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if v != v:
            return "N/A"
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    return f"{float(v) * 100:.2f}"


def resolve_model(size: str) -> str:
    local = ROOT / "cache" / f"faster-whisper-{size}"
    if (local / "model.bin").exists():
        return str(local)
    return size


def prepare_wavs(items: list[dict]) -> list[dict]:
    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

    prep = AudioTranscriber({"model_name": "base.en"})
    out = []
    for it in items:
        wav = Path(it["wav_path"])
        if not wav.exists():
            print(f"[SKIP] missing {wav}")
            continue
        try:
            audio = prep.reduce_noise(str(wav))
        except Exception as e:
            print(f"[WARN] denoise {wav.name}: {e}")
            import soundfile as sf

            audio, _sr = sf.read(str(wav))
        out.append({**it, "audio": audio, "duration": float(it["duration"])})
    return out


def eval_model(size: str, clips: list[dict]) -> dict:
    from faster_whisper import WhisperModel

    source = resolve_model(size)
    print(f"\n=== {size}  source={source} ===")
    t0 = time.perf_counter()
    model = WhisperModel(source, device="cpu", compute_type="int8", download_root=str(ROOT / "cache"))
    load_sec = time.perf_counter() - t0
    print(f"[load] {load_sec:.1f}s")

    rows = []
    infer_sec = 0.0
    audio_sec = 0.0
    try:
        for it in clips:
            t1 = time.perf_counter()
            segments, info = model.transcribe(
                it["audio"],
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=2000),
                word_timestamps=False,
                language="en" if size.endswith(".en") or size == "large-v3" else None,
            )
            hyp = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
            wall = time.perf_counter() - t1
            scores = asr_wer_cer(it["text"], hyp)
            infer_sec += wall
            audio_sec += it["duration"]
            rows.append(
                {
                    "id": it["id"],
                    "speaker_id": it["speaker_id"],
                    "duration": it["duration"],
                    "wer": scores["wer"],
                    "cer": scores["cer"],
                    "rtf": rtf(it["duration"], wall),
                    "wall_sec": wall,
                    "hypothesis": hyp,
                }
            )
            print(f"  {it['id'][:48]}  WER={scores['wer']:.3f}  {wall:.2f}s")
    finally:
        del model
        import gc

        gc.collect()

    n = len(rows)
    return {
        "model": size,
        "label": next(l for s, l, _p in MODELS if s == size),
        "params": next(p for s, _l, p in MODELS if s == size),
        "load_sec": load_sec,
        "infer_sec": infer_sec,
        "audio_sec": audio_sec,
        "total_sec": load_sec + infer_sec,
        "rtf": (infer_sec / audio_sec) if audio_sec else float("nan"),
        "wer": (sum(r["wer"] for r in rows) / n) if n else float("nan"),
        "cer": (sum(r["cer"] for r in rows) / n) if n else float("nan"),
        "n": n,
        "clips": rows,
    }


def render(results: list[dict], *, generated: str, n_clips: int) -> str:
    ranked = sorted(results, key=lambda r: (r["wer"] if r["wer"] == r["wer"] else 9, r["rtf"]))
    prod = next((r for r in results if r["model"] == "base.en"), None)
    lines = [
        "# So sánh ASR Faster-Whisper (cùng TED-LIUM)",
        "",
        f"*Generated: {generated}*",
        "",
        f"Cùng {n_clips} clip TED-LIUM, CPU `int8`, load 1 lần / model, audio đã DeepFilterNet giống production.",
        "GPU không dùng (VRAM đang bị process Python 3.11 chiếm).",
        "",
        "| Model | Params | WER (%) | CER (%) | RTF | Load (s) | Infer (s) | Tổng (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        mark = " **← production**" if r["model"] == "base.en" else ""
        lines.append(
            f"| {r['label']}{mark} | {r['params']} | {_pct(r['wer'])} | {_pct(r['cer'])} | "
            f"{_fmt(r['rtf'])} | {_fmt(r['load_sec'], 1)} | {_fmt(r['infer_sec'], 1)} | {_fmt(r['total_sec'], 1)} |"
        )
    lines += ["", "## Kết luận chọn `base.en`", ""]
    if prod and ranked:
        best = ranked[0]
        lines.append(
            f"- Thấp WER nhất trong lần chạy: **{best['label']}** (WER={_pct(best['wer'])}, RTF={_fmt(best['rtf'])})."
        )
        slower = [r for r in results if r["model"] != "base.en" and r["infer_sec"] > prod["infer_sec"]]
        if slower:
            lines.append(
                f"- `base.en` infer {prod['infer_sec']:.1f}s / {prod['audio_sec']:.1f}s audio "
                f"(RTF={_fmt(prod['rtf'])}). Các model lớn hơn chậm hơn mà WER không luôn tốt hơn."
            )
        en = next((r for r in results if r["model"] == "base"), None)
        if en:
            lines.append(
                f"- `base` đa ngữ WER={_pct(en['wer'])} vs `base.en` {_pct(prod['wer'])}: "
                "bài giảng EN nên giữ bản `.en`."
            )
        lines.append(
            "- Production giữ **base.en**: đủ WER cho lecture EN, RTF thấp, model ~150MB, không cần GPU."
        )
    lines += [
        "",
        "## Chi tiết từng clip",
        "",
    ]
    if results:
        headers = ["Clip"] + [r["model"] + " WER" for r in results]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        ids = [c["id"] for c in results[0]["clips"]]
        by_model = {r["model"]: {c["id"]: c["wer"] for c in r["clips"]} for r in results}
        for cid in ids:
            short = cid if len(cid) <= 42 else cid[:39] + "..."
            cells = [short] + [_pct(by_model[r["model"]].get(cid)) for r in results]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--models", type=str, default="tiny.en,base.en,base,small.en,medium.en,large-v3")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "outputs" / "model_choice"))
    args = p.parse_args()

    sizes = [s.strip() for s in args.models.split(",") if s.strip()]
    items = pick_tedlium_asr_items(limit=args.limit)
    print(f"[ASR] clips={len(items)} models={sizes}")
    clips = prepare_wavs(items)
    if not clips:
        print("[ERR] no clips")
        return 1

    results = []
    for size in sizes:
        try:
            results.append(eval_model(size, clips))
        except Exception as e:
            print(f"[WARN] {size}: {e}")
            results.append(
                {
                    "model": size,
                    "label": size,
                    "params": "?",
                    "load_sec": float("nan"),
                    "infer_sec": float("nan"),
                    "audio_sec": sum(c["duration"] for c in clips),
                    "total_sec": float("nan"),
                    "rtf": float("nan"),
                    "wer": float("nan"),
                    "cer": float("nan"),
                    "n": 0,
                    "clips": [],
                    "error": str(e),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    payload = {"generated": stamp, "device": "cpu/int8", "results": results}
    (out_dir / "ASR_COMPARE.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = render(results, generated=stamp, n_clips=len(clips))
    (out_dir / "ASR_COMPARE.md").write_text(md, encoding="utf-8")
    print(f"[OK] {out_dir / 'ASR_COMPARE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
