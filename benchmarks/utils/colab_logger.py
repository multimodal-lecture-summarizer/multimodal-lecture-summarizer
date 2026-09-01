"""
Colab Logger — structured step logging for long-running notebooks (Approach A, D-T15 real-data-only)
Usage in notebooks:
    from benchmarks.utils.colab_logger import StepLogger
    logger = StepLogger("02-phase2")
    logger.step(1, "Load manifest", total=6)
    # ... work ...
    logger.done("Load manifest", extra={"n": len(items)})
    logger.summary()
Works on Colab Free: prints with timestamp, flush, and optional elapsed per step.
No external deps; uses only stdlib. Keeps YAGNI/KISS.
"""
import time
import sys
from pathlib import Path

class StepLogger:
    def __init__(self, name: str, log_file: Path | None = None):
        self.name = name
        self.t0 = time.time()
        self.steps = []
        self.log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(self, msg: str) -> str:
        elapsed = time.time() - self.t0
        ts = time.strftime("%H:%M:%S")
        return f"[{ts} +{elapsed:6.1f}s][{self.name}] {msg}"

    def _write(self, line: str):
        print(line, flush=True)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def step(self, idx: int, title: str, total: int | None = None, extra: str = ""):
        tag = f"Step {idx}/{total}" if total else f"Step {idx}"
        self._write(self._fmt(f"▶ {tag}: {title} {extra}".strip()))
        self.steps.append((idx, title, time.time()))

    def done(self, title: str, extra: dict | None = None):
        # find last step with same title
        now = time.time()
        # compute elapsed for last matching step if any
        elapsed = ""
        for _, t, st in reversed(self.steps):
            if t == title:
                elapsed = f"({now - st:.1f}s)"
                break
        extra_s = ""
        if extra:
            extra_s = " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        self._write(self._fmt(f"✓ {title} done {elapsed}{extra_s}".strip()))

    def warn(self, msg: str):
        self._write(self._fmt(f"⚠ {msg}"))

    def error(self, msg: str):
        self._write(self._fmt(f"✗ {msg}"), )

    def summary(self):
        total = time.time() - self.t0
        self._write(self._fmt(f"— Summary: {len(self.steps)} steps, total {total/60:.1f} min —"))

# Convenience: tqdm fallback if not installed
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        # minimal fallback: just iterate, print every 10%
        total = kwargs.get("total") or (len(iterable) if hasattr(iterable, "__len__") else None)
        if total:
            step = max(1, total // 10)
            for i, x in enumerate(iterable):
                if i % step == 0:
                    print(f"[tqdm] {i}/{total} ({i/total*100:.0f}%)", flush=True)
                yield x
        else:
            yield from iterable
