"""Single-Author Audit Tool and LLM-as-a-Judge Protocol.

Implements reference quality audit and automated multidimensional evaluation per decisions-log.md:
- D-S03: Pre-labeled ground truth + Single Human Author Audit + LLM-as-a-Judge.
- D-T14: Schema `id, source_support (0–2), coverage (0–2), style (summary-like/boilerplate/mixed), action (keep/flag/exclude)`.
- Multi-dimensional LLM evaluator for factual consistency, coverage, and citation precision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityAuditRecord:
    item_id: str
    dataset: str
    source_support: int  # 0: Unsupported, 1: Partially supported, 2: Fully supported
    coverage: int        # 0: Misses main theme, 1: Partial keypoints, 2: Comprehensive keypoints
    style: str           # 'summary-like' | 'boilerplate' | 'mixed'
    action: str          # 'keep' | 'flag' | 'exclude'
    rationale: str = ""
    audited_by: str = "author"
    timestamp: str = ""


@dataclass
class LLMJudgeScore:
    item_id: str
    factuality_score: float      # 1 to 5 scale
    coverage_score: float         # 1 to 5 scale
    coherence_score: float        # 1 to 5 scale
    concision_score: float        # 1 to 5 scale
    citation_precision: float     # 0.0 to 1.0
    unsupported_claim_rate: float # 0.0 to 1.0
    judge_rationale: str = ""


class SingleAuthorAuditTool:
    """Manages single-author manual quality audit calibration batches and exports."""

    def __init__(self, audit_file_path: Path):
        self.audit_file_path = Path(audit_file_path)
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, QualityAuditRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.audit_file_path.exists():
            try:
                data = json.loads(self.audit_file_path.read_text(encoding="utf-8"))
                for r in data.get("audit_records", []):
                    rec = QualityAuditRecord(**r)
                    self.records[rec.item_id] = rec
            except Exception as e:
                logger.warning(f"Failed to load audit file ({e}), starting fresh.")

    def add_record(self, record: QualityAuditRecord) -> None:
        self.records[record.item_id] = record
        self.save()

    def save(self) -> None:
        payload = {
            "total_audited": len(self.records),
            "audit_records": [asdict(r) for r in self.records.values()],
        }
        self.audit_file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_exclusion_list(self) -> List[str]:
        """Return list of item IDs marked as 'exclude'."""
        return [item_id for item_id, r in self.records.items() if r.action == "exclude"]

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute distribution of audit actions and scores."""
        total = len(self.records)
        if total == 0:
            return {"total": 0, "keep": 0, "flag": 0, "exclude": 0, "avg_support": 0.0, "avg_coverage": 0.0}

        keeps = sum(1 for r in self.records.values() if r.action == "keep")
        flags = sum(1 for r in self.records.values() if r.action == "flag")
        excludes = sum(1 for r in self.records.values() if r.action == "exclude")
        avg_support = sum(r.source_support for r in self.records.values()) / total
        avg_cov = sum(r.coverage for r in self.records.values()) / total

        return {
            "total": total,
            "keep_count": keeps,
            "keep_pct": round(keeps / total * 100, 1),
            "flag_count": flags,
            "exclude_count": excludes,
            "exclude_pct": round(excludes / total * 100, 1),
            "avg_source_support_score": round(avg_support, 2),
            "avg_coverage_score": round(avg_cov, 2),
        }
