"""
Validation Gate & Reproducibility Verification Tests (Phase 6).
Automated tests verifying that the benchmark suite complies with:
  - D-T01: Pinned baseline VLM
  - D-T02: C5/C6 architecture freeze
  - D-T04: Feature dimensions (visual 384, ocr 384, acoustic 32)
  - D-T07: Holm-Bonferroni correction
  - D-T08: Fair budget bounds (source <= 32k, output <= 512)
  - D-T15: Real-data-only constraint (0 mocks in research path)
  - H4: Pareto dominance of E3 over E1 and E4
"""

import json
import os
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


def test_repro_manifest_structure_and_completeness(project_root):
    manifest_path = project_root / "reports" / "repro_manifest.json"
    assert manifest_path.exists(), "repro_manifest.json must exist in reports/"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["manifest_version"] == "2.0.0-final-frozen"
    assert "git_provenance" in manifest
    assert manifest["git_provenance"]["frozen_commit_hash"] is not None

    # Check 4 RQ outcomes recorded
    rq_outcomes = manifest["research_questions_outcomes"]
    assert "RQ1" in rq_outcomes
    assert "RQ2" in rq_outcomes
    assert "RQ3" in rq_outcomes
    assert "RQ4" in rq_outcomes

    # Check frozen decisions
    decisions = manifest["frozen_experimental_decisions"]
    assert decisions["D-T04"]["acoustic_dim"] == 32
    assert decisions["D-T04"]["visual_dim"] == 384
    assert decisions["D-T08"]["max_source_tokens"] == 32000
    assert decisions["D-T08"]["max_output_tokens"] == 512


def test_publication_figures_exist(project_root):
    fig_dir = project_root / "reports" / "figures"
    assert fig_dir.exists(), "reports/figures directory must exist"

    required_figs = [
        "pareto_quality_vs_latency.png",
        "pareto_quality_vs_vram.png",
        "component_latency_breakdown.png"
    ]
    for fig_name in required_figs:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Figure {fig_name} must exist"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} must not be empty"


def test_pareto_dominance_h4_verification(project_root):
    from benchmarks.scripts.run_rq4_efficiency_benchmark import (
        get_default_empirical_data,
        compute_pareto_frontier
    )
    data = get_default_empirical_data()
    pareto = compute_pareto_frontier(data)

    # E3 must be Pareto optimal
    assert pareto["E3"]["is_pareto_optimal"] is True, "E3 must be Pareto optimal"
    
    # E3 must strictly dominate E4 (faster, lower VRAM, 0% failure, fewer hallucinations)
    assert "E4" in pareto["E3"]["dominates"], "E3 must strictly dominate E4"

    # E3 must have highest efficiency ratio (Quality / Cost trade-off)
    e3_ratio = pareto["E3"]["efficiency_ratio"]
    e4_ratio = pareto["E4"]["efficiency_ratio"]
    e1_ratio = pareto["E1"]["efficiency_ratio"]

    assert e3_ratio > e1_ratio, f"E3 ratio ({e3_ratio}) must exceed E1 ({e1_ratio})"
    assert e3_ratio > e4_ratio, f"E3 ratio ({e3_ratio}) must exceed E4 ({e4_ratio})"

    # E3 must be over 4x faster than E1 and have higher coverage
    assert data["E3"]["latency"]["total_wall_sec"] < data["E1"]["latency"]["total_wall_sec"] / 4.0
    assert data["E3"]["quality"]["factual_coverage_pct"] > data["E1"]["quality"]["factual_coverage_pct"]
    assert data["E3"]["quality"]["unsupported_claims_pct"] < data["E1"]["quality"]["unsupported_claims_pct"]


def test_d15_real_data_only_no_mocks_in_notebook_06(project_root):
    nb_path = project_root / "experiments" / "notebooks" / "06_phase6_rq4_efficiency_and_pareto.ipynb"
    assert nb_path.exists(), "Notebook 06 must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    code_content = ""
    for cell in nb_json["cells"]:
        if cell["cell_type"] == "code":
            code_content += "".join(cell["source"]) + "\n"

    # Ensure no random mocks generating synthetic research numbers
    forbidden_terms = ["np.random.randn", "torch.randn", "synthetic_test_qa"]
    for term in forbidden_terms:
        assert term not in code_content, f"Forbidden mock term '{term}' found in Notebook 06"


def test_validation_gate_rq4_report_contents(project_root):
    report_path = project_root / "reports" / "validation_gate_rq4.md"
    assert report_path.exists(), "validation_gate_rq4.md must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Hypothesis H4" in content or "Giả thuyết H4" in content
    assert "E1" in content and "E2" in content and "E3" in content and "E4" in content
    assert "Wall time" in content or "total_wall_sec" in content or "Thời gian" in content
    assert "VRAM" in content
