"""Centralized registry cho multimodal lecture summarizer candidates.

Module này là single source of truth cho:
- candidates.yaml       — danh sách model candidates (id, class, kwargs, budget)
- ablation_specs.yaml   — 2^k toggle grids per task
- leaderboard_weights.yaml — multi-metric weights per task

Được load bởi `experiments.scripts.registry_loader`.
"""
from pathlib import Path

REGISTRY_DIR = Path(__file__).resolve().parent