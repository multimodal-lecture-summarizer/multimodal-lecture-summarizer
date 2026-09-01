# Migration Log — 4 Plans → 1 Unified

**Date:** 2026-09-01
**Action:** Gộp 4 plans rời rạc thành `260901-unified-scientific-benchmark` (6 phases), sau đó xóa các plan cũ.
**Requested by:** user (`hãy gộp thành 1 plan duy nhất ... sau đó xóa các plan cũ đi`)

## Nguồn

| # | Dir cũ | Loại | Phases/status | Đích trong unified |
|---|--------|------|---------------|-------------------|
| 1 | `plans/260830-1917-scientific-benchmark/` | Non-ck scientific framework (8 docs: `00-master-plan`→`06-risks`, `decisions-log`, `related-work` + `probes/` cache) | No ck status | `framework/` (11 docs copied, probes/cache KHÔNG copy — cache là derived data, không phải plan) + encoded trong `plan.md` Overview/Constraints và `phase-01` |
| 2 | `plans/260831-fix-notebooks-3-5/` | ck plan | 4 phases Pending | `phase-02` (P2 Notebook Integrity) + `phase-06` consistency sweep; original `plan.md` archived `archive/sources/260831-fix-notebooks-3-5-plan.md` |
| 3 | `plans/260831-ratelimit-free-summarization/` | ck plan | 4 phases Completed (`4013290`+`4ab8963`) | `phase-03` (P3 RateLimit-Free Engine); archived `archive/sources/260831-ratelimit-free-summarization-plan.md` |
| 4 | `plans/260831-benchmark-tdd-hardening/` | ck plan | 4 phases Pending | `phase-04` VISTA, `phase-05` YTSeg+Retrieval, `phase-06` Validation Gate; archived `archive/sources/260831-benchmark-tdd-hardening-plan.md` |

## Quyết định hợp nhất (reconciled)

- **VISTA mandatory vs fallback:** `hardening` validate chọn `VISTA bắt buộc` (no auto TIB fallback) override `260830 D-T12` Tier2. Unified giữ **mandatory** (user decision 2026-09-01) nhưng nêu conflict trong `plan.md Open Questions` + `phase-04 Risk`. TIB 80 giữ như external validation luôn.
- **n=300 locked:** `hardening` Q1 khóa `300` (không phải 100-300 range) — unified P5 khóa `300`.
- **Bỏ TDD:** `hardening` validate Q3 bỏ TDD red→green — unified ghi rõ `post-hoc verification` cho mọi phase, không yêu cầu failing tests trước impl.
- **Probes/cache không migrate:** `260830 probes/cache/vtssum` (~2k JSONs) là derived probe data, không phải plan logic — không copy, reference tại origin nếu cần. Nếu cần reproduce, chạy lại probe per `03-colab-runbook §5`.

## Unified structure

```
plans/260901-unified-scientific-benchmark/
  plan.md + phase-01..06 (6 phases, DAG P1→P2→P3→P5→P6, P1→P4→P6)
  framework/  (00..06 + decisions-log + related-work + gap-analysis + README)
  archive/sources/ (3 original ck plan.md)
  MIGRATION.md (this file)
```

## Verify before delete

- `ck plan validate plans/260901-unified-scientific-benchmark/plan.md` → `[OK] Valid — 6 phases`
- `framework/` 11 docs present, `archive/sources/` 3 files present
- 4 dirs cũ đã xóa sau verify

## Rollback

Nếu cần rollback: `git checkout` hoặc restore từ `archive/sources/` + `framework/` → tách lại 4 dirs. Git history vẫn giữ commits `4013290`, `4ab8963` liên quan.
