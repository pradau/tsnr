# Regression Test Runbook

This runbook explains how to regenerate tSNR regression outputs from scratch and verify that:
- stats JSON files are in compact form by default
- plotting still works
- pytest regression tests pass

## Scope

This covers:
- tracked phantom regression fixtures in `phantom_regression/`
- local brain regression fixtures in `brain_regression/` (gitignored)
- test suites:
  - `tests/test_tsnr.py`
  - `tests/test_plot_tsnr_stats.py`

## Prerequisites

- Run all commands from repo root: `tSNR/`
- `uv` installed
- regression inputs present:
  - `phantom_regression/inputs/*.npz` (tracked)
  - `brain_regression/inputs/*.nii.gz` (local only)

Install dependencies:

```bash
uv sync --group dev
```

## 1) Regenerate Phantom Derivatives (Patch ROI)

```bash
uv run tsnr.py "phantom_regression/inputs/AlbertaChildrensHospital_Basic_fMRIQASnap_2024_03_11_11_06_50__E52718S1.npz" phantom --output-dir "phantom_regression/derivatives_patch"
uv run tsnr.py "phantom_regression/inputs/AlbertaChildrensHospital_Basic_fMRIQASnap_2026_04_02_14_26_27__E53214S1.npz" phantom --output-dir "phantom_regression/derivatives_patch"
```

Expected outputs:
- `phantom_regression/derivatives_patch/*_tsnr_map.nii.gz`
- `phantom_regression/derivatives_patch/*_tsnr_stats.json`

## 2) Regenerate Phantom Derivatives (full_minus_edges ROI)

```bash
uv run tsnr.py "phantom_regression/inputs/AlbertaChildrensHospital_Basic_fMRIQASnap_2024_03_11_11_06_50__E52718S1.npz" phantom --phantom-roi-mode full_minus_edges --output-dir "phantom_regression/derivatives_full_minus_edges"
uv run tsnr.py "phantom_regression/inputs/AlbertaChildrensHospital_Basic_fMRIQASnap_2026_04_02_14_26_27__E53214S1.npz" phantom --phantom-roi-mode full_minus_edges --output-dir "phantom_regression/derivatives_full_minus_edges"
```

Expected outputs:
- `phantom_regression/derivatives_full_minus_edges/*_tsnr_map.nii.gz`
- `phantom_regression/derivatives_full_minus_edges/*_tsnr_stats.json`

## 3) Regenerate Brain Derivatives (Local, gitignored)

```bash
uv run tsnr.py "brain_regression/inputs/sub-3334_GOOD.nii.gz" brain --output-dir "brain_regression/derivatives"
uv run tsnr.py "brain_regression/inputs/sub-3341_BAD.nii.gz" brain --output-dir "brain_regression/derivatives"
```

Notes:
- If no matching T1 exists in an `anat/` location, fallback masking warning is expected.
- `brain_regression/` is local-only and should stay untracked.

## 4) Rebuild Report Plots

### Phantom patch reports

```bash
uv run plot_tsnr_stats.py --stats-dir "phantom_regression/derivatives_patch" --out-dir "phantom_regression/reports_patch"
```

### Phantom full_minus_edges reports

```bash
uv run plot_tsnr_stats.py --stats-dir "phantom_regression/derivatives_full_minus_edges" --out-dir "phantom_regression/reports_full_minus_edges"
```

### Brain reports (local)

```bash
uv run plot_tsnr_stats.py --stats-dir "brain_regression/derivatives" --out-dir "brain_regression/reports"
```

Expected report files in each report directory:
- `metrics_panel_by_session_sem.png`
- `slice_metrics_panel_by_session_sem.png`
- `aggregated_metric_summary.csv`

## 5) Run Regression Tests

```bash
uv run pytest tests/test_tsnr.py tests/test_plot_tsnr_stats.py
```

Expected result:
- all tests pass

## 6) Review Changed Files

```bash
git status --short
```

Typical tracked diffs after fixture regeneration:
- `phantom_regression/derivatives_patch/*_tsnr_stats.json`
- `phantom_regression/derivatives_full_minus_edges/*_tsnr_stats.json`

Possible local-only diffs (should remain untracked):
- `brain_regression/derivatives/*`
- `brain_regression/reports/*`

## Troubleshooting

- `Error: no matching BOLD NIfTIs ...` when pointing `tsnr.py` at `phantom_regression/inputs/`:
  - expected; that folder contains `.npz`, not `*_bold.nii*`
  - run per-file `.npz` commands as shown above
- Missing brain inputs:
  - restore local files under `brain_regression/inputs/`
- Plot warnings about unresolved phantom session date in brain files:
  - expected when filenames are not QA date-formatted

