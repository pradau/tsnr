# Phantom regression datasets (versioned)

This directory is committed on purpose so clones and CI can rely on the same Alberta QA phantom inputs and recomputed **`tsnr.py`** / **`plot_tsnr_stats.py`** artifacts.

Do not confuse with **`brain_regression/`**, which is gitignored (local-only brain EPI).

## Layout

| Path | Contents |
|------|----------|
| `inputs/` | fMRIQA-style phantom cache `*.npz` files |
| `derivatives_patch/` | Stats and tSNR maps from **`--phantom-roi-mode patch`** (fixed square ROI) |
| `reports_patch/` | PNG and CSV from **`plot_tsnr_stats.py --stats-dir derivatives_patch`** |
| `derivatives_full_minus_edges/` | Stats and maps from **`--phantom-roi-mode full_minus_edges`** |
| `reports_full_minus_edges/` | Reports for that derivatives folder |
| `exports_nifti/` | Optional 4D NIfTI exports of the same phantom sessions |

Naming **`BAD`** vs **`GOOD`** in filenames reflects presence vs absence of the intermittent white-pixel artifact; see the main README **Local regression datasets** section.
