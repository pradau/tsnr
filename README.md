# tSNR Calculator

Author: Perry Radau  
Date: 2026-04-20  

Purpose: Compute temporal signal-to-noise ratio (tSNR) maps and per-run JSON statistics from 4D fMRI. Phantom mode uses a fixed-slice ROI; brain mode summarizes within a brain mask (optional FSL-based registration from T1 when available).

Dependencies: Python 3.10+; Python packages nibabel, numpy, and scipy (versions in [`pyproject.toml`](pyproject.toml)); [uv](https://docs.astral.sh/uv/) for installs; optional [FSL](https://fsl.fmrib.ox.ac.uk/fsl) for T1 brain masking in brain mode; pytest in the dev dependency group for tests.

Compute raw temporal signal-to-noise ratio (tSNR) from fMRI data with two modes:

- `phantom`: fixed square ROI summary on one slice
- `brain`: whole-brain masked summary from a 4D NIfTI

Behavior and the stats JSON contract are documented below. Optional FSL-based T1 brain masking for `brain` mode is described under **Brain mode**.

## Quick Start

```bash
uv sync --group dev
uv run tsnr.py /path/to/run.nii.gz phantom
uv run tsnr.py /path/to/bids/func/sub-01_task-rest_bold.nii.gz brain
uv run pytest
```

Default brain masking uses T1 BET plus registration when a `*T1w.nii*` is available under `anat/` relative to the BOLD file (typical BIDS `func` / `anat` layout). Options `--threshold` and `--erosion-voxels` tune the **intensity fallback** when that pipeline is not used; see **Brain mode**.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- **Brain mode (T1 mask):** [FSL](https://fsl.fmrib.ox.ac.uk/fsl) on the host, with `FSLDIR` set (or install at the default path used by the script). If FSL is missing or the T1 pipeline fails, analysis falls back to the intensity-based mask; see `parameters.brain_masking` in the stats JSON.

## Setup

From this directory:

```bash
uv sync
```

Include dev dependencies:

```bash
uv sync --group dev
```

## CLI Usage

Entry point is **`tsnr.py`** in this directory. Run from the project root:

```bash
uv run tsnr.py <input> <mode> [options]
```

`uv` executes the script with the project environment. Equivalent forms include `uv run python tsnr.py` and `uv run python -m tsnr`.

Examples:

```bash
uv run tsnr.py /path/to/run.nii.gz phantom
uv run tsnr.py /path/to/run.nii.gz phantom --roi-size 15 --slice-index 12
uv run tsnr.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
uv run tsnr.py /path/to/cache.npz phantom
uv run tsnr.py /path/to/run.nii.gz brain --full-fov-maps
uv run tsnr.py /path/to/run.nii.gz brain --write-tmean-tstd --output-dir /path/to/out
uv run tsnr.py /path/to/ses-x/func brain --output-dir /path/to/derivatives/tsnr
```

**Directory input (multi-echo / batch):** pass a **folder** (for example a BIDS `func/` directory) instead of a single file. The tool finds all `*_bold.nii.gz` and `*_bold.nii` in that folder (non-recursive), sorts them by filename, and runs the same mode and options on **each** file. Use `--input-pattern GLOB` to use a single custom glob instead of the default BOLD patterns (for example if your files do not contain `_bold` in the name). 

**Batch output naming (multi-task, multi-echo):** Derivatives are named from each input file’s stem (the filename without `.nii` / `.nii.gz`). BIDS-style BOLD names already encode subject, session, `task-*`, `echo-*`, and `*_bold`, so each map stays self-describing without extra prefixes. For example, if `func/` contains two tasks with four echoes each (eight runs), you get eight distinct outputs such as `sub-01_ses-1a_task-rest_echo-2_bold_tsnr_map.nii.gz` alongside `..._task-other_echo-1_bold_tsnr_map.nii.gz`, one pair of stats/maps per source BOLD. When `--output-dir` is set, all of those files are written there.

**FSL scratch (brain mode):** Intermediate BET/FLIRT files for a run live under `--output-dir` (or the input directory) at `.tsnr_fsl_work/<basename>/`, where `<basename>` matches that run’s output stem, so batch jobs do not overwrite each other’s working files.

Useful options:

- `--first-timepoint` / `--last-timepoint`: 0-based volume range. Default `first-timepoint=2` drops the first two volumes to reduce non-steady-state transient effects in fMRI signal. Use `--first-timepoint 0` to include from the first volume, or `--first-timepoint 1` to drop only the first.
- `--write-tmean-tstd`: also write temporal mean and standard deviation maps (`*_Tmean.nii.gz`, `*_Tstd.nii.gz`).
- `--full-fov-maps`: do not NaN voxels outside the ROI in written maps.
- `--input-pattern`: when `input` is a directory, override the default `*_bold.nii.gz` / `*_bold.nii` discovery with one glob.

## Inputs

- **Single file:** NIfTI `.nii` or `.nii.gz` (must be 4D, at least 2 time points), or for `phantom` mode only, an fMRIQA-style phantom cache `.npz`.
- **Directory:** only NIfTI batching is supported (default BOLD globs above). NPZ remains single-file `phantom` use only.

## Outputs

Outputs go to `--output-dir`, or to a default `derivatives/tsnr` location when omitted:

- BIDS-style `.../ses-*/func/<run>_bold.nii.gz` input -> `.../ses-*/derivatives/tsnr/`
- Other file locations -> `<input_parent>/derivatives/tsnr/`

- `<basename>_tsnr_map.nii.gz`
- `<basename>_tsnr_stats.json`
- Optional: `<basename>_Tmean.nii.gz`, `<basename>_Tstd.nii.gz` when `--write-tmean-tstd` is set

By default, map voxels **outside** the analysis ROI (phantom patch or brain mask) are set to **NaN** so maps match the JSON summary. Use `--full-fov-maps` for full field of view. The stats JSON includes `output_map_censoring` (`roi_masked` or `full_fov`).

Basename rules:

- `file.nii` -> `file`
- `file.nii.gz` -> `file`
- `file.npz` -> `file`

### Stats JSON (common fields)

- Input metadata: `input_file`, `input_type`, `mode`
- Shape and time: `volume_shape`, `n_timepoints`, `timepoint_selection`
- Summary: `tsnr_mean`, `tsnr_median`, `tsnr_std`, `tsnr_min`, `tsnr_max`, `ftsnr`, `roi_mean_signal_std`, `n_voxels_in_roi`
  - `tsnr_std` is the spatial standard deviation of per-voxel tSNR across the ROI.
  - `roi_mean_signal_std` is the temporal standard deviation of the ROI-mean fMRI signal across frames (signal units, suitable for longitudinal plot error bars). `ftsnr` is the mean of that ROI-mean series divided by `roi_mean_signal_std`.
  - `roi_mean_tr_spike_metrics`: see **ROI mean TR spike metrics** below.
- `map_affine_source`, `output_map_censoring`
- Mode-specific `parameters` (see below)

#### ROI mean TR spike metrics (`roi_mean_tr_spike_metrics`)

These fields summarize **per-TR (volume) outliers** on the **ROI-mean fMRI time course**: for each timepoint, the mean signal is taken over the same voxels used for `ftsnr` / `roi_mean_signal_std` (phantom ROI or brain mask). Computations use the series **after** `timepoint_selection` (by default the first two volumes are dropped; see `--first-timepoint`).

Outliers are flagged with **robust z** using **|z| > 3**:

1. **Robust z (median and MAD):** Let `m` be the median of the ROI-mean series and `MAD` the median absolute deviation from `m`. A robust scale is `1.4826 * MAD` (for Gaussian-like tails, this tracks the standard deviation). If that scale is effectively zero, the implementation falls back to the sample standard deviation of the series. Robust z is `(value - m) / scale`.

Typical JSON keys (see your stats file for the exact set): `n_timepoints`, `method_robust_z`, `robust_median`, `mad`, `robust_sigma`, `n_tr_abs_robust_z_gt_3`, `pct_tr_abs_robust_z_gt_3`, `max_abs_robust_z`.

- **`robust_z_per_tr`:** list of **signed** robust z-scores, one entry per TR (same order as the ROI-mean series after `timepoint_selection`). Length matches `n_timepoints` when the series is valid and has at least two points; degenerate cases may yield an empty list or a single value. Used for scalar spike summaries and echo-level plots; **re-run `tsnr.py` in `brain` mode** to refresh if missing.

- **`roi_mean_signal_per_tr`:** ROI-mean signal value per TR (same order and length as `robust_z_per_tr` when present). Stored so **TR-index figures** can apply a **linear detrend** before robust z without changing any other reported statistics. Omitted in older JSON; refresh with current `tsnr.py`.

High counts or large `max_abs_*` values point to TRs with unusually high or low whole-ROI signal relative to the rest of the run (for example motion spikes, acquisition glitches, or extreme signal dropouts).

## Mode Notes

### Phantom mode

- Uses an intensity-weighted center of mass on the selected slice mean image
- Places a fixed odd-sized square ROI (`--roi-size`)
- Shifts ROI inward if needed to keep full ROI size within image bounds

### Brain mode

Brain masking tries a **T1-based** pipeline first when a suitable T1w NIfTI is found under `anat/` (see **T1 discovery**). If that pipeline succeeds, the ROI is the T1 brain mask **registered to the mean functional image** (BET on T1, BET on mean EPI, FLIRT, inverse warp). Otherwise the tool **falls back** to an intensity mask and prints a warning to stderr.

**Intensity fallback** (no T1 found, or BET/registration failure):

- Baseline is the mean signal over positive voxels in the temporal mean
- Cutoff is `threshold * baseline`, then binary erosion (`--erosion-voxels`)

**T1 discovery (BIDS-friendly, not BIDS-exclusive)**

The tool looks for `anat/` in two places relative to the functional NIfTI:

1. `parent.parent / "anat"` (typical: `.../ses-x/func/run.nii.gz` -> `.../ses-x/anat/`)
2. `parent / "anat"` (flat layout: `.../folder/run.nii.gz` -> `.../folder/anat/`)

Only files matching `*T1w.nii.gz` or `*T1w.nii` are candidates. If several exist, the one with the **earliest acquisition time** is chosen, using the sidecar JSON when possible:

- `AcquisitionDateTime` (ISO 8601), or
- `AcquisitionDate` plus `AcquisitionTime`

If no usable time is in JSON for a file, its modification time is used so the **earliest** file still wins among those without sidecar times.

If there is **no** `anat/` on those paths, or no `*T1w.nii*`, only the intensity mask is used (no BET).

**`parameters` in brain mode**

- `mask_baseline_mean_positive_signal`: mean over positive voxels in the temporal mean (descriptive).
- `intensity_brain_mask`: present **only when** the spatial mask was built from intensity rules. It contains `threshold` and `erosion_voxels` actually used for that mask. Omitted when the mask came from the T1 pipeline (those CLI flags did not define the ROI).
- `brain_masking`:
  - `method`: `t1_bet_registered_to_mean_epi` or `mean_intensity`
  - `t1_path`: absolute path to the T1 NIfTI used for BET when applicable, else `null`
  - `t1_to_functional_pipeline`: `success`, `not_attempted_no_t1`, `failed`, or `not_attempted_no_nifti`
  - `detail`: `null` on success; otherwise a short message or captured error text

## Validation Behavior

The CLI exits nonzero for invalid inputs, including:

- Unsupported extensions
- Non-4D NIfTI inputs
- NPZ used with `brain` mode
- Invalid ROI, threshold, or erosion arguments
- Empty ROI or empty brain mask after processing
- Non-finite input values

## Testing

```bash
uv run pytest
```

`tests/test_tsnr.py` covers `tsnr.py`: phantom and brain paths, T1 discovery and ordering, brain masking JSON, intensity-only parameters, BET fallback when FSL steps fail, timepoint selection (including default `--first-timepoint`), directory batch discovery and CLI, output naming, and ROI mean TR spike metrics (including `robust_z_per_tr` and `roi_mean_signal_per_tr`).

`tests/test_plot_tsnr_stats.py` covers `plot_tsnr_stats.py` (and batch helpers where applicable): BIDS entity parsing, discovery and aggregation, subject/session filters, `run_report` outputs (echo panels, CSV, optional `robust_z_vs_tr` and `roi_mean_signal_vs_tr` figures), detrended robust-z helpers, and per-subplot outlier counts on TR-index plots.

The installable package wheel still contains only `tsnr.py` (see `pyproject.toml`). The plotting scripts `plot_tsnr_stats.py` and `plot_robust_z_tr_all_sessions.py` live in the repository root; run them with `uv run` from this directory.

## Plotting tSNR, fTSNR, and ROI variability

Use `plot_tsnr_stats.py` to generate comparison plots from existing derivatives stats JSON files. This script is read-only on derivatives (it only reads `*_tsnr_stats.json`) and writes report files to a separate output folder.

Run from the project root:

```bash
uv run plot_tsnr_stats.py --bids-root /path/to/bids
```

Useful options:

- `--out-dir`: output directory for plots and CSV (default: `reports/tsnr_plots`)
- `--error-bar {sd,sem,ci95}`: spread used for CSV `error` column and for panel y error bars when error bars are shown (default: `sem`)
- `--subject sub-XXXX` and `--session ses-X`: optional filters
- **When both `--subject` and `--session` are set** (typical single-session QA): **one line per task** by default (no pooling across tasks), and **echo panels omit error bars** by default so you see separate curves without SEM/SD caps. The CSV still lists `mean` and `error` per group.
- `--pool-across-tasks`: pool all tasks for each session/echo (one line per session; use when you explicitly want aggregation across tasks).
- `--group-by-task`: when you are **not** using both subject and session filters, this splits lines by `session + task` instead of session only (same grouping key as the default single-session behavior).
- `--show-error-bars`: force capped error bars on the echo panels (for example if you have multiple runs per task per echo and want variability shown).
- `--no-error-bars`: force line-only panels even when tasks are pooled.

Example for one subject and session (defaults: split tasks, no error bars on PNGs):

```bash
uv run plot_tsnr_stats.py \
  --bids-root /Users/pradau/Data/bids-perry \
  --subject sub-3334 \
  --session ses-1a \
  --out-dir reports/tsnr_plots_sub-3334_ses-1a
```

Expected inputs:

- Stats files under `sub-*/ses-*/derivatives/tsnr/*_tsnr_stats.json`
- Filenames should include BIDS entities used for grouping: `sub-*`, `ses-*`, `task-*`, `echo-*`
- Required JSON metrics: `tsnr_mean`, `tsnr_std`, `ftsnr`, `roi_mean_signal_std`
- Optional spike QC: if **every** stats file includes `roi_mean_tr_spike_metrics` with `max_abs_robust_z`, `pct_tr_abs_robust_z_gt_3`, and `n_tr_abs_robust_z_gt_3`, a second figure and extra CSV rows are produced (see below).
- **`--robust-z-tr-panels`** (requires **`--subject`** and **`--session`**): writes **`robust_z_vs_tr_<subject>_<session>.png`**, a multi-panel figure of **robust z vs TR index** with **one subplot per `*_tsnr_stats.json`** in that session (for example eight panels when there are eight BOLD-derived stats files). When **`roi_mean_signal_per_tr`** is in the JSON (current `tsnr.py`), the plotter **linearly detrends** that ROI-mean series **then** applies the same robust-z rule (**only for this figure**; echo panels and CSV still use undetrended spike metrics). Older stats without raw per-TR means fall back to plotting **`robust_z_per_tr`** from the file. Refresh stats with `uv run tsnr.py "<bids>/sub-.../ses-.../func" brain` if needed.

- **`--roi-mean-signal-tr-panels`** (requires **`--subject`** and **`--session`**): writes **`roi_mean_signal_vs_tr_<subject>_<session>.png`**, the **raw ROI-mean fMRI signal** (same units as in the stats JSON) vs TR index—**no robust z**—so you can judge drift and spikes without median/MAD being influenced by many outliers. Requires **`roi_mean_signal_per_tr`** in each stats file.

Example (single session, TR-index grid plus usual panels):

```bash
uv run plot_tsnr_stats.py \
  --bids-root /path/to/bids \
  --subject sub-3334 \
  --session ses-1a \
  --error-bar sem \
  --out-dir reports/tsnr_plots_sub-3334_ses-1a \
  --robust-z-tr-panels
```

**Batch all sessions:** `plot_robust_z_tr_all_sessions.py` discovers every `sub-*/ses-*` pair that has stats under `derivatives/tsnr/` and writes one **`robust_z_vs_tr_<sub>_<ses>.png`** per session to **`--out-dir`** (default: `<bids-root>/reports/robust_z_tr_sessions`). Add **`--roi-mean-signal-tr-panels`** to also write **`roi_mean_signal_vs_tr_<sub>_<ses>.png`** per session. Use **`--dry-run`** to list sessions without plotting.

```bash
uv run plot_robust_z_tr_all_sessions.py --bids-root /path/to/bids
uv run plot_robust_z_tr_all_sessions.py --bids-root /path/to/bids --roi-mean-signal-tr-panels
```

Outputs:

- `metrics_panel_by_echo_<error>.png` (three subplots, top to bottom: `ftsnr`, `roi_mean_signal_std`, `tsnr_mean`; error bars only when enabled)
- `spike_metrics_panel_by_echo_<error>.png` when all inputs carry spike metrics (three subplots: max |robust z|, % TRs with |robust z| > 3, TR count with |robust z| > 3; error bars only when enabled)
- `robust_z_vs_tr_<subject>_<session>.png` when `--robust-z-tr-panels` is set and per-TR data are present (linear detrend before z when `roi_mean_signal_per_tr` is stored)
- `roi_mean_signal_vs_tr_<subject>_<session>.png` when `--roi-mean-signal-tr-panels` is set (raw ROI mean vs TR)
- `aggregated_metric_summary.csv` in tidy long format:
  - columns: `metric,sub,ses,echo,task,n_runs,mean,error`
  - one row per `(metric, ses, echo[, task])`; `metric` includes core summaries and, when available, the three spike fields above

When `--subject` and/or `--session` filters are provided, the panel title includes those labels.
