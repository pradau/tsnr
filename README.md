# tSNR Calculator

Compute raw temporal signal-to-noise ratio (tSNR) from fMRI data with two modes:

- `phantom`: fixed square ROI summary on one slice
- `brain`: whole-brain masked summary from a 4D NIfTI

Behavior and the stats JSON contract are documented in this file. Optional FSL-based T1 brain masking for `brain` mode is described under **Brain mode** below.

## Quick Start

```bash
uv sync --group dev
uv run python tsnr.py /path/to/run.nii.gz phantom
uv run python tsnr.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
uv run pytest
```

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

Run with:

```bash
uv run python tsnr.py <input> <mode> [options]
```

Examples:

```bash
uv run python tsnr.py /path/to/run.nii.gz phantom
uv run python tsnr.py /path/to/run.nii.gz phantom --roi-size 15 --slice-index 12
uv run python tsnr.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
uv run python tsnr.py /path/to/cache.npz phantom
uv run python tsnr.py /path/to/run.nii.gz brain --full-fov-maps
```

## Inputs

- NIfTI: `.nii` or `.nii.gz` (must be 4D, at least 2 time points)
- NPZ: `fMRIQA` phantom cache (`.npz`) is supported only for `phantom` mode

## Outputs

Outputs are written to `--output-dir` (or input directory by default):

- `<basename>_tsnr_map.nii.gz`
- `<basename>_tsnr_stats.json`

By default, `*_tsnr_map.nii.gz` and optional `*_Tmean.nii.gz` / `*_Tstd.nii.gz` voxels **outside** the analysis ROI (phantom patch or brain mask) are set to **NaN** so maps align with the JSON summary. Use `--full-fov-maps` to write the full field of view instead. The stats JSON includes `output_map_censoring` (`roi_masked` or `full_fov`).

Basename behavior:

- `file.nii` -> `file`
- `file.nii.gz` -> `file`
- `file.npz` -> `file`

Stats JSON includes:

- input metadata (`input_file`, `input_type`, `mode`)
- shape/time info (`volume_shape`, `n_timepoints`)
- summary stats (`tsnr_mean`, `tsnr_median`, `tsnr_std`, `tsnr_min`, `tsnr_max`)
- voxel count (`n_voxels_in_roi`)
- map affine source (`map_affine_source`)
- map masking mode (`output_map_censoring`)
- mode-specific `parameters` (see below; brain mode includes `brain_masking`)

## Mode Notes

### Phantom mode

- Uses an intensity-weighted center of mass on the selected slice mean image
- Places a fixed odd-sized square ROI (`--roi-size`)
- Shifts ROI inward if needed to keep full ROI size within image bounds

### Brain mode

Brain masking tries a **T1-based** pipeline first when a T1 NIfTI is found under `anat/` (see layout below). If that pipeline succeeds, the ROI is the T1 brain mask **registered to the mean functional image** (BET on T1, BET on mean EPI, FLIRT, inverse warp). Otherwise the tool **falls back** to an intensity mask and prints one warning to stderr.

**Intensity fallback** (used when no T1 is found or the T1 pipeline fails):

- baseline is mean signal over positive voxels in the temporal mean
- cutoff is `threshold * baseline`
- binary erosion (`--erosion-voxels`)

`parameters.threshold` and `parameters.erosion_voxels` always record CLI values. They define the mask only when `parameters.brain_masking.method` is `mean_intensity`.

**BIDS-style layout for T1 discovery:** the functional file is usually `.../<subject|session>/func/<run>_bold.nii.gz` and T1 is the first `*.nii*` under `.../<subject|session>/anat/`. If the bold file sits next to `anat/` instead, that directory is used.

**`parameters.brain_masking` (brain mode JSON):**

- `method`: `t1_bet_registered_to_mean_epi` when the T1-derived mask was applied; `mean_intensity` when the ROI came from thresholding plus erosion (including all fallbacks).
- `t1_path`: absolute path to the T1 used for BET, or `null` if none was found or used.
- `t1_to_functional_pipeline`: `success` if the T1 mask was warped to functional space; `not_attempted_no_t1` if no T1 was found; `failed` if BET/FLIRT or mask loading failed; `not_attempted_no_nifti` only in unusual cases without spatial NIfTI metadata.
- `detail`: `null` on success; otherwise a short explanation or error text (for example stderr from FSL when `failed`).

## Validation Behavior

The CLI fails with a nonzero exit for invalid inputs, including:

- unsupported extensions
- non-4D NIfTI inputs
- NPZ used with `brain` mode
- invalid ROI/threshold/erosion arguments
- empty ROI or empty brain mask after processing
- non-finite input values

## Testing

Run all tests:

```bash
uv run pytest
```

Current acceptance tests cover:

- NIfTI phantom and brain happy paths
- zero-std handling
- ROI boundary handling
- invalid dimension errors
- empty mask failures
- NPZ phantom compatibility (v1/v2 cache versions)
- `.nii.gz` output naming contract
- JSON output contract fields
- BIDS-style and flat-layout T1 discovery (`find_t1_in_anat`)
- brain-mode `parameters.brain_masking` (fallback when no T1; mocked T1 pipeline failure)
