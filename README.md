# tSNR Calculator

Compute raw temporal signal-to-noise ratio (tSNR) from fMRI data with two modes:

- `phantom`: fixed square ROI summary on one slice
- `brain`: whole-brain masked summary from a 4D NIfTI

This project implements the behavior in `tSNR_SPEC.md`.

## Quick Start

```bash
uv sync --group dev
uv run python main.py /path/to/run.nii.gz phantom
uv run python main.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
uv run pytest
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

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
uv run python main.py <input> <mode> [options]
```

Examples:

```bash
uv run python main.py /path/to/run.nii.gz phantom
uv run python main.py /path/to/run.nii.gz phantom --roi-size 15 --slice-index 12
uv run python main.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
uv run python main.py /path/to/cache.npz phantom
```

## Inputs

- NIfTI: `.nii` or `.nii.gz` (must be 4D, at least 2 time points)
- NPZ: `fMRIQA` phantom cache (`.npz`) is supported only for `phantom` mode

## Outputs

Outputs are written to `--output-dir` (or input directory by default):

- `<basename>_tsnr_map.nii.gz`
- `<basename>_tsnr_stats.json`

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
- mode-specific `parameters`

## Mode Notes

### Phantom mode

- Uses an intensity-weighted center of mass on the selected slice mean image
- Places a fixed odd-sized square ROI (`--roi-size`)
- Shifts ROI inward if needed to keep full ROI size within image bounds

### Brain mode

- Builds mask from mean volume threshold:
  - baseline is mean signal over positive voxels
  - cutoff is `threshold * baseline`
- Applies binary erosion (`--erosion-voxels`)

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
