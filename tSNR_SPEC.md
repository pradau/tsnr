# fMRI tSNR Calculator - Implementation Spec

## Overview

A Python CLI that computes raw tSNR from fMRI data and writes a 3D tSNR map plus summary statistics.

It supports two analysis modes:

- `phantom`: fixed square ROI on a selected slice
- `brain`: whole-brain masked summary over a 4D NIfTI input

Primary input is a 4D NIfTI file. In `phantom` mode only, the script must also accept `fMRIQA` pixel-cache `.npz` files compatible with `../fMRIQA/plot_fmriqa_snr_vs_time.py`.

No detrending is applied. This computes raw tSNR, not SFNR.

---

## Reuse Requirements

Implementation should reuse the existing `../fMRIQA` phantom logic where sensible instead of re-deriving parallel behavior:

- reuse the `fMRIQA` pixel-cache contract from `../fMRIQA/plot_fmriqa_snr_vs_time.py`
- reuse or mirror the same ROI boundary semantics as `place_fixed_square_roi()`
- reuse or mirror cache-version handling from `load_fmriqa_pixel_cache()` and `read_pixel_cache_file_version()`

Behavior does not need to be byte-for-byte identical to `fMRIQA`, but phantom ROI placement and `.npz` compatibility should not silently drift from that codebase.

---

## Inputs

### Required arguments

- `input`: path to either:
  - a 4D NIfTI file (`.nii` or `.nii.gz`) for `phantom` or `brain`
  - an `fMRIQA` pixel-cache file (`.npz`) for `phantom` only
- `mode`: `phantom` or `brain`

### Optional arguments

- `--output-dir`: directory for outputs (default: same directory as input)

### Phantom mode options

- `--roi-size`: square ROI width and height in voxels (default: 15)
- `--slice-index`: integer slice index to analyze; default is the middle slice

### Brain mode options

- `--threshold`: mask threshold as a fraction of the mean signal among positive voxels in the mean volume (default: 0.25)
- `--erosion-voxels`: number of binary-erosion iterations to apply to the brain mask (default: 2)

---

## Accepted Input Formats

### NIfTI input

- must be a readable `.nii` or `.nii.gz` file
- must contain exactly 4 dimensions
- time dimension must be at least 2
- data are cast to `float64` before computation
- non-finite voxels (`NaN`, `Inf`, `-Inf`) are not allowed; fail with a clear error

### Phantom NPZ input

- supported only when `mode=phantom`
- must be compatible with the `fMRIQA` pixel-cache format written by `save_fmriqa_pixel_cache()`
- cache versions 1 and 2 must be accepted
- required payload is the `volume` array with shape `(n_slices, n_times, rows, cols)`
- `n_times` must be at least 2
- non-finite voxels are not allowed; fail with a clear error

### Rejected inputs

- `brain` mode with `.npz` input
- 3D NIfTI input
- NIfTI with more than 4 dimensions
- empty files or unreadable files
- inputs where the requested `roi-size` is larger than the in-plane image dimensions

---

## Validation Rules

Validate arguments and fail fast with a nonzero exit and a concise error message when:

- `mode` is invalid
- file suffix does not match the selected mode
- `--slice-index` is outside valid range
- `--roi-size` is not a positive odd integer
- `--threshold` is not strictly between 0 and 1
- `--erosion-voxels` is negative
- the selected phantom ROI or brain mask contains zero voxels after all processing

Representative error messages should be specific enough to diagnose the failure, for example:

- `Expected 4D NIfTI input, got shape (64, 64, 28)`
- `brain mode does not accept .npz input`
- `roi-size 31 exceeds image dimensions 21x21`
- `brain mask is empty after erosion`

If the input is NIfTI, preserve affine and header where possible. If the input is NPZ, there is no source affine to preserve.

---

## Processing Steps

### Shared tSNR computation

1. Load the source data into a numeric 4D array.
2. Normalize to a common internal layout:
   - NIfTI: `(x, y, z, t)`
   - NPZ phantom cache: convert from `(slices, time, rows, cols)` to an equivalent internal 4D layout before analysis
3. Compute voxel-wise mean over time.
4. Compute voxel-wise sample standard deviation over time using `ddof=1`.
5. Compute tSNR as `mean / std`.
6. Set tSNR to `0` wherever `std <= 0` or the result is non-finite.
7. Apply the mode-specific mask or ROI to extract summary voxels.

### Phantom mode

1. Select the target slice:
   - use `--slice-index` when provided
   - otherwise use the middle slice
2. Build a 2D reference image for ROI placement from the selected slice mean image.
3. Compute the intensity-weighted center of mass on that 2D reference image.
4. Place a square `roi-size x roi-size` ROI centered on the nearest voxel to that center of mass.
5. If the centered ROI would extend outside the image, shift it inward so the final ROI keeps the requested size.
6. If the image is smaller than `roi-size` in either in-plane dimension, fail with a clear error.
7. Extract tSNR values from that fixed ROI on the selected slice.

### Brain mode

1. Compute the mean volume across time.
2. Create a provisional mask of strictly positive voxels in the mean volume.
3. Compute the threshold baseline as `mean(mean_volume[mean_volume > 0])`.
4. Threshold at `threshold x baseline` to create the binary brain mask.
5. Apply `scipy.ndimage.binary_erosion` for `erosion-voxels` iterations.
6. If thresholding or erosion produces an empty mask, fail with a clear error.
7. Extract tSNR values from the final eroded mask.

---

## Output Naming

All outputs are written to `output_dir`.

Derive the output basename as follows:

- for `file.nii`, basename is `file`
- for `file.nii.gz`, basename is `file` (strip `.nii.gz` as a unit, not just `.gz`)
- for `file.npz`, basename is `file`

Do not use naive `Path.stem` behavior for `.nii.gz` naming.

When the input is an external `fMRIQA` cache such as `AlbertaY_fMRIQASnap_2021_01_01_00_00_00__E1S1.npz`, preserve that full stem as the basename.

---

## Outputs

### Files written

- `{basename}_tsnr_map.nii.gz`
- `{basename}_tsnr_stats.json`

### tSNR map behavior

- For NIfTI input:
  - write a 3D tSNR NIfTI in the same voxel space as the input
  - preserve affine and header where possible
- For NPZ phantom input:
  - still write a 3D tSNR NIfTI
  - use an identity affine because the cache does not provide a NIfTI affine
  - document this in the JSON output

### JSON contents

```json
{
  "input_file": "...",
  "input_type": "nifti|fmriqa_pixel_cache_npz",
  "mode": "phantom|brain",
  "n_timepoints": 200,
  "volume_shape": [64, 64, 28, 200],
  "tsnr_mean": 0.0,
  "tsnr_median": 0.0,
  "tsnr_std": 0.0,
  "tsnr_min": 0.0,
  "tsnr_max": 0.0,
  "n_voxels_in_roi": 0,
  "map_affine_source": "input|identity",
  "parameters": {}
}
```

`parameters` must echo the effective values used for the selected mode.

`volume_shape` must report the internal analysis layout in `(x, y, z, t)` order, even if the source NPZ was loaded from `(slices, time, rows, cols)`.

Summary statistics are computed over the final selected voxels only:

- `tsnr_mean`: arithmetic mean
- `tsnr_median`: median
- `tsnr_std`: population standard deviation of the selected tSNR values
- `tsnr_min`: minimum selected tSNR
- `tsnr_max`: maximum selected tSNR

For `phantom`, include at minimum:

- `roi_size`
- `slice_index`
- `roi_bounds` as `[row_start, row_end, col_start, col_end]`

For `brain`, include at minimum:

- `threshold`
- `erosion_voxels`
- `mask_baseline_mean_positive_signal`

If no summary voxels remain after masking or ROI selection, do not write misleading NaN stats. Fail instead.

---

## Dependencies

- `nibabel` - NIfTI I/O
- `numpy` - array operations
- `scipy.ndimage` - center of mass and binary erosion
- `pathlib` - all file operations
- `argparse` - CLI
- `json` - stats output
- `pytest` - automated tests

Optional phantom reuse from sibling repo:

- `../fMRIQA/plot_fmriqa_snr_vs_time.py`

---

## Acceptance Tests

Write pytest tests under `tests/` that cover at least the following cases with synthetic data:

- place tests in `tests/` mirroring the eventual source layout
- prefer array-generated fixtures created inside the tests over checked-in binary test data
- use temporary directories and write synthetic NIfTI files with `nibabel`
- for NPZ compatibility tests, either:
  - generate a minimal compatible cache inline, or
  - call the reused `fMRIQA` helper to write a valid cache fixture
- run the suite with `uv run pytest`

1. NIfTI phantom happy path:
   - small 4D synthetic phantom
   - correct tSNR map shape
   - correct phantom ROI summary stats

2. NIfTI brain happy path:
   - small 4D synthetic brain-like volume with background zeros
   - thresholding uses positive-voxel mean, not the global mean including zeros
   - erosion reduces mask size as expected

3. Zero-std handling:
   - constant voxel time series produce tSNR `0`
   - output contains no `NaN` or `Inf`

4. Phantom ROI boundary handling:
   - ROI center near image edge shifts inward and keeps full requested size
   - oversized ROI fails clearly

5. Invalid input dimensionality:
   - 3D NIfTI fails
   - 5D NIfTI fails

6. Empty mask failure cases:
   - thresholded brain mask empty
   - erosion removes all voxels

7. NPZ phantom compatibility:
   - version 1 or version 2 `fMRIQA` pixel-cache style `.npz`
   - summary stats run successfully in `phantom` mode
   - `brain` mode rejects the same `.npz`

8. Output naming:
   - `.nii.gz` input produces `{basename}_tsnr_*` without leaving an extra `.nii`

9. JSON contract:
   - includes `input_type`, `volume_shape`, `map_affine_source`, and mode-specific `parameters`

These tests are the minimum acceptance bar before the script is considered complete.

---

## Constraints and Notes

- Use `pathlib` throughout for all file path handling
- No OOP; use a functional structure with a `main()` entry point
- No inline comments; comments, if any, should be on their own lines only
- No logging framework; use `print` only if needed and keep it minimal
- Keep phantom NPZ support optional and limited to `phantom` mode
- Do not introduce a different phantom ROI convention than the established `fMRIQA` cache workflow without explicitly documenting the reason
