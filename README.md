# tSNR Calculator

Compute raw temporal signal-to-noise ratio (tSNR) from fMRI data with two modes:

- `phantom`: fixed square ROI summary on one slice
- `brain`: whole-brain masked summary from a 4D NIfTI

Behavior and the stats JSON contract are documented below. Optional FSL-based T1 brain masking for `brain` mode is described under **Brain mode**.

## Quick Start

```bash
uv sync --group dev
uv run tsnr.py /path/to/run.nii.gz phantom
uv run tsnr.py /path/to/run.nii.gz brain --threshold 0.25 --erosion-voxels 2
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
```

Useful options:

- `--first-timepoint` / `--last-timepoint`: 0-based volume range (default skips the first volume with `first-timepoint=1`, similar to legacy `2..-` on the time axis).
- `--write-tmean-tstd`: also write temporal mean and standard deviation maps (`*_Tmean.nii.gz`, `*_Tstd.nii.gz`).
- `--full-fov-maps`: do not NaN voxels outside the ROI in written maps.

## Inputs

- NIfTI: `.nii` or `.nii.gz` (must be 4D, at least 2 time points)
- NPZ: fMRIQA-style phantom cache (`.npz`) is supported only for `phantom` mode

## Outputs

Outputs go to `--output-dir`, or to the input file’s directory if omitted:

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
- Summary: `tsnr_mean`, `tsnr_median`, `tsnr_std`, `tsnr_min`, `tsnr_max`, `n_voxels_in_roi`
- `map_affine_source`, `output_map_censoring`
- Mode-specific `parameters` (see below)

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

Coverage includes phantom and brain paths, T1 discovery and ordering, brain masking JSON, intensity-only parameters in stats, BET fallback when FSL steps fail, timepoint selection, and output naming.
