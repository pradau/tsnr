**Spec for Cursor: Add BET + T1-registration brain masking as default for `brain` mode**

### Goal
Extend the existing `brain` mode in `tsnr.py` so that **BET on the T1 anatomical image** (followed by registration of the brain mask to functional space) becomes the **default** masking method.

The current simple mean-volume thresholding + erosion method must remain as the **fallback** only (used when no T1 can be found or when any step of the BET pipeline fails).

All existing CLI behavior, outputs (`_tsnr_map.nii.gz`, `_tsnr_stats.json`), JSON contract, validation rules, and `--full-fov-maps` option must stay **exactly** the same as described in the attached `README.md`.

### Assumptions (BIDS layout)
- The input to the script is still the 4D fMRI file (positional argument or `--input`).
- The parent directory of the input file contains two BIDS-style subfolders:
  - `anat/` (contains the T1-weighted image)
  - `func/` (contains the input fMRI)
- There is exactly one T1 file in `anat/`. We will use the first file matching `*.nii*` (`.nii` or `.nii.gz`).
- FSL is available at `$FSLDIR` (with fallback to `/Users/pradau/fsl` if the environment variable is not set).

### Required changes (strictly limited to this plan)

1. **Add two small helper functions** (place them near the top of `tsnr.py`):
   - `find_t1_in_anat(input_dir: Path) -> Path | None`  
     Returns the first `*.nii*` file found in `input_dir / "anat"`, or `None` if the folder is missing or empty.
   - `create_bet_mask_for_func(t1_path: Path, func_path: Path, output_dir: Path) -> Path`  
     Implements the exact pipeline we discussed:
     - BET on T1 (`bet t1 t1_brain -m -f 0.35 -R`)
     - Compute mean functional volume
     - Quick BET on mean functional (for registration reference)
     - FLIRT registration (rigid, dof=6) of mean-func-brain → T1-brain
     - Convert transform (inverse) and apply to T1 mask → functional space
     - Return path to the final `func_brain_mask.nii.gz` (saved in `output_dir`)

     Use the `setup_fsl_env()` / subprocess pattern (sourcing `fsl.sh` only inside the function) so it works with the user's regular Python 3.12.7.

2. **Modify the `brain` mode processing logic** (inside the main function that handles mode=="brain"):
   - After loading the 4D fMRI and computing the mean volume:
     - Call `find_t1_in_anat(input_dir)`
     - If a T1 is found:
       - Try `create_bet_mask_for_func(...)`
       - If it succeeds → use the returned mask as `brain_mask`
     - If **no T1 is found** OR **any exception occurs** in the BET pipeline:
       - Print a single clear warning to stderr:
         ```
         WARNING: No T1 found in anat/ (or BET pipeline failed) — falling back to intensity thresholding
         ```
       - Fall back to the **existing** mean-threshold + erosion code (using the current `--threshold` and `--erosion-voxels` values).
   - The rest of the code (mask application, tSNR map creation, stats JSON, NaN censoring, etc.) remains **completely unchanged**.

3. **No other changes**
   - Do **not** add any new CLI flags, arguments, or options.
   - Do **not** change phantom mode at all.
   - Do **not** modify output file names, JSON fields, or validation behavior.
   - Do **not** add extra features (no neck cleanup, no bias correction, no epi_reg, etc.).
   - Update the docstring / comments in the brain-mode section to note that BET is now default.

### Implementation notes for Cursor
- Keep the code style identical to the existing codebase (same imports, Path usage, error handling, etc.).
- Make the new BET function self-contained and reusable.
- Handle FSL environment sourcing exactly as we discussed earlier so the user never has to `source fsl.sh` manually.
- Ensure the fallback path is identical to the current implementation (no behavior change when T1 is unavailable).

### Files to edit
- Only `tsnr.py`

Once implemented, the `brain` mode will automatically prefer the more accurate T1-derived mask while preserving the exact same user experience and fallback safety net.