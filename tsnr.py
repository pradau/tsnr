# Author: Perry Radau
# Date: 2026-04-16
# Compute raw tSNR maps and summary statistics for phantom and brain fMRI inputs.
# Dependencies: Python 3.10+, nibabel, numpy, scipy; optional FSL (brain T1 mask)
# Usage: uv run tsnr.py /path/to/input.nii.gz phantom
#        uv run tsnr.py /path/to/func brain
#        uv run tsnr.py /path/to/input.nii.gz brain --threshold 0.25 --erosion-voxels 2
#        uv run tsnr.py /path/to/cache.npz phantom --roi-size 15
#        uv run tsnr.py /path/to/input.nii.gz phantom --write-tmean-tstd
#        uv run tsnr.py /path/to/input.nii.gz phantom --first-timepoint 0
#        (default --first-timepoint is 2: drop first two volumes)

"""
tSNR calculation helpers and CLI orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass, label

# Absolute robust-z cutoff for TR spike counts (ROI and per-slice ROI-mean series).
ROBUST_Z_SPIKE_ABS_THRESHOLD: float = 4.0


def fsl_dir() -> Path:
    """Return FSL installation root from ``FSLDIR`` or a default path.
    Returns:
        Path: Directory containing FSL (must include ``etc/fslconf/fsl.sh``).
    """
    return Path(os.environ.get("FSLDIR", "/Users/pradau/fsl"))


def _bids_json_sidecar_for_nifti(nifti_path: Path) -> Path:
    """Return the BIDS companion ``.json`` path for a NIfTI file.
    Args:
        nifti_path (Path): Path ending in ``.nii`` or ``.nii.gz``.
    Returns:
        Path: Sidecar path with the same basename stem (BIDS convention).
    """
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[: -len(".nii.gz")] + ".json")
    if name.endswith(".nii"):
        return nifti_path.with_name(name[: -len(".nii")] + ".json")
    return nifti_path.with_suffix(".json")


def _parse_bids_date_string(value: object) -> Optional[date]:
    """Parse BIDS ``AcquisitionDate`` (YYYY-MM-DD or YYYYMMDD).
    Args:
        value (object): Raw JSON value.
    Returns:
        Optional[date]: Parsed calendar date, or ``None``.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if len(s) == 8 and s.isdigit():
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _parse_bids_time_string(value: object) -> Optional[time]:
    """Parse BIDS ``AcquisitionTime`` (e.g. ``HH:MM:SS`` or ``H:M:S.fraction``).
    Args:
        value (object): Raw JSON value.
    Returns:
        Optional[time]: Parsed clock time, or ``None``.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.isdigit() and len(s) == 6:
        return time(int(s[:2]), int(s[2:4]), int(s[4:6]))
    parts = s.split(":")
    if len(parts) < 3:
        return None
    h = int(parts[0])
    m = int(parts[1])
    sec_rest = parts[2]
    if "." in sec_rest:
        sec_str, frac = sec_rest.split(".", 1)
        sec = int(sec_str)
        micro = int((float("0." + frac) * 1_000_000))
    else:
        sec = int(sec_rest)
        micro = 0
    return time(h, m, sec, micro)


def _parse_acquisition_datetime_from_sidecar(json_path: Path) -> Optional[datetime]:
    """Best-effort acquisition instant from a BIDS sidecar JSON.
    Uses ``AcquisitionDateTime`` when present; otherwise combines
    ``AcquisitionDate`` and ``AcquisitionTime`` when both parse.
    Args:
        json_path (Path): Path to a ``.json`` sidecar.
    Returns:
        Optional[datetime]: Timezone-naive UTC-ish local acquisition time, or
        ``None`` if nothing usable was found.
    """
    if not json_path.is_file():
        return None
    try:
        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    adt = data.get("AcquisitionDateTime")
    if isinstance(adt, str) and adt.strip():
        s = adt.strip()
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            parsed = datetime.fromisoformat(s)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            pass

    d = _parse_bids_date_string(data.get("AcquisitionDate"))
    t = _parse_bids_time_string(data.get("AcquisitionTime"))
    if d is not None and t is not None:
        return datetime.combine(d, t)
    return None


def _t1w_sort_key(nifti_path: Path) -> Tuple[int, float, str]:
    """Sort key: earliest acquisition first; JSON-based times beat file mtime.
    Args:
        nifti_path (Path): Candidate ``*T1w.nii`` file under ``anat/``.
    Returns:
        Tuple[int, float, str]: ``(0, timestamp, name)`` when a sidecar time was
        parsed, else ``(1, mtime, name)`` for mtime fallback.
    """
    jp = _bids_json_sidecar_for_nifti(nifti_path)
    dt = _parse_acquisition_datetime_from_sidecar(jp)
    if dt is not None:
        return (0, dt.timestamp(), nifti_path.name)
    try:
        mtime = float(nifti_path.stat().st_mtime)
    except OSError:
        mtime = float("inf")
    return (1, mtime, nifti_path.name)


def _list_t1w_niftis_in_anat(anat_dir: Path) -> List[Path]:
    """Collect ``*T1w.nii`` and ``*T1w.nii.gz`` files (no duplicates).
    Args:
        anat_dir (Path): BIDS ``anat`` directory.
    Returns:
        List[Path]: Sorted list of T1w NIfTI paths (deterministic order before
        acquisition-time sort).
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for pattern in ("*T1w.nii.gz", "*T1w.nii"):
        for p in anat_dir.glob(pattern):
            if not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(p)
    return out


def find_t1_in_anat(func_input_path: Path) -> Optional[Path]:
    """Locate a T1-weighted NIfTI under BIDS-style ``anat/`` next to ``func/``.
    Tries, in order:
    1. ``func_input_path.parent.parent / "anat"`` when the bold file lives under
       ``.../<subject|session>/func/``.
    2. ``func_input_path.parent / "anat"`` for flat or custom layouts.
    In each existing ``anat`` directory, considers only files matching BIDS-style
    ``*T1w.nii.gz`` and ``*T1w.nii``. If several exist, chooses the one with the
    earliest acquisition timestamp from the matching ``.json`` sidecar
    (``AcquisitionDateTime``, or ``AcquisitionDate`` + ``AcquisitionTime``). If
    no sidecar time is available for a candidate, its file modification time is
    used so the earliest file still wins among those without JSON times.
    Args:
        func_input_path (Path): Path to the 4D functional NIfTI.
    Returns:
        Optional[Path]: Selected T1w NIfTI, or ``None`` if none match.
    """
    candidates_dirs = (func_input_path.parent.parent / "anat", func_input_path.parent / "anat")
    for anat_dir in candidates_dirs:
        if not anat_dir.is_dir():
            continue
        t1w_list = _list_t1w_niftis_in_anat(anat_dir)
        if not t1w_list:
            continue
        t1w_list.sort(key=_t1w_sort_key)
        return t1w_list[0]
    return None


def list_bold_niftis_in_dir(directory: Path, pattern: Optional[str] = None) -> List[Path]:
    """List BIDS-style BOLD NIfTI files in a directory (non-recursive).
    By default matches ``*_bold.nii.gz`` and ``*_bold.nii`` so typical ``sbref`` and
    non-bold files are excluded. Pass ``pattern`` to use a single custom glob instead
    (for example when filenames omit ``_bold``).
    Args:
        directory (Path): Directory to scan (must exist and be a directory).
        pattern (Optional[str]): If set, only this glob pattern is used relative to
            ``directory``. If ``None``, the default BOLD patterns apply.
    Returns:
        List[Path]: Sorted, deduplicated file paths (lexicographic by basename).
    Raises:
        ValueError: If ``directory`` is not an existing directory.
    """
    if not directory.is_dir():
        raise ValueError(f"Not a directory or directory does not exist: {directory}")
    seen: set[Path] = set()
    out: List[Path] = []
    patterns: Tuple[str, ...] = (pattern,) if pattern is not None else ("*_bold.nii.gz", "*_bold.nii")
    for pat in patterns:
        for p in directory.glob(pat):
            if not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(p)
    out.sort(key=lambda x: x.name)
    return out


def _format_brain_pipeline_error(exc: BaseException) -> str:
    """Produce a concise message for JSON when the T1-to-functional pipeline fails.
    Args:
        exc (BaseException): Exception raised during BET, FLIRT, or mask loading.
    Returns:
        str: Truncated, stderr-aware description for ``brain_masking.detail``.
    """
    if isinstance(exc, subprocess.CalledProcessError):
        parts = [f"FSL command failed with exit code {exc.returncode}"]
        if exc.stderr and exc.stderr.strip():
            parts.append(exc.stderr.strip())
        elif exc.stdout and exc.stdout.strip():
            parts.append(exc.stdout.strip())
        return "\n".join(parts)[:4000]
    return str(exc)[:4000]


def _brain_masking_success(t1_path: Path) -> Dict[str, Any]:
    """
    Build ``brain_masking`` JSON when the T1 BET plus FLIRT mask is used.
    """
    return {
        "method": "t1_bet_registered_to_mean_epi",
        "t1_path": str(t1_path.resolve()),
        "t1_to_functional_pipeline": "success",
        "detail": None,
    }


def _brain_masking_fallback_no_t1() -> Dict[str, Any]:
    """
    Build ``brain_masking`` JSON when no T1 is found under ``anat/``.
    """
    return {
        "method": "centroid_seeded",
        "t1_path": None,
        "t1_to_functional_pipeline": "not_attempted_no_t1",
        "detail": (
            "No T1-weighted NIfTI matching *T1w.nii.gz or *T1w.nii found under anat/ "
            "(checked next to the functional path and one level up for BIDS-style "
            "func/ layout)."
        ),
    }


def _brain_masking_fallback_pipeline_failed(t1_path: Optional[Path], exc: BaseException) -> Dict[str, Any]:
    """
    Build ``brain_masking`` JSON when BET or registration fails.
    """
    return {
        "method": "centroid_seeded",
        "t1_path": str(t1_path.resolve()) if t1_path is not None else None,
        "t1_to_functional_pipeline": "failed",
        "detail": _format_brain_pipeline_error(exc),
    }


def _brain_masking_fallback_no_nifti_header() -> Dict[str, Any]:
    """
    Build ``brain_masking`` JSON for defensive brain path without spatial NIfTI.
    """
    return {
        "method": "centroid_seeded",
        "t1_path": None,
        "t1_to_functional_pipeline": "not_attempted_no_nifti",
        "detail": "Spatial NIfTI header is required for brain masking.",
    }


def _run_fsl_bash(fsl_root: Path, command: str) -> None:
    """Run a shell command after sourcing FSL configuration.
    Args:
        fsl_root (Path): FSL installation root.
        command (str): Shell command(s) to run after ``fsl.sh`` is sourced.
    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
        FileNotFoundError: If ``fsl.sh`` is missing.
    """
    fsl_sh = fsl_root / "etc" / "fslconf" / "fsl.sh"
    if not fsl_sh.is_file():
        raise FileNotFoundError(f"FSL config not found: {fsl_sh}")
    inner = f'source "{fsl_sh}" && set -euo pipefail && {command}'
    subprocess.run(
        ["bash", "-lc", inner],
        check=True,
        capture_output=True,
        text=True,
    )


def create_bet_mask_for_func(
    t1_path: Path,
    mean_volume: np.ndarray,
    source_nifti: nib.Nifti1Image,
    work_dir: Path,
) -> Path:
    """Build a brain mask in functional space via BET on T1, FLIRT, and inverse warp.
    Pipeline: BET on T1 (``-m -f 0.35 -R``); save temporal mean EPI; quick BET on
    mean EPI; rigid FLIRT (dof=6) from mean-EPI-brain to T1-brain; invert transform;
    apply T1 brain mask to full mean EPI grid with nearest-neighbor interpolation.
    Args:
        t1_path (Path): T1-weighted NIfTI path.
        mean_volume (np.ndarray): 3D temporal mean (same space as ``source_nifti``).
        source_nifti (nib.Nifti1Image): Functional image header/affine for the mean.
        work_dir (Path): Directory for intermediate files (created if missing).
    Returns:
        Path: Path to ``func_brain_mask.nii.gz`` in ``work_dir``.
    Raises:
        ValueError: If output mask shape does not match ``mean_volume`` or mask is empty.
        subprocess.CalledProcessError: If an FSL tool fails.
        FileNotFoundError: If FSL is not installed at ``fsl_dir()``.
    """
    if mean_volume.shape != tuple(source_nifti.shape[:3]):
        raise ValueError("mean_volume shape must match functional NIfTI spatial shape")
    work_dir.mkdir(parents=True, exist_ok=True)
    fsl_root = fsl_dir()
    t1_abs = t1_path.resolve()
    stem_t1 = work_dir / "t1_bet"
    mean_vol_path = work_dir / "func_mean.nii.gz"
    stem_mean = work_dir / "mean_epi_bet"
    func2t1 = work_dir / "func2t1.mat"
    t12func = work_dir / "t12func.mat"
    mask_out = work_dir / "func_brain_mask.nii.gz"

    mean_img = nib.Nifti1Image(mean_volume.astype(np.float32), source_nifti.affine, source_nifti.header)
    nib.save(mean_img, str(mean_vol_path))

    _run_fsl_bash(fsl_root, f'bet "{t1_abs}" "{stem_t1}" -m -f 0.35 -R')
    _run_fsl_bash(fsl_root, f'bet "{mean_vol_path}" "{stem_mean}" -m')
    # FSL ``bet`` writes ``<stem>.nii.gz`` (brain) and ``<stem>_mask.nii.gz``, not
    # ``<stem>_brain.nii.gz`` (legacy naming from very old FSL versions).
    t1_brain = stem_t1.with_suffix(".nii.gz")
    mean_brain = stem_mean.with_suffix(".nii.gz")
    t1_mask = stem_t1.with_name(f"{stem_t1.name}_mask.nii.gz")
    if not t1_brain.is_file() or not mean_brain.is_file() or not t1_mask.is_file():
        raise ValueError("BET did not produce expected outputs")

    _run_fsl_bash(
        fsl_root,
        f'flirt -in "{mean_brain}" -ref "{t1_brain}" -omat "{func2t1}" -dof 6',
    )
    _run_fsl_bash(
        fsl_root,
        f'convert_xfm -omat "{t12func}" -inverse "{func2t1}"',
    )
    _run_fsl_bash(
        fsl_root,
        f'flirt -in "{t1_mask}" -ref "{mean_vol_path}" -applyxfm -init "{t12func}" '
        f'-out "{mask_out}" -interp nearestneighbour',
    )

    loaded = nib.load(str(mask_out))
    data = np.asarray(loaded.get_fdata(dtype=np.float64))
    if data.shape != mean_volume.shape:
        raise ValueError("Registered brain mask shape does not match mean volume")
    if not np.any(data > 0.5):
        raise ValueError("Registered brain mask is empty")
    return mask_out


def derive_basename(input_path: Path) -> str:
    """Derive output basename from input filename.
    Args:
        input_path (Path): Input file path.
    Returns:
        str: Basename with `.nii.gz` stripped as a unit when present.
    """
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return input_path.stem


def derive_qa_session_date(input_path: Path) -> Optional[str]:
    """Extract ``YYYY-MM-DD`` from an input basename for longitudinal phantom QA.
    Args:
        input_path (Path): Input NIfTI/NPZ path.
    Returns:
        Optional[str]: Normalized date string, or ``None`` when not present.
    """
    match = re.search(r"(20\d{2})[_-](\d{2})[_-](\d{2})", input_path.name)
    if match is None:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def default_output_dir_for_input(input_path: Path) -> Path:
    """Return default output directory for an input NIfTI/NPZ path.
    If the input lives under a BIDS-style ``func`` folder, use sibling
    ``derivatives/tsnr`` under that parent (for example ``ses-1a/derivatives/tsnr``).
    Otherwise, use ``<input_parent>/derivatives/tsnr``.
    Args:
        input_path (Path): Input file path.
    Returns:
        Path: Default destination directory for outputs.
    """
    parent = input_path.parent
    if parent.name == "func":
        return parent.parent / "derivatives" / "tsnr"
    return parent / "derivatives" / "tsnr"


def apply_timepoint_selection(
    data_4d: np.ndarray,
    first_index: int,
    last_index_inclusive: Optional[int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Slice the time axis to match legacy-style volume ranges (for example `2..-`).
    Default first index 2 (via CLI) drops the first two volumes to reduce non-steady-state
    transient effects in fMRI signal.
    Args:
        data_4d (np.ndarray): Full `(x, y, z, t)` volume.
        first_index (int): 0-based index of the first timepoint to include.
        last_index_inclusive (Optional[int]): 0-based index of the last timepoint to
            include, or None to use the final volume in the file.
    Returns:
        Tuple[np.ndarray, Dict[str, int]]: Sliced array and selection metadata
            (`first_index`, `last_index_inclusive`, `n_timepoints_in_file`,
            `n_timepoints_used`).
    Raises:
        ValueError: If indices are out of range or fewer than two timepoints remain.
    """
    n_t = int(data_4d.shape[3])
    if first_index < 0:
        raise ValueError("--first-timepoint must be >= 0")
    if first_index >= n_t:
        raise ValueError(
            f"--first-timepoint {first_index} is out of range for {n_t} volume(s) on file"
        )
    if last_index_inclusive is None:
        last = n_t - 1
    else:
        last = last_index_inclusive
    if last < 0 or last >= n_t:
        raise ValueError(
            f"--last-timepoint {last} is out of range for {n_t} volume(s) on file"
        )
    if last < first_index:
        raise ValueError("--last-timepoint must be >= --first-timepoint")
    sliced = data_4d[:, :, :, first_index : last + 1]
    n_used = int(sliced.shape[3])
    if n_used < 2:
        raise ValueError(
            "After timepoint selection, at least 2 volumes are required for tSNR "
            f"(got {n_used}); widen --first-timepoint/--last-timepoint or use more "
            "volumes in the input"
        )
    meta = {
        "first_index": int(first_index),
        "last_index_inclusive": int(last),
        "n_timepoints_in_file": n_t,
        "n_timepoints_used": n_used,
    }
    return sliced, meta


def validate_common_args(
    mode: str,
    threshold: float,
    erosion_voxels: int,
    phantom_roi_mode: str,
    phantom_edge_erosion_voxels: int,
    phantom_full_threshold_fraction: float,
) -> None:
    """Validate common CLI arguments.
    Args:
        mode (str): Analysis mode.
        threshold (float): Brain threshold fraction.
        erosion_voxels (int): Brain erosion iterations.
        phantom_roi_mode (str): Phantom ROI mode.
        phantom_edge_erosion_voxels (int): Phantom edge-erosion iterations.
        phantom_full_threshold_fraction (float): Phantom full-mask threshold fraction.
    Raises:
        ValueError: If any value is invalid.
    """
    if mode not in ("phantom", "brain"):
        raise ValueError(f"Invalid mode: {mode}")
    if not (0.0 < threshold < 1.0):
        raise ValueError("--threshold must be strictly between 0 and 1")
    if erosion_voxels < 0:
        raise ValueError("--erosion-voxels must be >= 0")
    if phantom_roi_mode not in ("patch", "full_minus_edges"):
        raise ValueError("--phantom-roi-mode must be one of: patch, full_minus_edges")
    if phantom_edge_erosion_voxels < 0:
        raise ValueError("--phantom-edge-erosion-voxels must be >= 0")
    if not (0.0 < phantom_full_threshold_fraction < 1.0):
        raise ValueError(
            "--phantom-full-threshold-fraction must be strictly between 0 and 1"
        )


def load_nifti_4d(input_path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load a 4D NIfTI into float64.
    Args:
        input_path (Path): NIfTI path.
    Returns:
        Tuple[np.ndarray, nib.Nifti1Image]: Data array `(x, y, z, t)` and image object.
    Raises:
        ValueError: If dimensionality or content is invalid.
    """
    image = nib.load(str(input_path))
    data = np.asarray(image.get_fdata(dtype=np.float64), dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI input, got shape {data.shape}")
    if data.shape[3] < 2:
        raise ValueError(f"Need at least 2 time points, got {data.shape[3]}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Input contains non-finite values")
    return data, image


def load_phantom_npz_4d(input_path: Path) -> np.ndarray:
    """Load fMRIQA-style NPZ cache into `(x, y, z, t)` layout.
    Args:
        input_path (Path): NPZ cache path.
    Returns:
        np.ndarray: Array with shape `(x, y, z, t)` in float64.
    Raises:
        ValueError: If cache version or volume shape is invalid.
    """
    with np.load(str(input_path), allow_pickle=True) as data:
        if "cache_version" not in data:
            raise ValueError("Unsupported pixel cache: missing cache_version")
        version = int(np.asarray(data["cache_version"]).item())
        if version not in (1, 2):
            raise ValueError(f"Unsupported pixel cache version {version}")
        if "volume" not in data:
            raise ValueError("Unsupported pixel cache: missing volume")
        volume = np.asarray(data["volume"], dtype=np.float64)
    return fmriqa_volume_to_data_4d(volume)


def fmriqa_volume_to_data_4d(volume: np.ndarray) -> np.ndarray:
    """Convert fMRIQA pixel cache volume to ``(x, y, z, t)`` layout used by this module.

    Args:
        volume (np.ndarray): Array shaped ``(n_slices, n_times, n_rows, n_cols)`` (fMRIQA cache).

    Returns:
        np.ndarray: Array shaped ``(x, y, z, t)`` in float64 (same convention as ``load_phantom_npz_4d``).

    Raises:
        ValueError: If shape is invalid, fewer than two time points, or values are non-finite.
    """
    if volume.ndim != 4:
        raise ValueError(
            f"Expected NPZ volume shape (slices, time, rows, cols), got {volume.shape}"
        )
    if volume.shape[1] < 2:
        raise ValueError(f"Need at least 2 time points, got {volume.shape[1]}")
    if not np.all(np.isfinite(volume)):
        raise ValueError("Input contains non-finite values")
    return np.transpose(volume, (2, 3, 0, 1))


def run_phantom_analysis_from_4d(
    data_4d: np.ndarray,
    *,
    roi_size: int = 15,
    slice_index: Optional[int] = None,
    phantom_roi_mode: str = "patch",
    phantom_edge_erosion_voxels: int = 1,
    phantom_full_threshold_fraction: float = 0.35,
    first_timepoint: int = 2,
    last_timepoint: Optional[int] = None,
) -> Dict[str, Any]:
    """Phantom-mode tSNR map, ROI extraction, and summary statistics from 4D data.

    Applies the same timepoint selection, voxel-wise tSNR map, phantom ROI placement,
    and ``summarize`` logic as :func:`run_analysis` in ``phantom`` mode.

    Args:
        data_4d (np.ndarray): Full ``(x, y, z, t)`` volume before timepoint selection.
        roi_size (int): Odd positive phantom ROI edge length in pixels.
        slice_index (Optional[int]): Z slice index, or ``None`` for middle slice.
        phantom_roi_mode (str): ``patch`` or ``full_minus_edges``.
        phantom_edge_erosion_voxels (int): 2D per-slice erosion iterations for
            ``full_minus_edges``.
        phantom_full_threshold_fraction (float): Intensity threshold as a fraction
            of local reference for ``full_minus_edges``.
        first_timepoint (int): 0-based first time index to include (default ``2`` drops first two volumes).
        last_timepoint (Optional[int]): 0-based last time index inclusive, or ``None`` for end.

    Returns:
        Dict[str, Any]: Keys include ``data_4d`` (after time trim), ``mean_volume``, ``std_volume``,
        ``tsnr_map``, ``selected_values``, ``parameters``, ``timepoint_selection``, ``summary``
        (spatial tSNR plus ``ftsnr`` and ``roi_mean_signal_std`` from the ROI mean time
        course),
        ``n_voxels_in_roi``, ``n_timepoints``, ``volume_shape`` (list of int).

    Raises:
        ValueError: If ``data_4d`` is invalid or analysis fails.
    """
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D array (x, y, z, t), got shape {data_4d.shape}")
    if data_4d.shape[3] < 2:
        raise ValueError(f"Need at least 2 time points, got {data_4d.shape[3]}")
    if not np.all(np.isfinite(data_4d)):
        raise ValueError("Input contains non-finite values")
    data_4d, timepoint_selection = apply_timepoint_selection(
        data_4d,
        first_index=first_timepoint,
        last_index_inclusive=last_timepoint,
    )
    mean_volume, std_volume, tsnr_map = compute_tsnr_map(data_4d)
    selected_values, parameters = extract_phantom_tsnr(
        tsnr_map,
        mean_volume,
        roi_size,
        slice_index,
        phantom_roi_mode=phantom_roi_mode,
        phantom_edge_erosion_voxels=phantom_edge_erosion_voxels,
        phantom_full_threshold_fraction=phantom_full_threshold_fraction,
    )
    roi_mask = build_phantom_analysis_mask(data_4d.shape[:3], mean_volume, parameters)
    summary = {**summarize(selected_values), **compute_ftsnr_metrics(data_4d, roi_mask)}
    return {
        "data_4d": data_4d,
        "mean_volume": mean_volume,
        "std_volume": std_volume,
        "tsnr_map": tsnr_map,
        "selected_values": selected_values,
        "parameters": parameters,
        "timepoint_selection": timepoint_selection,
        "summary": summary,
        "n_voxels_in_roi": int(selected_values.size),
        "n_timepoints": int(data_4d.shape[3]),
        "volume_shape": [int(x) for x in data_4d.shape],
    }


def compute_tsnr_map(data_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voxelwise tSNR map.
    Args:
        data_4d (np.ndarray): Input shape `(x, y, z, t)`.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: `(mean_volume, std_volume, tsnr_map)`.
    """
    mean_volume = np.mean(data_4d, axis=3)
    std_volume = np.std(data_4d, axis=3, ddof=1)
    tsnr_map = np.zeros_like(mean_volume, dtype=np.float64)
    valid = std_volume > 0.0
    tsnr_map[valid] = mean_volume[valid] / std_volume[valid]
    tsnr_map[~np.isfinite(tsnr_map)] = 0.0
    return mean_volume, std_volume, tsnr_map


def place_phantom_roi(
    reference_slice: np.ndarray,
    roi_size: int,
) -> Tuple[int, int, int, int]:
    """Place centered square ROI with in-bounds shifting.
    Args:
        reference_slice (np.ndarray): 2D mean image.
        roi_size (int): Odd positive side length.
    Returns:
        Tuple[int, int, int, int]: `(row_start, row_end, col_start, col_end)`.
    Raises:
        ValueError: If ROI cannot be placed.
    """
    if roi_size < 1 or roi_size % 2 == 0:
        raise ValueError("--roi-size must be a positive odd integer")
    rows, cols = reference_slice.shape
    if roi_size > rows or roi_size > cols:
        raise ValueError(f"roi-size {roi_size} exceeds image dimensions {rows}x{cols}")
    weights = np.maximum(reference_slice.astype(np.float64, copy=False), 0.0)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("Cannot place phantom ROI: selected slice has non-positive intensities")
    com_row, com_col = center_of_mass(weights)
    half = roi_size // 2
    center_row = int(np.clip(round(float(com_row)), half, rows - 1 - half))
    center_col = int(np.clip(round(float(com_col)), half, cols - 1 - half))
    row_start = center_row - half
    row_end = center_row + half + 1
    col_start = center_col - half
    col_end = center_col + half + 1
    return row_start, row_end, col_start, col_end


def extract_phantom_tsnr(
    tsnr_map: np.ndarray,
    mean_volume: np.ndarray,
    roi_size: int,
    slice_index: Optional[int],
    phantom_roi_mode: str = "patch",
    phantom_edge_erosion_voxels: int = 1,
    phantom_full_threshold_fraction: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract phantom ROI values from selected slice.
    Args:
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D mean volume.
        roi_size (int): ROI side length for ``patch`` mode.
        slice_index (Optional[int]): Selected slice or None for middle.
        phantom_roi_mode (str): ``patch`` or ``full_minus_edges``.
        phantom_edge_erosion_voxels (int): 2D per-slice erosion iterations used
            only by ``full_minus_edges``.
        phantom_full_threshold_fraction (float): Intensity threshold fraction for
            ``full_minus_edges`` (vs local reference).
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: ROI values and parameter metadata.
    Raises:
        ValueError: If slice index or ROI extraction is invalid.
    """
    if phantom_roi_mode == "full_minus_edges":
        threshold_fraction = float(phantom_full_threshold_fraction)
        mask = build_phantom_full_mask(
            mean_volume,
            phantom_edge_erosion_voxels,
            threshold_fraction=threshold_fraction,
        )
        values = tsnr_map[mask]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError("phantom full-minus-edges ROI contains zero valid voxels")
        params = {
            "phantom_roi_mode": "full_minus_edges",
            "full_phantom_mask": {
                "source": "centroid_seeded_region_grow_3d_above_fraction_of_local_centroid_percentile",
                "reference_radius_voxels": 15,
                "reference_percentile": 90.0,
                "threshold_fraction": float(threshold_fraction),
                "edge_erosion_voxels": int(phantom_edge_erosion_voxels),
                "erosion_axis": "xy_per_slice",
            },
        }
        return finite_values, params

    z_dim = tsnr_map.shape[2]
    target_slice = z_dim // 2 if slice_index is None else slice_index
    if not (0 <= target_slice < z_dim):
        raise ValueError(f"--slice-index is outside valid range [0, {z_dim - 1}]")
    ref_slice = mean_volume[:, :, target_slice]
    row_start, row_end, col_start, col_end = place_phantom_roi(ref_slice, roi_size)
    values = tsnr_map[row_start:row_end, col_start:col_end, target_slice]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("phantom ROI contains zero valid voxels")
    params = {
        "phantom_roi_mode": "patch",
        "roi_size": roi_size,
        "slice_index": int(target_slice),
        "roi_bounds": [int(row_start), int(row_end), int(col_start), int(col_end)],
    }
    return finite_values, params


def build_brain_mask(
    mean_volume: np.ndarray,
    threshold: float,
    erosion_voxels: int,
) -> np.ndarray:
    """Build the same boolean brain mask used for ROI statistics.
    Args:
        mean_volume (np.ndarray): 3D temporal mean volume.
        threshold (float): Relative threshold fraction of local reference intensity.
        erosion_voxels (int): Binary erosion iterations.
    Returns:
        np.ndarray: Boolean array shaped like ``mean_volume``.
    Raises:
        ValueError: If there are no positive voxels to define a baseline.
    """
    seed_xyz, reference_intensity = local_centroid_percentile_reference(
        mean_volume,
        radius_voxels=15,
        percentile=90.0,
        label_prefix="brain",
    )
    mask = _mask_from_centroid_seeded_threshold(
        mean_volume,
        float(threshold) * float(reference_intensity),
        seed_xyz,
        label_prefix="brain",
    )
    if erosion_voxels > 0:
        mask = binary_erosion(mask, iterations=erosion_voxels)
    if np.any(mask):
        mask = largest_connected_component(mask, label_prefix="brain")
    return mask


def _mask_from_centroid_seeded_threshold(
    mean_volume: np.ndarray,
    threshold_value: float,
    seed_xyz: Tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    """Keep the 3D connected component that contains the seed after thresholding.
    Args:
        mean_volume (np.ndarray): 3D temporal mean volume.
        threshold_value (float): Absolute intensity cutoff.
        seed_xyz (Tuple[int, int, int]): Voxel to seed the component (from centroid
            or relocated if needed).
        label_prefix (str): Prefix for error messages.
    Returns:
        np.ndarray: Boolean mask for one 6-connected component.
    Raises:
        ValueError: If thresholding or component selection fails.
    """
    candidate = mean_volume >= float(threshold_value)
    if not np.any(candidate):
        raise ValueError(f"{label_prefix} mask is empty after thresholding")
    seed = seed_xyz
    if not bool(candidate[seed]):
        flat_idx = int(np.argmax(mean_volume * candidate))
        seed = tuple(int(x) for x in np.unravel_index(flat_idx, mean_volume.shape))
    labeled, n_labels = label(candidate)
    if n_labels < 1:
        raise ValueError(f"{label_prefix} region-growing failed to find connected components")
    seed_label = int(labeled[seed])
    if seed_label <= 0:
        raise ValueError(f"{label_prefix} seed is not inside thresholded candidate mask")
    return labeled == seed_label


def local_centroid_percentile_reference(
    mean_volume: np.ndarray,
    radius_voxels: int,
    percentile: float,
    label_prefix: str,
) -> Tuple[Tuple[int, int, int], float]:
    """Robust local reference intensity from a centroid neighborhood.
    Args:
        mean_volume (np.ndarray): 3D temporal mean volume.
        radius_voxels (int): Radius of 3D neighborhood around centroid.
        percentile (float): Percentile of local positive finite samples.
        label_prefix (str): Prefix for error messages.
    Returns:
        Tuple[Tuple[int, int, int], float]: Seed voxel and reference intensity.
    Raises:
        ValueError: If no positive signal or no valid local samples exist.
    """
    positive = mean_volume > 0.0
    if not np.any(positive):
        raise ValueError(f"{label_prefix} mask baseline could not be computed: no positive voxels")
    com = center_of_mass(np.maximum(mean_volume, 0.0))
    seed_xyz = tuple(
        int(np.clip(round(float(c)), 0, mean_volume.shape[i] - 1))
        for i, c in enumerate(com)
    )
    if radius_voxels < 1:
        radius_voxels = 1
    gx, gy, gz = np.ogrid[: mean_volume.shape[0], : mean_volume.shape[1], : mean_volume.shape[2]]
    dist2 = (gx - seed_xyz[0]) ** 2 + (gy - seed_xyz[1]) ** 2 + (gz - seed_xyz[2]) ** 2
    local = dist2 <= int(radius_voxels) ** 2
    valid = local & np.isfinite(mean_volume) & (mean_volume > 0.0)
    samples = mean_volume[valid]
    if samples.size < 1:
        raise ValueError(f"{label_prefix} local centroid neighborhood has no valid positive voxels")
    ref = float(np.percentile(samples, float(percentile)))
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError(f"{label_prefix} local centroid reference intensity is invalid")
    return seed_xyz, ref


def largest_connected_component(mask: np.ndarray, label_prefix: str) -> np.ndarray:
    """Keep only the largest connected component in a boolean mask.
    Args:
        mask (np.ndarray): Candidate boolean mask.
        label_prefix (str): Prefix for error messages.
    Returns:
        np.ndarray: Boolean mask containing only the largest component.
    Raises:
        ValueError: If the mask is empty or no component can be selected.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        raise ValueError(f"{label_prefix} mask is empty before connected-component cleanup")
    labeled, n_labels = label(mask_bool)
    if n_labels < 1:
        raise ValueError(f"{label_prefix} connected-component cleanup found no components")
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    keep_label = int(np.argmax(component_sizes))
    if keep_label <= 0:
        raise ValueError(f"{label_prefix} connected-component cleanup failed to select a component")
    return labeled == keep_label


def build_phantom_roi_mask(volume_shape: Tuple[int, int, int], params: Dict[str, Any]) -> np.ndarray:
    """Build a 3D mask that is True only on the phantom ROI slice patch.
    Args:
        volume_shape (Tuple[int, int, int]): ``(x, y, z)`` shape of 3D maps.
        params (Dict[str, Any]): Phantom ``parameters`` dict with ``roi_bounds`` and
            ``slice_index``.
    Returns:
        np.ndarray: Boolean mask shaped ``volume_shape``.
    """
    row_start, row_end, col_start, col_end = params["roi_bounds"]
    z = int(params["slice_index"])
    mask = np.zeros(volume_shape, dtype=bool)
    mask[row_start:row_end, col_start:col_end, z] = True
    return mask


def build_phantom_full_mask(
    mean_volume: np.ndarray,
    edge_erosion_voxels: int,
    threshold_fraction: float = 0.35,
) -> np.ndarray:
    """Build a 3D phantom mask from centroid-seeded region growing and 2D erosion.
    Args:
        mean_volume (np.ndarray): 3D temporal mean volume.
        edge_erosion_voxels (int): In-plane erosion iterations for each z slice.
        threshold_fraction (float): Threshold fraction of local reference intensity.
    Returns:
        np.ndarray: Boolean mask shaped like ``mean_volume`` with a single 6-connected
        cluster.
    Raises:
        ValueError: If centroid seed, grown mask, or eroded mask is empty.
    """
    seed_xyz, reference_intensity = local_centroid_percentile_reference(
        mean_volume,
        radius_voxels=15,
        percentile=90.0,
        label_prefix="phantom",
    )
    threshold_value = float(threshold_fraction) * float(reference_intensity)
    base_mask = _mask_from_centroid_seeded_threshold(
        mean_volume,
        threshold_value,
        seed_xyz,
        label_prefix="phantom",
    )
    final: np.ndarray
    if edge_erosion_voxels <= 0:
        final = base_mask
    else:
        eroded = np.zeros_like(base_mask, dtype=bool)
        for z in range(base_mask.shape[2]):
            slice_mask = base_mask[:, :, z]
            if np.any(slice_mask):
                eroded[:, :, z] = binary_erosion(slice_mask, iterations=edge_erosion_voxels)
        if not np.any(eroded):
            raise ValueError("phantom mask is empty after edge erosion")
        final = eroded
    return largest_connected_component(final, label_prefix="phantom")


def build_phantom_analysis_mask(
    volume_shape: Tuple[int, int, int],
    mean_volume: np.ndarray,
    parameters: Dict[str, Any],
) -> np.ndarray:
    """Build the phantom ROI mask for summary and map censoring.
    Args:
        volume_shape (Tuple[int, int, int]): Spatial shape of 3D maps.
        mean_volume (np.ndarray): 3D temporal mean volume.
        parameters (Dict[str, Any]): Phantom parameters block from stats.
    Returns:
        np.ndarray: Boolean mask shaped ``volume_shape``.
    """
    roi_mode = str(parameters.get("phantom_roi_mode", "patch"))
    if roi_mode == "full_minus_edges":
        mask_meta = parameters.get("full_phantom_mask", {})
        edge_erosion_voxels = int(mask_meta.get("edge_erosion_voxels", 1))
        threshold_fraction = float(mask_meta.get("threshold_fraction", 0.35))
        return build_phantom_full_mask(
            mean_volume,
            edge_erosion_voxels,
            threshold_fraction=threshold_fraction,
        )
    return build_phantom_roi_mask(volume_shape, parameters)


def extract_brain_tsnr(
    tsnr_map: np.ndarray,
    mean_volume: np.ndarray,
    threshold: float,
    erosion_voxels: int,
    brain_mask: Optional[np.ndarray] = None,
    brain_masking_report: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract brain-mask tSNR values.
    In ``brain`` mode, when ``brain_mask`` is ``None``, the mask is built from the
    temporal mean using centroid-seeded thresholding with a local percentile
    reference, then erosion and largest-component cleanup. When
    ``brain_mask`` is set (for example from T1 BET + registration), that array is
    used instead.
    JSON metadata: ``intensity_brain_mask`` (method, threshold, and erosion) is
    written only when the spatial mask was built from fallback rules. It is omitted when a
    supplied mask array defined the ROI (for example T1-based BET), so stats do not
    list unused CLI values as if they had driven masking.
    Args:
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D mean volume.
        threshold (float): Relative threshold fraction.
        erosion_voxels (int): Binary erosion iterations.
        brain_mask (Optional[np.ndarray]): If provided, boolean mask shaped like
            ``mean_volume``; otherwise centroid-seeded thresholding + erosion is used.
        brain_masking_report (Optional[Dict[str, Any]]): Optional ``brain_masking``
            object merged into ``parameters`` (method, T1 path, pipeline status).
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Brain values and parameter metadata.
    Raises:
        ValueError: If mask creation fails or is empty.
    """
    if brain_mask is None:
        mask = build_brain_mask(mean_volume, threshold, erosion_voxels)
    else:
        mask = np.asarray(brain_mask, dtype=bool)
        if mask.shape != mean_volume.shape:
            raise ValueError("brain_mask must match mean_volume shape")
    if not np.any(mask):
        raise ValueError("brain mask is empty after erosion")
    baseline = float(np.mean(mean_volume[mean_volume > 0.0]))
    values = tsnr_map[mask]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("brain mask contains zero valid voxels")
    params: Dict[str, Any] = {
        "mask_baseline_mean_positive_signal": baseline,
    }
    if brain_mask is None:
        params["intensity_brain_mask"] = {
            "method": "centroid_seeded",
            "reference_radius_voxels": 15,
            "reference_percentile": 90.0,
            "threshold": float(threshold),
            "erosion_voxels": int(erosion_voxels),
        }
    if brain_masking_report is not None:
        params["brain_masking"] = brain_masking_report
    return finite_values, params


def summarize(values: np.ndarray) -> Dict[str, float]:
    """Build summary statistics from selected tSNR values.
    Args:
        values (np.ndarray): Selected finite tSNR values.
    Returns:
        Dict[str, float]: Stats dictionary.
    """
    return {
        "tsnr_mean": float(np.mean(values)),
        "tsnr_median": float(np.median(values)),
        "tsnr_std": float(np.std(values, ddof=0)),
        "tsnr_min": float(np.min(values)),
        "tsnr_max": float(np.max(values)),
    }


def compute_roi_tr_spike_metrics(roi_mean_series: np.ndarray) -> Dict[str, Any]:
    """TR-level outlier metrics on the ROI-mean time course (after timepoint selection).

    Uses robust z (median and MAD scaled by 1.4826) to flag volumes with
    ``|z| > ROBUST_Z_SPIKE_ABS_THRESHOLD`` (4).
    Helps detect corrupted TRs with very high or very low signal relative to the run.

    Args:
        roi_mean_series (np.ndarray): 1D ROI-mean signal, length ``n_timepoints``.

    Returns:
        Dict[str, Any]: Robust counts, fractions, max |z|, scale parameters,
        ``robust_z_per_tr``, and ``roi_mean_signal_per_tr`` (raw ROI mean per TR;
        the latter supports TR-index plots that linear-detrend before z).
    """
    series = np.asarray(roi_mean_series, dtype=np.float64).ravel()
    n_t = int(series.size)
    out: Dict[str, Any] = {
        "n_timepoints": n_t,
        "method_robust_z": "median_and_mad_scaled_1.4826",
    }
    if n_t < 1 or not np.all(np.isfinite(series)):
        out.update(
            {
                "robust_median": 0.0,
                "mad": 0.0,
                "robust_sigma": 0.0,
                "n_tr_abs_robust_z_gt_4": 0,
                "pct_tr_abs_robust_z_gt_4": 0.0,
                "max_abs_robust_z": 0.0,
                "robust_z_per_tr": [],
                "roi_mean_signal_per_tr": [],
            }
        )
        return out
    if n_t < 2:
        out.update(
            {
                "robust_median": float(series[0]),
                "mad": 0.0,
                "robust_sigma": 0.0,
                "n_tr_abs_robust_z_gt_4": 0,
                "pct_tr_abs_robust_z_gt_4": 0.0,
                "max_abs_robust_z": 0.0,
                "robust_z_per_tr": [0.0],
                "roi_mean_signal_per_tr": [float(series[0])],
            }
        )
        return out

    median = float(np.median(series))
    mad = float(np.median(np.abs(series - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        robust_sigma = float(np.std(series, ddof=0))
    rs = float(robust_sigma)
    if rs <= 1e-12:
        signed_z = np.zeros_like(series)
        abs_robust_z = np.zeros_like(series)
    else:
        signed_z = (series - median) / rs
        abs_robust_z = np.abs(signed_z)
    z_cut = float(ROBUST_Z_SPIKE_ABS_THRESHOLD)
    n_rob = int(np.sum(abs_robust_z > z_cut))

    out.update(
        {
            "robust_median": median,
            "mad": mad,
            "robust_sigma": rs,
            "n_tr_abs_robust_z_gt_4": n_rob,
            "pct_tr_abs_robust_z_gt_4": 100.0 * float(n_rob) / float(n_t),
            "max_abs_robust_z": float(np.max(abs_robust_z)),
            "robust_z_per_tr": [float(x) for x in signed_z],
            "roi_mean_signal_per_tr": [float(x) for x in series],
        }
    )
    return out


def compute_ftsnr_metrics(data_4d: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
    """Temporal ftSNR from the spatially averaged ROI signal across time.
    Args:
        data_4d (np.ndarray): Shape ``(x, y, z, t)`` after timepoint selection.
        roi_mask (np.ndarray): Boolean mask shaped ``(x, y, z)``.
    Returns:
        Dict[str, Any]: ``ftsnr``, ``roi_mean_signal_std``, and
        ``roi_mean_tr_spike_metrics`` (TR-level outlier counts on ROI-mean series).
        ``ftsnr`` and ``roi_mean_signal_std`` are zero when temporal std is zero or ROI
        is empty, matching voxelwise tSNR degenerate handling.
    """
    roi_ts = data_4d[roi_mask]
    if roi_ts.size == 0:
        return {
            "ftsnr": 0.0,
            "roi_mean_signal_std": 0.0,
            "roi_mean_tr_spike_metrics": compute_roi_tr_spike_metrics(np.array([], dtype=np.float64)),
        }
    roi_mean_series = np.mean(roi_ts, axis=0)
    spike_block = compute_roi_tr_spike_metrics(roi_mean_series)
    roi_mean_signal_std = float(np.std(roi_mean_series, ddof=0))
    if not np.isfinite(roi_mean_signal_std) or roi_mean_signal_std <= 0.0:
        return {
            "ftsnr": 0.0,
            "roi_mean_signal_std": 0.0,
            "roi_mean_tr_spike_metrics": spike_block,
        }
    ftsnr = float(np.mean(roi_mean_series) / roi_mean_signal_std)
    if not np.isfinite(ftsnr):
        return {
            "ftsnr": 0.0,
            "roi_mean_signal_std": roi_mean_signal_std,
            "roi_mean_tr_spike_metrics": spike_block,
        }
    return {
        "ftsnr": ftsnr,
        "roi_mean_signal_std": roi_mean_signal_std,
        "roi_mean_tr_spike_metrics": spike_block,
    }


def _eligible_slice_indices(
    slice_voxel_counts: Sequence[int],
    min_voxels_floor: int = 50,
    min_voxels_ratio: float = 0.40,
) -> Tuple[List[int], int]:
    """Determine z-slices eligible for slice-level spike analysis.
    Args:
        slice_voxel_counts (Sequence[int]): ROI voxel counts for every z slice.
        min_voxels_floor (int): Minimum absolute voxel count for eligibility.
        min_voxels_ratio (float): Minimum ratio of the max supported slice.
    Returns:
        Tuple[List[int], int]: Eligible z indices and the integer voxel-count threshold.
    """
    if min_voxels_floor < 1:
        raise ValueError("min_voxels_floor must be >= 1")
    if not (0.0 < min_voxels_ratio <= 1.0):
        raise ValueError("min_voxels_ratio must be within (0, 1]")
    max_slice_vox = int(max(slice_voxel_counts)) if slice_voxel_counts else 0
    threshold = int(max(min_voxels_floor, np.ceil(min_voxels_ratio * float(max_slice_vox))))
    eligible = [idx for idx, n_vox in enumerate(slice_voxel_counts) if int(n_vox) >= threshold]
    return eligible, threshold


def compute_slice_ftsnr_metrics(
    data_4d: np.ndarray,
    roi_mask: np.ndarray,
    min_voxels_floor: int = 50,
    min_voxels_ratio: float = 0.40,
) -> Dict[str, Any]:
    """Compute z-slice robust-z spike metrics for localized artifact detection.
    Args:
        data_4d (np.ndarray): Shape ``(x, y, z, t)`` after timepoint selection.
        roi_mask (np.ndarray): Boolean ROI mask shaped ``(x, y, z)``.
        min_voxels_floor (int): Eligibility floor for ROI voxels per z-slice.
        min_voxels_ratio (float): Eligibility ratio versus the max-supported slice.
    Returns:
        Dict[str, Any]: Per-slice |robust z| spike summaries aligned with TR-level
        ``roi_mean_tr_spike_metrics`` (same ``|z|>4`` rule), plus per-slice details.
    """
    z_dim = int(data_4d.shape[2])
    slice_voxel_counts: List[int] = [int(np.count_nonzero(roi_mask[:, :, z])) for z in range(z_dim)]
    n_slices_with_roi = int(sum(1 for n in slice_voxel_counts if n > 0))
    eligible_indices, voxel_threshold = _eligible_slice_indices(
        slice_voxel_counts,
        min_voxels_floor=min_voxels_floor,
        min_voxels_ratio=min_voxels_ratio,
    )
    z_cut = float(ROBUST_Z_SPIKE_ABS_THRESHOLD)

    per_slice: List[Dict[str, Any]] = []
    eligible_rows: List[Dict[str, Any]] = []
    for z in range(z_dim):
        n_vox = int(slice_voxel_counts[z])
        eligible = z in eligible_indices
        if eligible:
            empty_spike = compute_roi_tr_spike_metrics(np.array([], dtype=np.float64))
            row: Dict[str, Any] = {
                "slice_index": int(z),
                "n_voxels": n_vox,
                "eligible": bool(eligible),
                "slice_roi_mean_tr_spike_metrics": empty_spike,
                "slice_n_tr_abs_robust_z_gt_4": 0,
                "slice_pct_tr_abs_robust_z_gt_4": 0.0,
                "slice_max_abs_robust_z": 0.0,
            }
            if n_vox > 0:
                slice_mask = roi_mask[:, :, z]
                roi_ts = data_4d[:, :, z, :][slice_mask]
                roi_mean_series = np.mean(roi_ts, axis=0)
                spike_block = compute_roi_tr_spike_metrics(roi_mean_series)
                row["slice_roi_mean_tr_spike_metrics"] = spike_block
                row["slice_n_tr_abs_robust_z_gt_4"] = int(spike_block.get("n_tr_abs_robust_z_gt_4", 0))
                row["slice_pct_tr_abs_robust_z_gt_4"] = float(
                    spike_block.get("pct_tr_abs_robust_z_gt_4", 0.0)
                )
                row["slice_max_abs_robust_z"] = float(spike_block.get("max_abs_robust_z", 0.0))
            per_slice.append(row)
            eligible_rows.append(row)

    if eligible_rows:
        worst_pct_row = max(
            eligible_rows,
            key=lambda r: (
                float(r["slice_pct_tr_abs_robust_z_gt_4"]),
                float(r["slice_max_abs_robust_z"]),
            ),
        )
        worst_max_row = max(eligible_rows, key=lambda r: float(r["slice_max_abs_robust_z"]))
        worst_slice_spike_pct_slice_index = int(worst_pct_row["slice_index"])
        worst_slice_spike_max_abs_slice_index = int(worst_max_row["slice_index"])
        worst_slice_spike_pct_tr_abs_robust_z_gt_4 = float(worst_pct_row["slice_pct_tr_abs_robust_z_gt_4"])
        worst_slice_spike_max_abs_robust_z = float(worst_max_row["slice_max_abs_robust_z"])
    else:
        worst_slice_spike_pct_slice_index = -1
        worst_slice_spike_max_abs_slice_index = -1
        worst_slice_spike_pct_tr_abs_robust_z_gt_4 = 0.0
        worst_slice_spike_max_abs_robust_z = 0.0

    return {
        "axis": "z",
        "n_slices_total": z_dim,
        "n_slices_with_roi": n_slices_with_roi,
        "n_slices_eligible": int(len(eligible_indices)),
        "eligibility_rule": {
            "min_voxels_floor": int(min_voxels_floor),
            "min_voxels_ratio_of_max_slice": float(min_voxels_ratio),
            "computed_min_voxels_threshold": int(voxel_threshold),
        },
        "slice_spike_abs_z_threshold": z_cut,
        "worst_slice_spike_pct_slice_index": worst_slice_spike_pct_slice_index,
        "worst_slice_spike_pct_tr_abs_robust_z_gt_4": worst_slice_spike_pct_tr_abs_robust_z_gt_4,
        "worst_slice_spike_max_abs_slice_index": worst_slice_spike_max_abs_slice_index,
        "worst_slice_spike_max_abs_robust_z": worst_slice_spike_max_abs_robust_z,
        "same_slice_for_both_spike_flags": bool(
            worst_slice_spike_pct_slice_index >= 0
            and worst_slice_spike_pct_slice_index == worst_slice_spike_max_abs_slice_index
        ),
        "per_slice": per_slice,
    }


def _spike_metrics_compact_view(spike_block: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact TR-spike metrics without large per-TR arrays.
    Args:
        spike_block (Dict[str, Any]): Full spike metrics dictionary.
    Returns:
        Dict[str, Any]: Compact spike metrics preserving scalar summaries.
    """
    out = dict(spike_block)
    out.pop("robust_z_per_tr", None)
    out.pop("roi_mean_signal_per_tr", None)
    return out


def compact_slice_ftsnr_metrics(slice_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact slice metrics for JSON output.
    In compact mode, ``per_slice`` is omitted to reduce JSON size; consumers
    should interpret missing per-slice rows as empty/not-applicable slices.
    Args:
        slice_metrics (Dict[str, Any]): Full slice metrics block.
    Returns:
        Dict[str, Any]: Compact slice metrics block.
    """
    out = dict(slice_metrics)
    out.pop("per_slice", None)
    return out


def _analysis_roi_mask_for_summary(
    mode: str,
    volume_shape_3d: Tuple[int, int, int],
    mean_volume: np.ndarray,
    parameters: Dict[str, Any],
    brain_mask_override: Optional[np.ndarray],
) -> np.ndarray:
    """Same 3D ROI mask used for JSON summaries and optional map censoring.
    Args:
        mode (str): ``phantom`` or ``brain``.
        volume_shape_3d (Tuple[int, int, int]): Spatial shape of 3D maps.
        mean_volume (np.ndarray): 3D temporal mean (brain intensity mask path).
        parameters (Dict[str, Any]): Mode-specific parameters dict.
        brain_mask_override (Optional[np.ndarray]): T1-registered mask when set.
    Returns:
        np.ndarray: Boolean mask shaped ``volume_shape_3d``.
    Raises:
        ValueError: Brain mode without override and without intensity mask params.
    """
    if mode == "brain":
        if brain_mask_override is not None:
            return np.asarray(brain_mask_override, dtype=bool)
        ib = parameters.get("intensity_brain_mask")
        if not isinstance(ib, dict):
            raise ValueError(
                "brain mode without a T1 mask override requires "
                "parameters['intensity_brain_mask'] from intensity masking"
            )
        return build_brain_mask(
            mean_volume,
            float(ib["threshold"]),
            int(ib["erosion_voxels"]),
        )
    if mode == "phantom":
        return build_phantom_analysis_mask(volume_shape_3d, mean_volume, parameters)
    return np.ones(volume_shape_3d, dtype=bool)


def apply_spatial_mask_nan(
    tsnr_map: np.ndarray,
    mean_volume: np.ndarray,
    std_volume: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy volumes and set voxels outside ``mask`` to NaN (float32).
    Args:
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D temporal mean.
        std_volume (np.ndarray): 3D temporal std.
        mask (np.ndarray): Boolean mask, same shape as each volume.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Masked copies as ``float32``.
    """
    tsnr_out = tsnr_map.astype(np.float32, copy=True)
    mean_out = mean_volume.astype(np.float32, copy=True)
    std_out = std_volume.astype(np.float32, copy=True)
    inv = ~mask
    tsnr_out[inv] = np.nan
    mean_out[inv] = np.nan
    std_out[inv] = np.nan
    return tsnr_out, mean_out, std_out


def save_outputs(
    input_path: Path,
    output_dir: Path,
    input_type: str,
    mode: str,
    data_4d: np.ndarray,
    tsnr_map: np.ndarray,
    mean_volume: np.ndarray,
    std_volume: np.ndarray,
    selected_values: np.ndarray,
    parameters: Dict[str, Any],
    timepoint_selection: Dict[str, int],
    source_nifti: Optional[nib.Nifti1Image],
    write_tmean_tstd: bool,
    mask_maps: bool = True,
    brain_mask_override: Optional[np.ndarray] = None,
    slice_min_voxels_floor: int = 50,
    slice_min_voxels_ratio: float = 0.40,
    compact_json: bool = True,
) -> Tuple[Path, Path, List[Path]]:
    """Write map and JSON outputs.
    Args:
        input_path (Path): Input path.
        output_dir (Path): Output directory.
        input_type (str): `nifti` or `fmriqa_pixel_cache_npz`.
        mode (str): Analysis mode.
        data_4d (np.ndarray): Internal `(x, y, z, t)` data after timepoint selection.
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D temporal mean (`Tmean`).
        std_volume (np.ndarray): 3D temporal standard deviation with `ddof=1` (`Tstd`).
        selected_values (np.ndarray): Values used for summary.
        parameters (Dict[str, Any]): Mode-specific parameters.
        timepoint_selection (Dict[str, int]): Indices and counts for the time axis.
        source_nifti (Optional[nib.Nifti1Image]): Source image for affine/header.
        write_tmean_tstd (bool): When True, also write `{basename}_Tmean.nii.gz`
            and `{basename}_Tstd.nii.gz` for troubleshooting.
        mask_maps (bool): When True (default), voxels outside the analysis ROI are
            set to NaN in written NIfTI maps so viewers match JSON ROI summaries.
        brain_mask_override (Optional[np.ndarray]): For ``brain`` mode, optional
            boolean mask (same shape as ``mean_volume``) to use instead of
            recomputing the intensity-based mask.
        slice_min_voxels_floor (int): Slice-level spike eligibility floor.
        slice_min_voxels_ratio (float): Slice-level eligibility ratio of max slice support.
        compact_json (bool): When True (default), omit high-volume JSON arrays
            and per-slice details that are not used in summary plotting.
    Returns:
        Tuple[Path, Path, List[Path]]: `(map_path, stats_path, optional_map_paths)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = derive_basename(input_path)
    map_path = output_dir / f"{basename}_tsnr_map.nii.gz"
    stats_path = output_dir / f"{basename}_tsnr_stats.json"

    analysis_roi_mask = _analysis_roi_mask_for_summary(
        mode,
        tsnr_map.shape,
        mean_volume,
        parameters,
        brain_mask_override,
    )
    ft_metrics = compute_ftsnr_metrics(data_4d, analysis_roi_mask)
    if compact_json:
        ft_metrics = dict(ft_metrics)
        roi_spike = ft_metrics.get("roi_mean_tr_spike_metrics")
        if isinstance(roi_spike, dict):
            ft_metrics["roi_mean_tr_spike_metrics"] = _spike_metrics_compact_view(roi_spike)
    slice_ft_metrics: Optional[Dict[str, Any]] = compute_slice_ftsnr_metrics(
        data_4d,
        analysis_roi_mask,
        min_voxels_floor=slice_min_voxels_floor,
        min_voxels_ratio=slice_min_voxels_ratio,
    )
    if compact_json and isinstance(slice_ft_metrics, dict):
        slice_ft_metrics = compact_slice_ftsnr_metrics(slice_ft_metrics)

    output_map_censoring = "full_fov"
    tsnr_save = tsnr_map
    mean_save = mean_volume
    std_save = std_volume
    if mask_maps:
        tsnr_save, mean_save, std_save = apply_spatial_mask_nan(
            tsnr_map,
            mean_volume,
            std_volume,
            analysis_roi_mask,
        )
        output_map_censoring = "roi_masked"

    if source_nifti is not None:
        map_img = nib.Nifti1Image(
            tsnr_save.astype(np.float32),
            source_nifti.affine,
            source_nifti.header,
        )
        map_affine_source = "input"
    else:
        map_img = nib.Nifti1Image(tsnr_save.astype(np.float32), np.eye(4, dtype=np.float64))
        map_affine_source = "identity"

    nib.save(map_img, str(map_path))
    optional_paths: List[Path] = []
    if write_tmean_tstd:
        tmean_path = output_dir / f"{basename}_Tmean.nii.gz"
        tstd_path = output_dir / f"{basename}_Tstd.nii.gz"
        if source_nifti is not None:
            mean_img = nib.Nifti1Image(
                mean_save.astype(np.float32),
                source_nifti.affine,
                source_nifti.header,
            )
            std_img = nib.Nifti1Image(
                std_save.astype(np.float32),
                source_nifti.affine,
                source_nifti.header,
            )
        else:
            affine = np.eye(4, dtype=np.float64)
            mean_img = nib.Nifti1Image(mean_save.astype(np.float32), affine)
            std_img = nib.Nifti1Image(std_save.astype(np.float32), affine)
        nib.save(mean_img, str(tmean_path))
        nib.save(std_img, str(tstd_path))
        optional_paths = [tmean_path, tstd_path]
    payload: Dict[str, Any] = {
        "input_file": str(input_path),
        "input_type": input_type,
        "mode": mode,
        "n_timepoints": int(data_4d.shape[3]),
        "volume_shape": [int(x) for x in data_4d.shape],
        **summarize(selected_values),
        **ft_metrics,
        "n_voxels_in_roi": int(selected_values.size),
        "map_affine_source": map_affine_source,
        "output_map_censoring": output_map_censoring,
        "timepoint_selection": timepoint_selection,
        "parameters": parameters,
    }
    qa_session_date = derive_qa_session_date(input_path)
    if qa_session_date is not None:
        payload["qa_session_date"] = qa_session_date
    if slice_ft_metrics is not None:
        payload["slice_ftsnr_metrics"] = slice_ft_metrics
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return map_path, stats_path, optional_paths


def run_analysis(
    input_path: Path,
    mode: str,
    output_dir: Optional[Path] = None,
    roi_size: int = 15,
    slice_index: Optional[int] = None,
    phantom_roi_mode: str = "patch",
    phantom_edge_erosion_voxels: int = 1,
    phantom_full_threshold_fraction: float = 0.35,
    threshold: float = 0.25,
    erosion_voxels: int = 2,
    write_tmean_tstd: bool = False,
    first_timepoint: int = 2,
    last_timepoint: Optional[int] = None,
    mask_maps: bool = True,
    slice_min_voxels_floor: int = 50,
    slice_min_voxels_ratio: float = 0.40,
    full_json_details: bool = False,
) -> Tuple[Path, Path, List[Path]]:
    """Run full tSNR analysis and write outputs.
    Args:
        input_path (Path): NIfTI or NPZ path.
        mode (str): `phantom` or `brain`.
        output_dir (Optional[Path]): Destination directory. When ``None``, defaults
            to sibling ``derivatives/tsnr`` (BIDS-style ``func`` inputs use
            ``<session>/derivatives/tsnr``).
        roi_size (int): Phantom ROI side.
        slice_index (Optional[int]): Phantom slice override.
        phantom_roi_mode (str): Phantom ROI mode (``patch`` or
            ``full_minus_edges``).
        phantom_edge_erosion_voxels (int): Per-slice in-plane erosion iterations
            for ``full_minus_edges``.
        phantom_full_threshold_fraction (float): Intensity threshold fraction for
            phantom ``full_minus_edges`` (vs local reference).
        threshold (float): Brain threshold (used for centroid-seeded fallback and
            recorded in JSON; when a T1-derived mask is used, it does not define that
            mask).
        erosion_voxels (int): Brain erosion iterations (same as ``threshold`` for
            fallback vs metadata).
        write_tmean_tstd (bool): When True, write optional `{basename}_Tmean.nii.gz`
            and `{basename}_Tstd.nii.gz` maps for troubleshooting (off by default).
        first_timepoint (int): 0-based index of the first volume to include. Default
            ``2`` drops the first two volumes to reduce non-steady-state transient effects
            in fMRI signal. Use ``0`` to include from the first volume, or ``1`` to drop only the first.
        last_timepoint (Optional[int]): 0-based index of the last volume to include,
            or ``None`` for the final volume in the file.
        mask_maps (bool): When True (default), set NIfTI map voxels outside the ROI
            to NaN. Set False for full field-of-view maps.
        slice_min_voxels_floor (int): Slice-level spike eligibility floor.
        slice_min_voxels_ratio (float): Slice-level eligibility ratio of max slice support.
        full_json_details (bool): When True, retain full per-TR and per-slice JSON
            arrays. Default False writes compact JSON summaries.
    Returns:
        Tuple[Path, Path, List[Path]]: `(map_path, stats_path, optional_intermediate_paths)`.
    Raises:
        ValueError: If validation or processing fails.
    """
    validate_common_args(
        mode,
        threshold,
        erosion_voxels,
        phantom_roi_mode,
        phantom_edge_erosion_voxels,
        phantom_full_threshold_fraction,
    )
    if not input_path.is_file():
        raise ValueError(f"Input does not exist: {input_path}")

    source_nifti: Optional[nib.Nifti1Image] = None
    suffixes = input_path.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or input_path.suffix == ".nii":
        input_type = "nifti"
        data_4d, source_nifti = load_nifti_4d(input_path)
    elif input_path.suffix == ".npz":
        if mode != "phantom":
            raise ValueError("brain mode does not accept .npz input")
        input_type = "fmriqa_pixel_cache_npz"
        data_4d = load_phantom_npz_4d(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.name}")

    target_dir = output_dir if output_dir is not None else default_output_dir_for_input(input_path)
    brain_mask_override: Optional[np.ndarray] = None

    if mode == "phantom":
        pa = run_phantom_analysis_from_4d(
            data_4d,
            roi_size=roi_size,
            slice_index=slice_index,
            phantom_roi_mode=phantom_roi_mode,
            phantom_edge_erosion_voxels=phantom_edge_erosion_voxels,
            phantom_full_threshold_fraction=phantom_full_threshold_fraction,
            first_timepoint=first_timepoint,
            last_timepoint=last_timepoint,
        )
        data_4d = pa["data_4d"]
        mean_volume = pa["mean_volume"]
        std_volume = pa["std_volume"]
        tsnr_map = pa["tsnr_map"]
        selected_values = pa["selected_values"]
        parameters = pa["parameters"]
        timepoint_selection = pa["timepoint_selection"]
    else:
        data_4d, timepoint_selection = apply_timepoint_selection(
            data_4d,
            first_index=first_timepoint,
            last_index_inclusive=last_timepoint,
        )
        mean_volume, std_volume, tsnr_map = compute_tsnr_map(data_4d)
        if source_nifti is not None:
            t1_path = find_t1_in_anat(input_path)
            if t1_path is None:
                print(
                    "WARNING: No T1w found in anat/ (or BET pipeline failed) "
                    "— falling back to non-T1 brain masking",
                    file=sys.stderr,
                )
                selected_values, parameters = extract_brain_tsnr(
                    tsnr_map,
                    mean_volume,
                    threshold,
                    erosion_voxels,
                    brain_masking_report=_brain_masking_fallback_no_t1(),
                )
            else:
                work_dir = target_dir / ".tsnr_fsl_work" / derive_basename(input_path)
                try:
                    mask_path = create_bet_mask_for_func(
                        t1_path,
                        mean_volume,
                        source_nifti,
                        work_dir,
                    )
                    mask_img = nib.load(str(mask_path))
                    mask_data = np.asarray(mask_img.get_fdata(dtype=np.float64))
                    brain_mask_override = mask_data > 0.5
                    selected_values, parameters = extract_brain_tsnr(
                        tsnr_map,
                        mean_volume,
                        threshold,
                        erosion_voxels,
                        brain_mask=brain_mask_override,
                        brain_masking_report=_brain_masking_success(t1_path),
                    )
                except (OSError, ValueError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                    brain_mask_override = None
                    print(
                        "WARNING: No T1w found in anat/ (or BET pipeline failed) "
                        "— falling back to non-T1 brain masking",
                        file=sys.stderr,
                    )
                    selected_values, parameters = extract_brain_tsnr(
                        tsnr_map,
                        mean_volume,
                        threshold,
                        erosion_voxels,
                        brain_masking_report=_brain_masking_fallback_pipeline_failed(t1_path, exc),
                    )
        else:
            selected_values, parameters = extract_brain_tsnr(
                tsnr_map,
                mean_volume,
                threshold,
                erosion_voxels,
                brain_masking_report=_brain_masking_fallback_no_nifti_header(),
            )

    return save_outputs(
        input_path=input_path,
        output_dir=target_dir,
        input_type=input_type,
        mode=mode,
        data_4d=data_4d,
        tsnr_map=tsnr_map,
        mean_volume=mean_volume,
        std_volume=std_volume,
        selected_values=selected_values,
        parameters=parameters,
        timepoint_selection=timepoint_selection,
        source_nifti=source_nifti,
        write_tmean_tstd=write_tmean_tstd,
        mask_maps=mask_maps,
        brain_mask_override=brain_mask_override,
        slice_min_voxels_floor=slice_min_voxels_floor,
        slice_min_voxels_ratio=slice_min_voxels_ratio,
        compact_json=not full_json_details,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.
    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description="Compute raw fMRI tSNR map and summary statistics.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a NIfTI or NPZ file, or a directory of BOLD runs (see --input-pattern)",
    )
    parser.add_argument("mode", choices=("phantom", "brain"), help="Analysis mode")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--roi-size", type=int, default=15, help="Phantom ROI side length (odd integer)")
    parser.add_argument("--slice-index", type=int, default=None, help="Phantom slice index (default: middle)")
    parser.add_argument(
        "--phantom-roi-mode",
        type=str,
        choices=("patch", "full_minus_edges"),
        default="patch",
        help="Phantom ROI selection mode: centered patch (default) or full phantom minus edges",
    )
    parser.add_argument(
        "--phantom-edge-erosion-voxels",
        type=int,
        default=1,
        help="Per-slice (x/y) erosion iterations used only with --phantom-roi-mode full_minus_edges",
    )
    parser.add_argument(
        "--phantom-full-threshold-fraction",
        type=float,
        default=0.35,
        metavar="F",
        help="Intensity threshold as fraction of local reference for --phantom-roi-mode full_minus_edges (default: 0.35)",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Brain threshold fraction")
    parser.add_argument("--erosion-voxels", type=int, default=2, help="Brain erosion iterations")
    parser.add_argument(
        "--write-tmean-tstd",
        action="store_true",
        help="Also write temporal mean/std volumes as {basename}_Tmean.nii.gz and "
        "{basename}_Tstd.nii.gz (optional troubleshooting; matches legacy naming)",
    )
    parser.add_argument(
        "--first-timepoint",
        type=int,
        default=2,
        metavar="IDX",
        help=(
            "0-based index of first volume to include (default: 2 drops the first two "
            "volumes to reduce non-steady-state transient effects; use 0 or 1 for a shorter lead-in)"
        ),
    )
    parser.add_argument(
        "--last-timepoint",
        type=int,
        default=None,
        metavar="IDX",
        help="0-based index of last volume to include (default: last volume in file)",
    )
    parser.add_argument(
        "--full-fov-maps",
        action="store_true",
        help="Write tSNR/Tmean/Tstd NIfTIs for the full field of view (do not NaN "
        "voxels outside the brain mask or phantom ROI)",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=None,
        metavar="GLOB",
        help="When input is a directory, glob pattern for NIfTIs (default: *_bold.nii.gz "
        "and *_bold.nii). Use a single pattern, e.g. *_task-rest_*.nii.gz",
    )
    parser.add_argument(
        "--slice-min-voxels-floor",
        type=int,
        default=50,
        help="Minimum ROI voxels required for a z-slice to be eligible for slice spike ranking.",
    )
    parser.add_argument(
        "--slice-min-voxels-ratio",
        type=float,
        default=0.40,
        help="Minimum eligible slice support as ratio of max slice ROI support (0,1].",
    )
    parser.add_argument(
        "--full-json-details",
        action="store_true",
        help="Write full per-TR and per-slice JSON details (default writes compact summaries).",
    )
    return parser


def _run_one_analysis_from_cli(args: argparse.Namespace, input_path: Path) -> None:
    """Invoke ``run_analysis`` for one input using CLI-bound options.
    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        input_path (Path): One NIfTI or NPZ file.
    Raises:
        ValueError: From ``run_analysis`` on invalid input or processing.
    """
    run_analysis(
        input_path=input_path,
        mode=args.mode,
        output_dir=args.output_dir,
        roi_size=args.roi_size,
        slice_index=args.slice_index,
        phantom_roi_mode=args.phantom_roi_mode,
        phantom_edge_erosion_voxels=args.phantom_edge_erosion_voxels,
        phantom_full_threshold_fraction=args.phantom_full_threshold_fraction,
        threshold=args.threshold,
        erosion_voxels=args.erosion_voxels,
        write_tmean_tstd=args.write_tmean_tstd,
        first_timepoint=args.first_timepoint,
        last_timepoint=args.last_timepoint,
        mask_maps=not args.full_fov_maps,
        slice_min_voxels_floor=args.slice_min_voxels_floor,
        slice_min_voxels_ratio=args.slice_min_voxels_ratio,
        full_json_details=bool(args.full_json_details),
    )


def cli(argv: Optional[list[str]] = None) -> int:
    """Run CLI entry point.
    Args:
        argv (Optional[list[str]]): Optional argument list.
    Returns:
        int: Process exit code.
    """
    args = build_arg_parser().parse_args(argv)
    input_path = args.input.expanduser()

    if input_path.is_file():
        try:
            _run_one_analysis_from_cli(args, input_path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    if input_path.is_dir():
        try:
            bold_paths = list_bold_niftis_in_dir(input_path, pattern=args.input_pattern)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if not bold_paths:
            default_hint = "*_bold.nii.gz and *_bold.nii" if args.input_pattern is None else args.input_pattern
            print(
                f"Error: no matching BOLD NIfTIs in {input_path} (pattern: {default_hint})",
                file=sys.stderr,
            )
            return 1
        failures = 0
        for bold in bold_paths:
            try:
                _run_one_analysis_from_cli(args, bold)
            except (ValueError, OSError) as exc:
                print(f"Error: {bold}: {exc}", file=sys.stderr)
                failures += 1
        return 1 if failures > 0 else 0

    print(f"Error: input is not a file or directory: {input_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
