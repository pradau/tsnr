# Author: Perry Radau
# Date: 2026-04-16
# Compute raw tSNR maps and summary statistics for phantom and brain fMRI inputs.
# Dependencies: Python 3.10+, nibabel, numpy, scipy
# Usage: uv run python main.py /path/to/input.nii.gz phantom
#        uv run python main.py /path/to/input.nii.gz brain --threshold 0.25 --erosion-voxels 2
#        uv run python main.py /path/to/cache.npz phantom --roi-size 15

"""tSNR calculation helpers and CLI orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass


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


def validate_common_args(mode: str, threshold: float, erosion_voxels: int) -> None:
    """Validate common CLI arguments.

    Args:
        mode (str): Analysis mode.
        threshold (float): Brain threshold fraction.
        erosion_voxels (int): Brain erosion iterations.

    Raises:
        ValueError: If any value is invalid.
    """
    if mode not in ("phantom", "brain"):
        raise ValueError(f"Invalid mode: {mode}")
    if not (0.0 < threshold < 1.0):
        raise ValueError("--threshold must be strictly between 0 and 1")
    if erosion_voxels < 0:
        raise ValueError("--erosion-voxels must be >= 0")


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
    if volume.ndim != 4:
        raise ValueError(f"Expected NPZ volume shape (slices, time, rows, cols), got {volume.shape}")
    if volume.shape[1] < 2:
        raise ValueError(f"Need at least 2 time points, got {volume.shape[1]}")
    if not np.all(np.isfinite(volume)):
        raise ValueError("Input contains non-finite values")
    data_4d = np.transpose(volume, (2, 3, 0, 1))
    return data_4d


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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract phantom ROI values from selected slice.

    Args:
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D mean volume.
        roi_size (int): ROI side length.
        slice_index (Optional[int]): Selected slice or None for middle.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: ROI values and parameter metadata.

    Raises:
        ValueError: If slice index or ROI extraction is invalid.
    """
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
        "roi_size": roi_size,
        "slice_index": int(target_slice),
        "roi_bounds": [int(row_start), int(row_end), int(col_start), int(col_end)],
    }
    return finite_values, params


def extract_brain_tsnr(
    tsnr_map: np.ndarray,
    mean_volume: np.ndarray,
    threshold: float,
    erosion_voxels: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract brain-mask tSNR values.

    Args:
        tsnr_map (np.ndarray): 3D tSNR map.
        mean_volume (np.ndarray): 3D mean volume.
        threshold (float): Relative threshold fraction.
        erosion_voxels (int): Binary erosion iterations.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Brain values and parameter metadata.

    Raises:
        ValueError: If mask creation fails or is empty.
    """
    positive = mean_volume > 0.0
    if not np.any(positive):
        raise ValueError("brain mask baseline could not be computed: no positive voxels")
    baseline = float(np.mean(mean_volume[positive]))
    mask = mean_volume >= (threshold * baseline)
    if erosion_voxels > 0:
        mask = binary_erosion(mask, iterations=erosion_voxels)
    if not np.any(mask):
        raise ValueError("brain mask is empty after erosion")
    values = tsnr_map[mask]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("brain mask contains zero valid voxels")
    params = {
        "threshold": float(threshold),
        "erosion_voxels": int(erosion_voxels),
        "mask_baseline_mean_positive_signal": baseline,
    }
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


def save_outputs(
    input_path: Path,
    output_dir: Path,
    input_type: str,
    mode: str,
    data_4d: np.ndarray,
    tsnr_map: np.ndarray,
    selected_values: np.ndarray,
    parameters: Dict[str, Any],
    source_nifti: Optional[nib.Nifti1Image],
) -> Tuple[Path, Path]:
    """Write map and JSON outputs.

    Args:
        input_path (Path): Input path.
        output_dir (Path): Output directory.
        input_type (str): `nifti` or `fmriqa_pixel_cache_npz`.
        mode (str): Analysis mode.
        data_4d (np.ndarray): Internal `(x, y, z, t)` data.
        tsnr_map (np.ndarray): 3D tSNR map.
        selected_values (np.ndarray): Values used for summary.
        parameters (Dict[str, Any]): Mode-specific parameters.
        source_nifti (Optional[nib.Nifti1Image]): Source image for affine/header.

    Returns:
        Tuple[Path, Path]: `(map_path, stats_path)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = derive_basename(input_path)
    map_path = output_dir / f"{basename}_tsnr_map.nii.gz"
    stats_path = output_dir / f"{basename}_tsnr_stats.json"

    if source_nifti is not None:
        map_img = nib.Nifti1Image(
            tsnr_map.astype(np.float32),
            source_nifti.affine,
            source_nifti.header,
        )
        map_affine_source = "input"
    else:
        map_img = nib.Nifti1Image(tsnr_map.astype(np.float32), np.eye(4, dtype=np.float64))
        map_affine_source = "identity"

    nib.save(map_img, str(map_path))
    payload: Dict[str, Any] = {
        "input_file": str(input_path),
        "input_type": input_type,
        "mode": mode,
        "n_timepoints": int(data_4d.shape[3]),
        "volume_shape": [int(x) for x in data_4d.shape],
        **summarize(selected_values),
        "n_voxels_in_roi": int(selected_values.size),
        "map_affine_source": map_affine_source,
        "parameters": parameters,
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return map_path, stats_path


def run_analysis(
    input_path: Path,
    mode: str,
    output_dir: Optional[Path] = None,
    roi_size: int = 15,
    slice_index: Optional[int] = None,
    threshold: float = 0.25,
    erosion_voxels: int = 2,
) -> Tuple[Path, Path]:
    """Run full tSNR analysis and write outputs.

    Args:
        input_path (Path): NIfTI or NPZ path.
        mode (str): `phantom` or `brain`.
        output_dir (Optional[Path]): Destination directory.
        roi_size (int): Phantom ROI side.
        slice_index (Optional[int]): Phantom slice override.
        threshold (float): Brain threshold.
        erosion_voxels (int): Brain erosion iterations.

    Returns:
        Tuple[Path, Path]: `(map_path, stats_path)`.

    Raises:
        ValueError: If validation or processing fails.
    """
    validate_common_args(mode, threshold, erosion_voxels)
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

    mean_volume, _std_volume, tsnr_map = compute_tsnr_map(data_4d)
    if mode == "phantom":
        selected_values, parameters = extract_phantom_tsnr(tsnr_map, mean_volume, roi_size, slice_index)
    else:
        selected_values, parameters = extract_brain_tsnr(tsnr_map, mean_volume, threshold, erosion_voxels)

    target_dir = output_dir if output_dir is not None else input_path.parent
    return save_outputs(
        input_path=input_path,
        output_dir=target_dir,
        input_type=input_type,
        mode=mode,
        data_4d=data_4d,
        tsnr_map=tsnr_map,
        selected_values=selected_values,
        parameters=parameters,
        source_nifti=source_nifti,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description="Compute raw fMRI tSNR map and summary statistics.")
    parser.add_argument("input", type=Path, help="Input .nii/.nii.gz or phantom-only .npz path")
    parser.add_argument("mode", choices=("phantom", "brain"), help="Analysis mode")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--roi-size", type=int, default=15, help="Phantom ROI side length (odd integer)")
    parser.add_argument("--slice-index", type=int, default=None, help="Phantom slice index (default: middle)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Brain threshold fraction")
    parser.add_argument("--erosion-voxels", type=int, default=2, help="Brain erosion iterations")
    return parser


def cli(argv: Optional[list[str]] = None) -> int:
    """Run CLI entry point.

    Args:
        argv (Optional[list[str]]): Optional argument list.

    Returns:
        int: Process exit code.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        run_analysis(
            input_path=args.input,
            mode=args.mode,
            output_dir=args.output_dir,
            roi_size=args.roi_size,
            slice_index=args.slice_index,
            threshold=args.threshold,
            erosion_voxels=args.erosion_voxels,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
