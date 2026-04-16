# Author: Perry Radau
# Date: 2026-04-16
# Validate tSNR calculator behavior for NIfTI and phantom NPZ inputs.
# Dependencies: Python 3.10+, pytest, numpy, nibabel
# Usage: uv run pytest tests/test_tsnr.py

"""Acceptance tests for tSNR implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsnr import run_analysis


def write_nifti(path: Path, data: np.ndarray) -> None:
    """Write a NIfTI file with identity affine.

    Args:
        path (Path): Output path.
        data (np.ndarray): NIfTI data.

    Returns:
        None: This function returns nothing.
    """
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def load_stats(path: Path) -> Dict[str, object]:
    """Read stats JSON output.

    Args:
        path (Path): JSON path.

    Returns:
        Dict[str, object]: Parsed payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def make_phantom_data(shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Create synthetic phantom-like 4D data.

    Args:
        shape (Tuple[int, int, int, int]): `(x, y, z, t)` shape.

    Returns:
        np.ndarray: Synthetic data.
    """
    x, y, z, t = shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(-1, 1, x),
        np.linspace(-1, 1, y),
        np.linspace(-1, 1, z),
        indexing="ij",
    )
    base = np.exp(-(grid_x**2 + grid_y**2 + grid_z**2) * 3.0) * 100.0
    out = np.zeros((x, y, z, t), dtype=np.float64)
    for ti in range(t):
        out[:, :, :, ti] = base + ti * 0.5
    return out


def test_nifti_phantom_happy_path(tmp_path: Path) -> None:
    """Phantom analysis writes expected outputs and stats."""
    data = make_phantom_data((21, 21, 9, 8))
    input_path = tmp_path / "phantom.nii.gz"
    write_nifti(input_path, data)

    map_path, stats_path = run_analysis(input_path=input_path, mode="phantom")
    assert map_path.name == "phantom_tsnr_map.nii.gz"
    assert stats_path.name == "phantom_tsnr_stats.json"
    image = nib.load(str(map_path))
    assert image.shape == (21, 21, 9)
    stats = load_stats(stats_path)
    assert stats["mode"] == "phantom"
    assert stats["n_voxels_in_roi"] == 225
    assert stats["map_affine_source"] == "input"
    assert isinstance(stats["parameters"], dict)


def test_nifti_brain_happy_path(tmp_path: Path) -> None:
    """Brain analysis uses positive-voxel baseline and erosion."""
    data = np.zeros((9, 9, 5, 6), dtype=np.float64)
    data[2:7, 2:7, 1:4, :] = 100.0
    for ti in range(data.shape[3]):
        data[2:7, 2:7, 1:4, ti] += ti
    input_path = tmp_path / "brain.nii.gz"
    write_nifti(input_path, data)

    _, stats_path = run_analysis(
        input_path=input_path,
        mode="brain",
        threshold=0.25,
        erosion_voxels=1,
    )
    stats = load_stats(stats_path)
    params = stats["parameters"]
    assert stats["mode"] == "brain"
    assert stats["n_voxels_in_roi"] > 0
    assert params["mask_baseline_mean_positive_signal"] > 0.0
    assert params["erosion_voxels"] == 1


def test_zero_std_handling_maps_to_zero(tmp_path: Path) -> None:
    """Constant series yields zero tSNR and finite output."""
    data = np.ones((11, 11, 3, 4), dtype=np.float64) * 10.0
    input_path = tmp_path / "constant.nii.gz"
    write_nifti(input_path, data)

    map_path, stats_path = run_analysis(input_path=input_path, mode="phantom", roi_size=11)
    values = np.asarray(nib.load(str(map_path)).get_fdata(dtype=np.float64))
    assert np.all(np.isfinite(values))
    assert np.all(values == 0.0)
    stats = load_stats(stats_path)
    assert stats["tsnr_mean"] == 0.0


def test_phantom_roi_near_boundary_is_shifted(tmp_path: Path) -> None:
    """Boundary ROI placement shifts in-bounds and retains full size."""
    data = np.zeros((21, 21, 5, 6), dtype=np.float64)
    data[1:4, 1:4, 2, :] = 100.0
    for ti in range(data.shape[3]):
        data[:, :, :, ti] += ti
    input_path = tmp_path / "edge.nii.gz"
    write_nifti(input_path, data)

    _, stats_path = run_analysis(input_path=input_path, mode="phantom", roi_size=15, slice_index=2)
    stats = load_stats(stats_path)
    row_start, row_end, col_start, col_end = stats["parameters"]["roi_bounds"]
    assert row_start == 0
    assert col_start == 0
    assert (row_end - row_start) == 15
    assert (col_end - col_start) == 15


def test_oversized_roi_fails(tmp_path: Path) -> None:
    """Oversized phantom ROI raises a clear error."""
    data = make_phantom_data((11, 11, 3, 5))
    input_path = tmp_path / "small.nii.gz"
    write_nifti(input_path, data)

    with pytest.raises(ValueError, match="roi-size .* exceeds image dimensions"):
        run_analysis(input_path=input_path, mode="phantom", roi_size=15)


def test_invalid_nifti_dimensions_fail(tmp_path: Path) -> None:
    """3D and 5D inputs are rejected."""
    input_3d = tmp_path / "three_d.nii.gz"
    write_nifti(input_3d, np.ones((5, 5, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="Expected 4D NIfTI input"):
        run_analysis(input_path=input_3d, mode="phantom")

    input_5d = tmp_path / "five_d.nii.gz"
    write_nifti(input_5d, np.ones((3, 3, 3, 3, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="Expected 4D NIfTI input"):
        run_analysis(input_path=input_5d, mode="brain")


def test_empty_brain_masks_fail(tmp_path: Path) -> None:
    """Threshold and erosion emptiness failures are surfaced."""
    data = np.ones((9, 9, 5, 6), dtype=np.float64)
    input_path = tmp_path / "mask_fail.nii.gz"
    write_nifti(input_path, data)
    with pytest.raises(ValueError, match="brain mask is empty after erosion"):
        run_analysis(
            input_path=input_path,
            mode="brain",
            threshold=0.99,
            erosion_voxels=10,
        )


def test_npz_phantom_compatibility_and_brain_rejection(tmp_path: Path) -> None:
    """NPZ v1/v2 load in phantom mode and reject in brain mode."""
    volume = np.ones((4, 5, 9, 9), dtype=np.float64)
    for ti in range(volume.shape[1]):
        volume[:, ti, :, :] += ti

    npz_v2 = tmp_path / "cache_v2.npz"
    np.savez_compressed(npz_v2, cache_version=np.int32(2), volume=volume)
    _, stats_v2 = run_analysis(input_path=npz_v2, mode="phantom", roi_size=9)
    payload_v2 = load_stats(stats_v2)
    assert payload_v2["input_type"] == "fmriqa_pixel_cache_npz"
    assert payload_v2["map_affine_source"] == "identity"

    npz_v1 = tmp_path / "cache_v1.npz"
    np.savez_compressed(npz_v1, cache_version=np.int32(1), volume=volume)
    _, stats_v1 = run_analysis(input_path=npz_v1, mode="phantom", roi_size=9)
    payload_v1 = load_stats(stats_v1)
    assert payload_v1["n_timepoints"] == 5

    with pytest.raises(ValueError, match="brain mode does not accept .npz input"):
        run_analysis(input_path=npz_v2, mode="brain")


def test_output_naming_for_nii_gz(tmp_path: Path) -> None:
    """Double-suffix NIfTI naming strips .nii.gz as one unit."""
    data = make_phantom_data((11, 11, 3, 4))
    input_path = tmp_path / "example.nii.gz"
    write_nifti(input_path, data)
    map_path, stats_path = run_analysis(input_path=input_path, mode="phantom", roi_size=11)
    assert map_path.name == "example_tsnr_map.nii.gz"
    assert stats_path.name == "example_tsnr_stats.json"


def test_json_contract_fields_present(tmp_path: Path) -> None:
    """Stats JSON includes required contract fields."""
    data = make_phantom_data((13, 13, 3, 4))
    input_path = tmp_path / "contract.nii.gz"
    write_nifti(input_path, data)
    _, stats_path = run_analysis(input_path=input_path, mode="phantom", roi_size=13)
    stats = load_stats(stats_path)
    assert stats["input_type"] == "nifti"
    assert stats["volume_shape"] == [13, 13, 3, 4]
    assert stats["map_affine_source"] == "input"
    assert "parameters" in stats
