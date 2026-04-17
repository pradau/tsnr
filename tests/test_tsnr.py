# Author: Perry Radau
# Date: 2026-04-16
# Validate tSNR calculator behavior for NIfTI and phantom NPZ inputs.
# Dependencies: Python 3.10+, pytest, numpy, nibabel
# Usage: uv run pytest tests/test_tsnr.py

"""Acceptance tests for tSNR implementation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsnr import cli, compute_tsnr_map, extract_brain_tsnr, find_t1_in_anat, list_bold_niftis_in_dir, run_analysis


def write_nifti(path: Path, data: np.ndarray) -> None:
    """Write a NIfTI file with identity affine.
    Args:
        path (Path): Output path.
        data (np.ndarray): NIfTI data.
    Returns:
        None: This function returns nothing."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def load_stats(path: Path) -> Dict[str, object]:
    """Read stats JSON output.
    Args:
        path (Path): JSON path.
    Returns:
        Dict[str, object]: Parsed payload."""
    return json.loads(path.read_text(encoding="utf-8"))


def make_phantom_data(shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Create synthetic phantom-like 4D data.
    Args:
        shape (Tuple[int, int, int, int]): `(x, y, z, t)` shape.
    Returns:
        np.ndarray: Synthetic data."""
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

    map_path, stats_path, _ = run_analysis(input_path=input_path, mode="phantom")
    assert map_path.name == "phantom_tsnr_map.nii.gz"
    assert stats_path.name == "phantom_tsnr_stats.json"
    image = nib.load(str(map_path))
    assert image.shape == (21, 21, 9)
    stats = load_stats(stats_path)
    assert stats["mode"] == "phantom"
    assert stats["n_voxels_in_roi"] == 225
    assert stats["map_affine_source"] == "input"
    assert isinstance(stats["parameters"], dict)
    assert stats["n_timepoints"] == 7
    assert stats["volume_shape"] == [21, 21, 9, 7]
    ts = stats["timepoint_selection"]
    assert ts["first_index"] == 1
    assert ts["last_index_inclusive"] == 7
    assert ts["n_timepoints_in_file"] == 8
    assert ts["n_timepoints_used"] == 7


def test_find_t1_in_anat_bids_func_under_subject(tmp_path: Path) -> None:
    """T1 is resolved from subject/anat when bold is under subject/func."""
    sub = tmp_path / "sub-01"
    func_dir = sub / "func"
    anat_dir = sub / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)
    bold = func_dir / "sub-01_task-rest_bold.nii.gz"
    t1 = anat_dir / "sub-01_T1w.nii.gz"
    bold.write_bytes(b"")
    t1.write_bytes(b"")
    assert find_t1_in_anat(bold) == t1


def test_find_t1_in_anat_flat_anat_sibling(tmp_path: Path) -> None:
    """When anat/ sits next to the bold file, that directory is used."""
    anat_dir = tmp_path / "anat"
    anat_dir.mkdir()
    t1 = anat_dir / "T1w.nii.gz"
    t1.write_bytes(b"")
    bold = tmp_path / "run.nii.gz"
    bold.write_bytes(b"")
    assert find_t1_in_anat(bold) == t1


def test_find_t1_in_anat_prefers_grandparent_when_both_exist(tmp_path: Path) -> None:
    """BIDS-style grandparent anat wins over parent anat when both exist."""
    sub = tmp_path / "sub-01"
    func_dir = sub / "func"
    anat_grand = sub / "anat"
    anat_parent = func_dir / "anat"
    func_dir.mkdir(parents=True)
    anat_grand.mkdir()
    anat_parent.mkdir()
    t1_grand = anat_grand / "a_T1w.nii.gz"
    t1_parent = anat_parent / "b_T1w.nii.gz"
    t1_grand.write_bytes(b"")
    t1_parent.write_bytes(b"")
    bold = func_dir / "bold.nii.gz"
    bold.write_bytes(b"")
    assert find_t1_in_anat(bold) == t1_grand


def test_find_t1_in_anat_earliest_sidecar_datetime(tmp_path: Path) -> None:
    """When multiple *T1w.nii.gz exist, the earliest AcquisitionDateTime is chosen."""
    ses = tmp_path / "ses-01"
    func_dir = ses / "func"
    anat_dir = ses / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)
    bold = func_dir / "sub-01_ses-01_task-rest_bold.nii.gz"
    bold.write_bytes(b"")
    later = anat_dir / "sub-01_ses-01_run-02_T1w.nii.gz"
    earlier = anat_dir / "sub-01_ses-01_run-01_T1w.nii.gz"
    later.write_bytes(b"")
    earlier.write_bytes(b"")
    (anat_dir / "sub-01_ses-01_run-02_T1w.json").write_text(
        '{"AcquisitionDateTime": "2024-12-02T14:00:00"}',
        encoding="utf-8",
    )
    (anat_dir / "sub-01_ses-01_run-01_T1w.json").write_text(
        '{"AcquisitionDateTime": "2024-12-02T12:00:00"}',
        encoding="utf-8",
    )
    assert find_t1_in_anat(bold) == earlier


def test_list_bold_niftis_in_dir_default_globs_and_sort(tmp_path: Path) -> None:
    """Default globs pick *_bold.nii* only, sorted by name."""
    func = tmp_path / "func"
    func.mkdir()
    (func / "sub-01_task-rest_echo-2_bold.nii.gz").write_bytes(b"")
    (func / "sub-01_task-rest_echo-1_bold.nii.gz").write_bytes(b"")
    (func / "sub-01_task-rest_sbref.nii.gz").write_bytes(b"")
    (func / "sub-01_task-rest_echo-1_bold.nii").write_bytes(b"")
    (func / "events.tsv").write_text("onset\tduration\n", encoding="utf-8")
    paths = list_bold_niftis_in_dir(func)
    assert [p.name for p in paths] == [
        "sub-01_task-rest_echo-1_bold.nii",
        "sub-01_task-rest_echo-1_bold.nii.gz",
        "sub-01_task-rest_echo-2_bold.nii.gz",
    ]


def test_list_bold_niftis_in_dir_custom_pattern(tmp_path: Path) -> None:
    """Custom --input-pattern restricts matches."""
    d = tmp_path / "d"
    d.mkdir()
    (d / "a_rest_run1.nii.gz").write_bytes(b"")
    (d / "b_rest_run2.nii.gz").write_bytes(b"")
    (d / "other_bold.nii.gz").write_bytes(b"")
    paths = list_bold_niftis_in_dir(d, pattern="*_run*.nii.gz")
    assert [p.name for p in paths] == ["a_rest_run1.nii.gz", "b_rest_run2.nii.gz"]


def test_list_bold_niftis_in_dir_requires_directory(tmp_path: Path) -> None:
    """Non-directory path raises ValueError."""
    with pytest.raises(ValueError, match="Not a directory"):
        list_bold_niftis_in_dir(tmp_path / "missing")


def test_cli_directory_batch_phantom(tmp_path: Path) -> None:
    """Directory input runs analysis per BOLD file and writes one stats JSON each."""
    func = tmp_path / "func"
    func.mkdir()
    data = make_phantom_data((11, 11, 3, 4))
    write_nifti(func / "sub-01_task-x_echo-1_bold.nii.gz", data)
    write_nifti(func / "sub-01_task-x_echo-2_bold.nii.gz", data)
    assert (
        cli(
            [
                str(func),
                "phantom",
                "--roi-size",
                "11",
                "--first-timepoint",
                "0",
            ]
        )
        == 0
    )
    assert (func / "sub-01_task-x_echo-1_bold_tsnr_stats.json").is_file()
    assert (func / "sub-01_task-x_echo-2_bold_tsnr_stats.json").is_file()


def test_cli_directory_no_matches_exits_error(tmp_path: Path) -> None:
    """Empty glob results exits with code 1."""
    func = tmp_path / "empty_func"
    func.mkdir()
    assert cli([str(func), "phantom"]) == 1


def test_cli_input_missing_path_exits_error(tmp_path: Path) -> None:
    """Nonexistent input that is neither file nor directory exits 1."""
    assert cli([str(tmp_path / "does_not_exist"), "phantom"]) == 1


def test_extract_brain_tsnr_omits_intensity_block_when_mask_supplied() -> None:
    """When a brain mask array is supplied, intensity_brain_mask is not in parameters."""
    tsnr_map = np.ones((5, 5, 3), dtype=np.float64)
    mean_volume = np.ones((5, 5, 3), dtype=np.float64) * 100.0
    mask = np.zeros((5, 5, 3), dtype=bool)
    mask[1:4, 1:4, 1] = True
    _, params = extract_brain_tsnr(
        tsnr_map,
        mean_volume,
        threshold=0.25,
        erosion_voxels=2,
        brain_mask=mask,
        brain_masking_report={"method": "t1_bet_registered_to_mean_epi"},
    )
    assert "intensity_brain_mask" not in params
    assert params["brain_masking"]["method"] == "t1_bet_registered_to_mean_epi"


def test_find_t1_in_anat_fallback_mtime_when_no_sidecar_time(tmp_path: Path) -> None:
    """Without usable JSON times, the earliest file mtime among T1w candidates wins."""
    import os
    import time

    sub = tmp_path / "sub-01"
    func_dir = sub / "func"
    anat_dir = sub / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)
    bold = func_dir / "bold.nii.gz"
    bold.write_bytes(b"")
    older = anat_dir / "sub-01_acq-A_T1w.nii.gz"
    newer = anat_dir / "sub-01_acq-B_T1w.nii.gz"
    older.write_bytes(b"x")
    newer.write_bytes(b"y")
    old_stamp = time.time() - 10_000.0
    os.utime(older, (old_stamp, old_stamp))
    new_stamp = time.time()
    os.utime(newer, (new_stamp, new_stamp))
    assert find_t1_in_anat(bold) == older


def test_nifti_brain_happy_path(tmp_path: Path) -> None:
    """Brain analysis uses positive-voxel baseline and erosion."""
    data = np.zeros((9, 9, 5, 6), dtype=np.float64)
    data[2:7, 2:7, 1:4, :] = 100.0
    for ti in range(data.shape[3]):
        data[2:7, 2:7, 1:4, ti] += ti
    input_path = tmp_path / "brain.nii.gz"
    write_nifti(input_path, data)

    _, stats_path, _ = run_analysis(
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
    assert params["intensity_brain_mask"]["erosion_voxels"] == 1
    assert params["intensity_brain_mask"]["threshold"] == 0.25
    bm = params["brain_masking"]
    assert bm["method"] == "mean_intensity"
    assert bm["t1_to_functional_pipeline"] == "not_attempted_no_t1"
    assert bm["t1_path"] is None
    assert bm["detail"] is not None


def test_brain_masking_json_when_t1_pipeline_fails(tmp_path: Path) -> None:
    """When T1 exists but FSL pipeline fails, JSON records failed pipeline and fallback."""
    sub = tmp_path / "sub-01"
    func_dir = sub / "func"
    anat_dir = sub / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)
    t1 = anat_dir / "sub-01_T1w.nii.gz"
    t1.write_bytes(b"")
    data = np.zeros((8, 8, 4, 5), dtype=np.float64)
    data[2:6, 2:6, 1:3, :] = 50.0
    bold = func_dir / "sub-01_task-rest_bold.nii.gz"
    write_nifti(bold, data)

    with patch("tsnr.create_bet_mask_for_func", side_effect=ValueError("synthetic FSL failure")):
        _, stats_path, _ = run_analysis(
            input_path=bold,
            mode="brain",
            threshold=0.25,
            erosion_voxels=0,
            first_timepoint=0,
        )
    stats = load_stats(stats_path)
    bm = stats["parameters"]["brain_masking"]
    assert bm["method"] == "mean_intensity"
    assert bm["t1_to_functional_pipeline"] == "failed"
    assert bm["t1_path"] == str(t1.resolve())
    assert "synthetic FSL failure" in (bm["detail"] or "")


def test_zero_std_handling_maps_to_zero(tmp_path: Path) -> None:
    """Constant series yields zero tSNR and finite output."""
    data = np.ones((11, 11, 3, 4), dtype=np.float64) * 10.0
    input_path = tmp_path / "constant.nii.gz"
    write_nifti(input_path, data)

    map_path, stats_path, _ = run_analysis(
        input_path=input_path, mode="phantom", roi_size=11, mask_maps=False
    )
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

    _, stats_path, _ = run_analysis(input_path=input_path, mode="phantom", roi_size=15, slice_index=2)
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
    _, stats_v2, _ = run_analysis(input_path=npz_v2, mode="phantom", roi_size=9)
    payload_v2 = load_stats(stats_v2)
    assert payload_v2["input_type"] == "fmriqa_pixel_cache_npz"
    assert payload_v2["map_affine_source"] == "identity"
    assert payload_v2["n_timepoints"] == 4
    assert payload_v2["timepoint_selection"]["n_timepoints_in_file"] == 5

    npz_v1 = tmp_path / "cache_v1.npz"
    np.savez_compressed(npz_v1, cache_version=np.int32(1), volume=volume)
    _, stats_v1, _ = run_analysis(input_path=npz_v1, mode="phantom", roi_size=9, first_timepoint=0)
    payload_v1 = load_stats(stats_v1)
    assert payload_v1["n_timepoints"] == 5

    with pytest.raises(ValueError, match="brain mode does not accept .npz input"):
        run_analysis(input_path=npz_v2, mode="brain")


def test_output_naming_for_nii_gz(tmp_path: Path) -> None:
    """Double-suffix NIfTI naming strips .nii.gz as one unit."""
    data = make_phantom_data((11, 11, 3, 4))
    input_path = tmp_path / "example.nii.gz"
    write_nifti(input_path, data)
    map_path, stats_path, _ = run_analysis(input_path=input_path, mode="phantom", roi_size=11)
    assert map_path.name == "example_tsnr_map.nii.gz"
    assert stats_path.name == "example_tsnr_stats.json"


def test_json_contract_fields_present(tmp_path: Path) -> None:
    """Stats JSON includes required contract fields."""
    data = make_phantom_data((13, 13, 3, 4))
    input_path = tmp_path / "contract.nii.gz"
    write_nifti(input_path, data)
    _, stats_path, _ = run_analysis(
        input_path=input_path,
        mode="phantom",
        roi_size=13,
        first_timepoint=0,
    )
    stats = load_stats(stats_path)
    assert stats["input_type"] == "nifti"
    assert stats["volume_shape"] == [13, 13, 3, 4]
    assert stats["map_affine_source"] == "input"
    assert "parameters" in stats
    assert "brain_masking" not in stats["parameters"]
    assert stats["timepoint_selection"]["n_timepoints_used"] == 4
    assert stats["output_map_censoring"] == "roi_masked"


def test_brain_json_includes_brain_masking_contract_keys(tmp_path: Path) -> None:
    """Brain mode stats always include parameters.brain_masking with documented keys."""
    data = np.zeros((9, 9, 5, 6), dtype=np.float64)
    data[2:7, 2:7, 1:4, :] = 100.0
    for ti in range(data.shape[3]):
        data[2:7, 2:7, 1:4, ti] += ti
    input_path = tmp_path / "brain_contract.nii.gz"
    write_nifti(input_path, data)
    _, stats_path, _ = run_analysis(
        input_path=input_path,
        mode="brain",
        threshold=0.25,
        erosion_voxels=1,
        first_timepoint=0,
    )
    stats = load_stats(stats_path)
    bm = stats["parameters"]["brain_masking"]
    for key in ("method", "t1_path", "t1_to_functional_pipeline", "detail"):
        assert key in bm


def test_timepoint_all_volumes_explicit_first_zero(tmp_path: Path) -> None:
    """first_timepoint=0 includes every volume (no leading skip)."""
    data = make_phantom_data((11, 11, 3, 4))
    input_path = tmp_path / "all_t.nii.gz"
    write_nifti(input_path, data)
    _, stats_path, _ = run_analysis(input_path=input_path, mode="phantom", roi_size=11, first_timepoint=0)
    stats = load_stats(stats_path)
    assert stats["n_timepoints"] == 4
    assert stats["timepoint_selection"]["first_index"] == 0


def test_default_timepoint_skip_requires_enough_volumes(tmp_path: Path) -> None:
    """Default skip cannot leave fewer than two volumes."""
    data = make_phantom_data((11, 11, 3, 2))
    input_path = tmp_path / "short.nii.gz"
    write_nifti(input_path, data)
    with pytest.raises(ValueError, match="at least 2 volumes are required"):
        run_analysis(input_path=input_path, mode="phantom", roi_size=11)


def test_write_tmean_tstd_optional_maps(tmp_path: Path) -> None:
    """Optional --write-tmean-tstd emits legacy-named mean/std volumes for debugging."""
    data = make_phantom_data((11, 11, 3, 5))
    input_path = tmp_path / "phantom.nii.gz"
    write_nifti(input_path, data)
    expected_mean, expected_std, _ = compute_tsnr_map(data)

    _, _, optional_paths = run_analysis(
        input_path=input_path,
        mode="phantom",
        roi_size=11,
        write_tmean_tstd=True,
        first_timepoint=0,
        mask_maps=False,
    )
    assert [p.name for p in optional_paths] == ["phantom_Tmean.nii.gz", "phantom_Tstd.nii.gz"]
    tmean = nib.load(str(optional_paths[0])).get_fdata(dtype=np.float64)
    tstd = nib.load(str(optional_paths[1])).get_fdata(dtype=np.float64)
    # Outputs are float32 on disk; allow small round-trip drift vs float64 reference.
    np.testing.assert_allclose(tmean, expected_mean, rtol=0, atol=1e-5)
    np.testing.assert_allclose(tstd, expected_std, rtol=0, atol=1e-5)


def test_cli_main_writes_outputs(tmp_path: Path) -> None:
    """Running ``python tsnr.py`` must invoke the CLI (not exit after import-only)."""
    data = make_phantom_data((11, 11, 3, 5))
    input_path = tmp_path / "cli_phantom.nii.gz"
    write_nifti(input_path, data)
    script = Path(__file__).resolve().parents[1] / "tsnr.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(input_path),
            "phantom",
            "--output-dir",
            str(tmp_path),
            "--roi-size",
            "11",
            "--first-timepoint",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "cli_phantom_tsnr_map.nii.gz").is_file()
    stats_path = tmp_path / "cli_phantom_tsnr_stats.json"
    assert stats_path.is_file()
    assert load_stats(stats_path)["output_map_censoring"] == "roi_masked"


def test_masked_maps_match_roi_voxel_count_phantom(tmp_path: Path) -> None:
    """Default map masking leaves NaNs only outside the phantom ROI on disk."""
    data = make_phantom_data((11, 11, 3, 5))
    input_path = tmp_path / "phantom.nii.gz"
    write_nifti(input_path, data)
    map_path, stats_path, _ = run_analysis(
        input_path=input_path, mode="phantom", roi_size=11, first_timepoint=0
    )
    stats = load_stats(stats_path)
    assert stats["output_map_censoring"] == "roi_masked"
    vox = np.asarray(nib.load(str(map_path)).get_fdata(dtype=np.float64))
    assert int(np.sum(np.isfinite(vox))) == int(stats["n_voxels_in_roi"])


def test_full_fov_maps_cli_flag(tmp_path: Path) -> None:
    """--full-fov-maps writes uncensored volumes and records it in JSON."""
    data = make_phantom_data((11, 11, 3, 5))
    input_path = tmp_path / "p.nii.gz"
    write_nifti(input_path, data)
    script = Path(__file__).resolve().parents[1] / "tsnr.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(input_path),
            "phantom",
            "--output-dir",
            str(tmp_path),
            "--roi-size",
            "11",
            "--first-timepoint",
            "0",
            "--full-fov-maps",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert load_stats(tmp_path / "p_tsnr_stats.json")["output_map_censoring"] == "full_fov"
