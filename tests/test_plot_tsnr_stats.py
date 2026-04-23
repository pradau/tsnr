# Author: Perry Radau
# Date: 2026-04-20
# Validate plotting and aggregation for tSNR stats reports.
# Dependencies: Python 3.10+, pytest, matplotlib, numpy
# Usage: uv run pytest tests/test_plot_tsnr_stats.py

"""
Tests for the standalone tSNR stats plotting script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from plot_tsnr_stats import (
    default_out_dir_for_phantom_stats_dir,
    discover_phantom_stats_files,
    _count_abs_robust_z_gt_4,
    cli as plot_cli,
    aggregate_metric_rows,
    discover_session_stats_files,
    discover_stats_files,
    discover_subject_sessions,
    filter_rows,
    load_metric_rows,
    parse_bids_entities_from_name,
    plot_roi_mean_signal_tr_session_grid,
    plot_robust_z_tr_session_grid,
    roi_mean_signal_series_from_spike_block,
    run_report,
    resolve_phantom_session_label,
    tr_plot_z_from_spike_block,
)


def _write_stats(
    path: Path,
    tsnr_mean: float,
    tsnr_std: float,
    ftsnr: float,
    roi_std: float,
    *,
    max_abs_robust_z: float = 2.0,
    pct_tr_abs_robust_z_gt_4: float = 0.0,
    n_tr_abs_robust_z_gt_4: float = 0.0,
    robust_z_len: int = 12,
    include_slice_metrics: bool = True,
    worst_slice_spike_pct_tr_abs_robust_z_gt_4: float = 12.5,
    worst_slice_spike_max_abs_robust_z: float = 6.2,
    worst_slice_spike_pct_slice_index: int = 4,
    worst_slice_spike_max_abs_slice_index: int = 7,
    n_slices_with_roi: int = 24,
    n_slices_eligible: int = 18,
) -> None:
    """Write a minimal stats payload used by plotting code.
    Args:
        path (Path): Output JSON path.
        tsnr_mean (float): Mean tSNR value.
        tsnr_std (float): Spatial tSNR spread.
        ftsnr (float): Functional tSNR.
        roi_std (float): ROI mean temporal std.
        max_abs_robust_z (float): Spike metric for tests.
        pct_tr_abs_robust_z_gt_4 (float): Spike metric for tests.
        n_tr_abs_robust_z_gt_4 (float): Spike metric for tests.
    Returns:
        None: This function returns nothing.
    """
    payload: Dict[str, object] = {
        "tsnr_mean": tsnr_mean,
        "tsnr_std": tsnr_std,
        "ftsnr": ftsnr,
        "roi_mean_signal_std": roi_std,
        "roi_mean_tr_spike_metrics": {
            "max_abs_robust_z": max_abs_robust_z,
            "pct_tr_abs_robust_z_gt_4": pct_tr_abs_robust_z_gt_4,
            "n_tr_abs_robust_z_gt_4": n_tr_abs_robust_z_gt_4,
            "robust_z_per_tr": [0.1 * float(i % 5) for i in range(robust_z_len)],
            "roi_mean_signal_per_tr": [1000.0 + 0.5 * float(i) for i in range(robust_z_len)],
        },
    }
    if include_slice_metrics:
        payload["slice_ftsnr_metrics"] = {
            "worst_slice_spike_pct_tr_abs_robust_z_gt_4": float(
                worst_slice_spike_pct_tr_abs_robust_z_gt_4
            ),
            "worst_slice_spike_max_abs_robust_z": float(worst_slice_spike_max_abs_robust_z),
            "worst_slice_spike_pct_slice_index": int(worst_slice_spike_pct_slice_index),
            "worst_slice_spike_max_abs_slice_index": int(worst_slice_spike_max_abs_slice_index),
            "n_slices_with_roi": int(n_slices_with_roi),
            "n_slices_eligible": int(n_slices_eligible),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset(tmp_path: Path) -> Path:
    """Create a synthetic BIDS derivatives tree with stats JSON files."""
    root = tmp_path / "bids"
    files: List[Path] = [
        root
        / "sub-01/ses-1a/derivatives/tsnr/sub-01_ses-1a_task-rest_echo-1_bold_tsnr_stats.json",
        root
        / "sub-01/ses-1a/derivatives/tsnr/sub-01_ses-1a_task-rest_echo-2_bold_tsnr_stats.json",
        root
        / "sub-02/ses-1a/derivatives/tsnr/sub-02_ses-1a_task-rest_echo-1_bold_tsnr_stats.json",
        root
        / "sub-02/ses-1a/derivatives/tsnr/sub-02_ses-1a_task-rest_echo-2_bold_tsnr_stats.json",
        root
        / "sub-01/ses-1b/derivatives/tsnr/sub-01_ses-1b_task-rest_echo-1_bold_tsnr_stats.json",
        root
        / "sub-01/ses-1b/derivatives/tsnr/sub-01_ses-1b_task-rest_echo-2_bold_tsnr_stats.json",
    ]
    values = [
        (80.0, 10.0, 120.0, 0.85, 2.1, 0.5, 1.0),
        (70.0, 11.0, 110.0, 0.90, 2.2, 1.0, 2.0),
        (84.0, 9.0, 124.0, 0.88, 2.3, 0.0, 0.0),
        (72.0, 12.0, 111.0, 0.93, 2.4, 2.0, 3.0),
        (90.0, 8.0, 130.0, 0.80, 2.0, 0.0, 0.0),
        (86.0, 7.0, 127.0, 0.82, 2.5, 1.5, 2.0),
    ]
    for path, metric_vals in zip(files, values):
        tsnr_m, tsnr_s, ft, roi, mz, pct, nt = metric_vals
        _write_stats(
            path,
            tsnr_m,
            tsnr_s,
            ft,
            roi,
            max_abs_robust_z=mz,
            pct_tr_abs_robust_z_gt_4=pct,
            n_tr_abs_robust_z_gt_4=nt,
        )
    return root


def test_cli_rejects_conflicting_error_bar_flags() -> None:
    """CLI exits with an error when --no-error-bars and --show-error-bars are both set."""
    code = plot_cli(["--bids-root", "/nonexistent-path-for-cli-test", "--no-error-bars", "--show-error-bars"])
    assert code == 1


def test_roi_mean_signal_series_from_spike_block() -> None:
    """Raw per-TR series is returned when present in spike block."""
    assert roi_mean_signal_series_from_spike_block({}) is None
    assert roi_mean_signal_series_from_spike_block({"roi_mean_signal_per_tr": [1.0, 2.0]}) == [1.0, 2.0]


def test_count_abs_robust_z_gt_4() -> None:
    """Outlier count matches |z| > 4 rule."""
    assert _count_abs_robust_z_gt_4([0.0, 4.0, 4.01, -4.001]) == 2
    assert _count_abs_robust_z_gt_4([0.0, 3.0, 3.5, -3.5]) == 0
    assert _count_abs_robust_z_gt_4([]) == 0


def test_tr_plot_z_linear_ramp_near_zero_after_detrend() -> None:
    """TR-index plot z is near flat when ROI mean is a pure linear ramp."""
    n = 30
    raw = [1000.0 + 2.5 * float(i) for i in range(n)]
    spike = {"roi_mean_signal_per_tr": raw, "robust_z_per_tr": [0.0] * n}
    z = tr_plot_z_from_spike_block(spike)
    assert z is not None
    assert float(np.max(np.abs(np.asarray(z, dtype=np.float64)))) < 1e-3


def test_parse_bids_entities_from_name_happy_path() -> None:
    """Parses sub/session/task/echo entities from stats file names."""
    path = Path("sub-01_ses-1a_task-rest_echo-3_bold_tsnr_stats.json")
    entities = parse_bids_entities_from_name(path)
    assert entities == {
        "sub": "sub-01",
        "ses": "ses-1a",
        "task": "task-rest",
        "echo": "echo-3",
    }


def test_parse_bids_entities_from_name_missing_echo_raises() -> None:
    """Missing required entity produces a clear exception."""
    with pytest.raises(ValueError, match="Missing echo entity"):
        parse_bids_entities_from_name(Path("sub-01_ses-1a_task-rest_bold_tsnr_stats.json"))


def test_discover_load_and_aggregate(tmp_path: Path) -> None:
    """Discovery and aggregation compute expected means and counts."""
    root = _make_dataset(tmp_path)
    stats = discover_stats_files(root)
    assert len(stats) == 6
    rows = load_metric_rows(stats)
    assert len(rows) == 6
    agg = aggregate_metric_rows(rows, metric="tsnr_mean", error_mode="sem", group_by_task=False)
    ses1a_echo1 = [row for row in agg if row["ses"] == "ses-1a" and row["echo"] == "echo-1"][0]
    assert ses1a_echo1["n_runs"] == 2
    assert ses1a_echo1["tsnr_mean_mean"] == pytest.approx(82.0)
    assert float(ses1a_echo1["tsnr_mean_error"]) > 0.0


def test_filter_rows_subject_and_session(tmp_path: Path) -> None:
    """Subject and session filters keep expected runs only."""
    root = _make_dataset(tmp_path)
    rows = load_metric_rows(discover_stats_files(root))
    filtered = filter_rows(rows, subject="sub-01", session="ses-1a")
    assert len(filtered) == 2
    assert {str(row["task"]) for row in filtered} == {"task-rest"}
    assert {str(row["echo"]) for row in filtered} == {"echo-1", "echo-2"}


def test_run_report_creates_png_and_csv(tmp_path: Path) -> None:
    """Full report generation writes expected files."""
    root = _make_dataset(tmp_path)
    out_dir = tmp_path / "reports" / "tsnr_plots"
    generated = run_report(root, out_dir=out_dir, error_mode="ci95", group_by_task=False)
    assert len(generated) == 3
    expected = [
        out_dir / "metrics_panel_by_echo_ci95.png",
        out_dir / "slice_metrics_panel_by_echo_ci95.png",
        out_dir / "aggregated_metric_summary.csv",
    ]
    for path in expected:
        assert path.exists()
        assert path in generated
    assert (out_dir / "spike_metrics_panel_by_echo_ci95.png").exists() is False
    csv_text = (out_dir / "aggregated_metric_summary.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert header == "metric,sub,ses,echo,task,n_runs,mean,error"


def test_run_report_spike_metrics_panels_opt_in(tmp_path: Path) -> None:
    """ROI spike figure and CSV rows are produced only when spike_metrics_panels is True."""
    root = _make_dataset(tmp_path)
    out_dir = tmp_path / "reports" / "spike_on"
    generated = run_report(
        root,
        out_dir=out_dir,
        error_mode="ci95",
        group_by_task=False,
        spike_metrics_panels=True,
    )
    assert out_dir / "spike_metrics_panel_by_echo_ci95.png" in generated
    assert (out_dir / "spike_metrics_panel_by_echo_ci95.png").exists()
    text = (out_dir / "aggregated_metric_summary.csv").read_text(encoding="utf-8")
    assert any(line.startswith("max_abs_robust_z,") for line in text.splitlines()[1:])


def test_run_report_filters_for_task_curves(tmp_path: Path) -> None:
    """Filtered report supports one line per task for one subject/session."""
    root = tmp_path / "bids"
    _write_stats(
        root / "sub-3334/ses-1a/derivatives/tsnr/sub-3334_ses-1a_task-laluna_echo-1_bold_tsnr_stats.json",
        40.0,
        7.0,
        130.0,
        0.8,
        max_abs_robust_z=3.0,
        pct_tr_abs_robust_z_gt_4=1.0,
        n_tr_abs_robust_z_gt_4=2.0,
    )
    _write_stats(
        root / "sub-3334/ses-1a/derivatives/tsnr/sub-3334_ses-1a_task-laluna_echo-2_bold_tsnr_stats.json",
        35.0,
        8.0,
        120.0,
        0.9,
        max_abs_robust_z=3.1,
        pct_tr_abs_robust_z_gt_4=2.0,
        n_tr_abs_robust_z_gt_4=3.0,
    )
    _write_stats(
        root / "sub-3334/ses-1a/derivatives/tsnr/sub-3334_ses-1a_task-partlycloudy_echo-1_bold_tsnr_stats.json",
        32.0,
        6.0,
        115.0,
        1.0,
        max_abs_robust_z=4.0,
        pct_tr_abs_robust_z_gt_4=0.5,
        n_tr_abs_robust_z_gt_4=1.0,
    )
    _write_stats(
        root / "sub-3334/ses-1a/derivatives/tsnr/sub-3334_ses-1a_task-partlycloudy_echo-2_bold_tsnr_stats.json",
        29.0,
        6.5,
        110.0,
        1.1,
        max_abs_robust_z=4.1,
        pct_tr_abs_robust_z_gt_4=1.5,
        n_tr_abs_robust_z_gt_4=2.0,
    )
    out_dir = tmp_path / "reports" / "tsnr_plots"
    generated = run_report(
        root,
        out_dir=out_dir,
        error_mode="sem",
        group_by_task=True,
        subject="sub-3334",
        session="ses-1a",
        show_error_bars=False,
    )
    csv_path = out_dir / "aggregated_metric_summary.csv"
    assert csv_path.exists() and csv_path in generated
    text = csv_path.read_text(encoding="utf-8")
    assert "task-laluna" in text
    assert "task-partlycloudy" in text
    assert "ftsnr,sub-3334,ses-1a,echo-1,task-laluna," in text
    assert not any(
        line.split(",", 1)[0] == "max_abs_robust_z" for line in text.splitlines()[1:]
    )
    assert "worst_slice_spike_pct_tr_abs_robust_z_gt_4,sub-3334,ses-1a,echo-1,task-laluna," in text
    assert "worst_slice_spike_max_abs_slice_index,sub-3334,ses-1a,echo-1,task-laluna," in text
    assert (out_dir / "spike_metrics_panel_by_echo_sem.png").exists() is False
    assert (out_dir / "slice_metrics_panel_by_echo_sem.png").exists()


def test_run_report_skips_slice_plots_when_missing_slice_metrics(tmp_path: Path) -> None:
    """Older stats without slice block still produce core plots and CSV."""
    root = tmp_path / "bids"
    _write_stats(
        root / "sub-01/ses-1a/derivatives/tsnr/sub-01_ses-1a_task-rest_echo-1_bold_tsnr_stats.json",
        80.0,
        10.0,
        120.0,
        0.85,
        include_slice_metrics=False,
    )
    _write_stats(
        root / "sub-01/ses-1a/derivatives/tsnr/sub-01_ses-1a_task-rest_echo-2_bold_tsnr_stats.json",
        70.0,
        11.0,
        110.0,
        0.90,
        include_slice_metrics=False,
    )
    out_dir = tmp_path / "reports" / "tsnr_plots"
    generated = run_report(root, out_dir=out_dir, error_mode="sem", group_by_task=False)
    assert (out_dir / "metrics_panel_by_echo_sem.png") in generated
    assert (out_dir / "spike_metrics_panel_by_echo_sem.png") not in generated
    assert (out_dir / "aggregated_metric_summary.csv") in generated
    assert (out_dir / "slice_metrics_panel_by_echo_sem.png") not in generated


def test_discover_subject_sessions_sorted_pairs(tmp_path: Path) -> None:
    """Unique subject/session pairs match BIDS layout under the dataset root."""
    root = _make_dataset(tmp_path)
    pairs = discover_subject_sessions(root)
    assert pairs == [
        ("sub-01", "ses-1a"),
        ("sub-01", "ses-1b"),
        ("sub-02", "ses-1a"),
    ]


def test_discover_session_stats_files_lists_runs(tmp_path: Path) -> None:
    """Session discovery returns sorted stats paths for one subject/session."""
    root = _make_dataset(tmp_path)
    paths = discover_session_stats_files(root, "sub-01", "ses-1a")
    assert len(paths) == 2
    assert all(p.name.endswith("_tsnr_stats.json") for p in paths)


def test_plot_roi_mean_signal_tr_session_grid_writes_png(tmp_path: Path) -> None:
    """ROI mean signal vs TR figure is written when roi_mean_signal_per_tr is present."""
    root = _make_dataset(tmp_path)
    session_paths = discover_session_stats_files(root, "sub-01", "ses-1a")
    out_png = tmp_path / "sig.png"
    assert plot_roi_mean_signal_tr_session_grid(session_paths, out_png, subject="sub-01", session="ses-1a")
    assert out_png.is_file()


def test_plot_robust_z_tr_session_grid_writes_png(tmp_path: Path) -> None:
    """Multi-panel robust z vs TR figure is written when series are present."""
    root = _make_dataset(tmp_path)
    session_paths = discover_session_stats_files(root, "sub-01", "ses-1a")
    out_png = tmp_path / "rz.png"
    assert plot_robust_z_tr_session_grid(session_paths, out_png, subject="sub-01", session="ses-1a")
    assert out_png.is_file()


def test_run_report_with_robust_z_tr_panels(tmp_path: Path) -> None:
    """Optional robust-z grid is emitted with --robust-z-tr-panels semantics."""
    root = _make_dataset(tmp_path)
    out_dir = tmp_path / "out"
    generated = run_report(
        root,
        out_dir=out_dir,
        error_mode="sem",
        group_by_task=False,
        subject="sub-01",
        session="ses-1a",
        robust_z_tr_panels=True,
    )
    assert out_dir / "robust_z_vs_tr_sub-01_ses-1a.png" in generated
    assert (out_dir / "robust_z_vs_tr_sub-01_ses-1a.png").exists()


def test_run_report_with_roi_mean_signal_tr_panels(tmp_path: Path) -> None:
    """Optional ROI mean signal vs TR grid is emitted when requested."""
    root = _make_dataset(tmp_path)
    out_dir = tmp_path / "out_roi"
    generated = run_report(
        root,
        out_dir=out_dir,
        error_mode="sem",
        group_by_task=False,
        subject="sub-01",
        session="ses-1a",
        roi_mean_signal_tr_panels=True,
    )
    assert out_dir / "roi_mean_signal_vs_tr_sub-01_ses-1a.png" in generated
    assert (out_dir / "roi_mean_signal_vs_tr_sub-01_ses-1a.png").exists()


def test_resolve_phantom_session_label_prefers_metadata_date() -> None:
    """Metadata date key wins over filename-derived date in auto mode."""
    payload: Dict[str, object] = {
        "qa_session_date": "2026-04-02",
        "input_file": "/tmp/fMRIQASnap_2024_03_11__E1.npz",
    }
    label = resolve_phantom_session_label(
        payload,
        stats_path=Path("run_tsnr_stats.json"),
        label_by="auto",
    )
    assert label == "2026-04-02"


def test_discover_phantom_stats_and_default_out_dir(tmp_path: Path) -> None:
    """Phantom discovery and dataset-local report defaults follow layout rules."""
    stats_dir = tmp_path / "dataset" / "derivatives" / "tsnr"
    stats_dir.mkdir(parents=True)
    p1 = stats_dir / "a_tsnr_stats.json"
    p1.write_text("{}", encoding="utf-8")
    assert discover_phantom_stats_files(stats_dir) == [p1]
    assert default_out_dir_for_phantom_stats_dir(stats_dir) == (
        tmp_path / "dataset" / "reports" / "tsnr_plots"
    )


def test_run_report_phantom_mode_generates_outputs(tmp_path: Path) -> None:
    """Phantom mode reads direct stats directory and writes QA-session panels."""
    stats_dir = tmp_path / "dataset" / "derivatives" / "tsnr"
    stats_dir.mkdir(parents=True)
    _write_stats(
        stats_dir / "AlbertaChildrensHospital_Basic_fMRIQASnap_2024_03_11__E52718S1_tsnr_stats.json",
        80.0,
        10.0,
        120.0,
        0.85,
        max_abs_robust_z=2.1,
        pct_tr_abs_robust_z_gt_4=0.5,
        n_tr_abs_robust_z_gt_4=1.0,
    )
    _write_stats(
        stats_dir / "AlbertaChildrensHospital_Basic_fMRIQASnap_2026_04_02__E53214S1_tsnr_stats.json",
        70.0,
        11.0,
        110.0,
        0.90,
        max_abs_robust_z=2.2,
        pct_tr_abs_robust_z_gt_4=1.0,
        n_tr_abs_robust_z_gt_4=2.0,
    )
    out_dir = default_out_dir_for_phantom_stats_dir(stats_dir)
    generated = run_report(
        bids_root=None,
        out_dir=out_dir,
        error_mode="sem",
        group_by_task=False,
        phantom_stats_dir=stats_dir,
        label_by="filename_date",
    )
    assert out_dir / "metrics_panel_by_session_sem.png" in generated
    assert out_dir / "spike_metrics_panel_by_session_sem.png" not in generated
    assert out_dir / "slice_metrics_panel_by_session_sem.png" in generated
    assert out_dir / "aggregated_metric_summary.csv" in generated
    csv_text = (out_dir / "aggregated_metric_summary.csv").read_text(encoding="utf-8")
    assert "2024-03-11" in csv_text
    assert "2026-04-02" in csv_text
