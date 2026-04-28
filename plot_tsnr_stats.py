# Author: Perry Radau
# Date: 2026-04-20
# Build comparison plots for tSNR/fTSNR metrics across BIDS sessions and echoes.
# Dependencies: Python 3.10+, matplotlib, numpy
# Usage: uv run plot_tsnr_stats.py --bids-root /path/to/bids

"""
Plot tSNR-derived statistics from existing derivatives JSON outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS: Tuple[str, ...] = ("ftsnr", "roi_mean_signal_std", "tsnr_mean")
SLICE_METRICS: Tuple[str, ...] = (
    "worst_slice_spike_pct_tr_abs_robust_z_gt_4",
    "worst_slice_spike_max_abs_robust_z",
)
SLICE_INDEX_METRICS: Tuple[str, ...] = (
    "worst_slice_spike_pct_slice_index",
    "worst_slice_spike_max_abs_slice_index",
)
SLICE_NON_SCALAR_KEYS: Tuple[str, ...] = (
    "axis",
    "eligibility_rule",
    "per_slice",
    "slice_spike_abs_z_threshold",
)
SPIKE_METRICS: Tuple[str, ...] = (
    "max_abs_robust_z",
    "pct_tr_abs_robust_z_gt_4",
    "n_tr_abs_robust_z_gt_4",
)
METRIC_YLABEL: Dict[str, str] = {
    "ftsnr": "ftsnr",
    "roi_mean_signal_std": "roi_mean_signal_std",
    "tsnr_mean": "tsnr_mean",
    "max_abs_robust_z": "max |robust z| (ROI mean TR)",
    "pct_tr_abs_robust_z_gt_4": "% TRs with |robust z| > 4",
    "n_tr_abs_robust_z_gt_4": "TR count with |robust z| > 4",
    "worst_slice_spike_pct_tr_abs_robust_z_gt_4": "worst-slice % TRs with |robust z| > 4",
    "worst_slice_spike_max_abs_robust_z": "worst-slice max |robust z|",
    "worst_slice_spike_pct_slice_index": "worst spike-rate slice index (Z)",
    "worst_slice_spike_max_abs_slice_index": "max |robust z| slice index (Z)",
    "n_slices_with_roi": "slice count with ROI support",
    "n_slices_eligible": "eligible slice count",
}
ERROR_BAR_CHOICES: Tuple[str, ...] = ("sd", "sem", "ci95")
LABEL_BY_CHOICES: Tuple[str, ...] = ("auto", "metadata_date", "filename_date")


def parse_bids_entities_from_name(stats_path: Path) -> Dict[str, str]:
    """Parse core BIDS entities from a stats filename.
    Args:
        stats_path (Path): Path like ``sub-01_ses-1_task-rest_echo-2_bold_tsnr_stats.json``.
    Returns:
        Dict[str, str]: Parsed ``sub``, ``ses``, ``task``, and ``echo``.
    Raises:
        ValueError: If required entities are missing.
    """
    name = stats_path.name
    base = re.sub(r"_tsnr_stats\.json$", "", name)
    patterns = {
        "sub": r"(sub-[^_]+)",
        "ses": r"(ses-[^_]+)",
        "task": r"(task-[^_]+)",
        "echo": r"(echo-[^_]+)",
    }
    out: Dict[str, str] = {}
    for key, pat in patterns.items():
        match = re.search(pat, base)
        if match is None:
            raise ValueError(f"Missing {key} entity in stats filename: {name}")
        out[key] = match.group(1)
    return out


def discover_stats_files(bids_root: Path) -> List[Path]:
    """Find tSNR stats JSON files in BIDS derivatives folders.
    Args:
        bids_root (Path): BIDS dataset root.
    Returns:
        List[Path]: Sorted stats paths.
    Raises:
        ValueError: If root is invalid.
    """
    if not bids_root.is_dir():
        raise ValueError(f"--bids-root is not a directory: {bids_root}")
    paths = sorted(bids_root.glob("sub-*/ses-*/derivatives/tsnr/*_tsnr_stats.json"))
    return [p for p in paths if p.is_file()]


def discover_non_bids_stats_files(stats_dir: Path) -> List[Path]:
    """Find non-BIDS stats JSON files in one directory (non-recursive).
    Args:
        stats_dir (Path): Directory containing ``*_tsnr_stats.json`` files.
    Returns:
        List[Path]: Sorted stats paths.
    Raises:
        ValueError: If ``stats_dir`` is not a directory.
    """
    if not stats_dir.is_dir():
        raise ValueError(f"--stats-dir is not a directory: {stats_dir}")
    return sorted(p for p in stats_dir.glob("*_tsnr_stats.json") if p.is_file())


def default_out_dir_for_stats_dir(stats_dir: Path) -> Path:
    """Default non-BIDS report directory, dataset-local when possible.
    Args:
        stats_dir (Path): Directory used with ``--stats-dir``.
    Returns:
        Path: Default report output directory.
    """
    if stats_dir.name == "tsnr" and stats_dir.parent.name == "derivatives":
        return stats_dir.parent.parent / "reports" / "tsnr_plots"
    return stats_dir.parent / "reports" / "tsnr_plots"


def _date_from_text(text: str) -> Optional[str]:
    """Extract ``YYYY-MM-DD`` from common ``YYYY_MM_DD``/``YYYY-MM-DD`` fragments.
    Args:
        text (str): Candidate text containing a date-like token.
    Returns:
        Optional[str]: Normalized date string, or ``None``.
    """
    match = re.search(r"(20\d{2})[_-](\d{2})[_-](\d{2})", text)
    if match is None:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _metadata_date_from_payload(payload: Dict[str, object]) -> Optional[str]:
    """Best-effort QA date from stats JSON metadata fields.
    Args:
        payload (Dict[str, object]): One stats JSON payload.
    Returns:
        Optional[str]: Normalized ``YYYY-MM-DD`` date, or ``None``.
    """
    direct_keys = ("qa_session_date", "acquisition_date", "session_date")
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str):
            parsed = _date_from_text(value)
            if parsed is not None:
                return parsed
    qa_meta = payload.get("qa_metadata")
    if isinstance(qa_meta, dict):
        for key in ("session_date", "acquisition_date", "snapshot_date"):
            value = qa_meta.get(key)
            if isinstance(value, str):
                parsed = _date_from_text(value)
                if parsed is not None:
                    return parsed
    return None


def resolve_non_bids_session_label(
    payload: Dict[str, object],
    stats_path: Path,
    label_by: str,
) -> str:
    """Resolve plotting session label for non-BIDS rows.
    Args:
        payload (Dict[str, object]): Stats JSON payload.
        stats_path (Path): Stats JSON path.
        label_by (str): One of ``auto``, ``metadata_date``, ``filename_date``.
    Returns:
        str: Label used as the session key.
    """
    metadata_date = _metadata_date_from_payload(payload)
    input_file = payload.get("input_file")
    input_name = Path(str(input_file)).name if isinstance(input_file, str) else ""
    filename_date = _date_from_text(input_name) or _date_from_text(stats_path.name)

    if label_by == "metadata_date":
        if metadata_date is not None:
            return metadata_date
        if filename_date is not None:
            print(f"Warning: metadata date missing in {stats_path}; using filename date {filename_date}.")
            return filename_date
    elif label_by == "filename_date":
        if filename_date is not None:
            return filename_date
    else:
        if metadata_date is not None:
            return metadata_date
        if filename_date is not None:
            return filename_date

    fallback = re.sub(r"_tsnr_stats\\.json$", "", stats_path.name)
    print(f"Warning: could not resolve QA session date in {stats_path}; using {fallback}.")
    return fallback


def discover_subject_sessions(bids_root: Path) -> List[Tuple[str, str]]:
    """List unique (sub, ses) pairs that have at least one tSNR stats JSON.
    Args:
        bids_root (Path): BIDS dataset root.
    Returns:
        List[Tuple[str, str]]: Sorted ``(sub-*, ses-*)`` pairs.
    """
    root = bids_root.resolve()
    pairs: set[Tuple[str, str]] = set()
    for p in discover_stats_files(bids_root):
        rel = p.relative_to(root)
        pairs.add((rel.parts[0], rel.parts[1]))
    return sorted(pairs)


def _read_json(path: Path) -> Dict[str, object]:
    """Read one stats JSON payload.
    Args:
        path (Path): JSON file path.
    Returns:
        Dict[str, object]: Parsed dictionary payload.
    """
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Stats JSON is not an object: {path}")
    return loaded


def load_metric_rows(stats_paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Load and validate rows used for plotting and aggregation.
    Args:
        stats_paths (Sequence[Path]): Stats JSON file paths.
    Returns:
        List[Dict[str, object]]: One row per valid run.
    """
    rows: List[Dict[str, object]] = []
    for stats_path in stats_paths:
        try:
            entities = parse_bids_entities_from_name(stats_path)
            payload = _read_json(stats_path)
            row: Dict[str, object] = {
                "stats_path": str(stats_path),
                "sub": entities["sub"],
                "ses": entities["ses"],
                "task": entities["task"],
                "echo": entities["echo"],
            }
            for metric in METRICS:
                value = payload.get(metric)
                if value is None:
                    raise ValueError(f"Missing required metric {metric} in {stats_path}")
                fv = float(value)
                if not math.isfinite(fv):
                    raise ValueError(f"Non-finite metric {metric} in {stats_path}")
                row[metric] = fv
            tsnr_std = payload.get("tsnr_std")
            if tsnr_std is None:
                raise ValueError(f"Missing required metric tsnr_std in {stats_path}")
            tsnr_std_f = float(tsnr_std)
            if not math.isfinite(tsnr_std_f):
                raise ValueError(f"Non-finite metric tsnr_std in {stats_path}")
            row["tsnr_std"] = tsnr_std_f
            spike_block = payload.get("roi_mean_tr_spike_metrics")
            if isinstance(spike_block, dict):
                missing = [k for k in SPIKE_METRICS if spike_block.get(k) is None]
                if not missing:
                    for key in SPIKE_METRICS:
                        fv = float(spike_block[key])
                        if not math.isfinite(fv):
                            raise ValueError(f"Non-finite spike metric {key} in {stats_path}")
                        row[key] = fv
                    row["has_spike_metrics"] = True
                else:
                    row["has_spike_metrics"] = False
            else:
                row["has_spike_metrics"] = False
            slice_block = payload.get("slice_ftsnr_metrics")
            if isinstance(slice_block, dict):
                missing_slice_keys = [k for k in SLICE_METRICS if slice_block.get(k) is None]
                if not missing_slice_keys:
                    for key in SLICE_METRICS:
                        fv = float(slice_block[key])
                        if not math.isfinite(fv):
                            raise ValueError(f"Non-finite slice metric {key} in {stats_path}")
                        row[key] = fv
                    for key, value in slice_block.items():
                        if key in SLICE_METRICS or key in SLICE_NON_SCALAR_KEYS or isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            fv = float(value)
                            if not math.isfinite(fv):
                                raise ValueError(f"Non-finite slice metric {key} in {stats_path}")
                            row[key] = fv
                    row["has_slice_metrics"] = True
                else:
                    row["has_slice_metrics"] = False
            else:
                row["has_slice_metrics"] = False
            rows.append(row)
        except (ValueError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Warning: skipping {stats_path}: {exc}")
    return rows


def load_non_bids_metric_rows(
    stats_paths: Sequence[Path],
    label_by: str,
) -> List[Dict[str, object]]:
    """Load non-BIDS rows for QA-session comparisons.
    Args:
        stats_paths (Sequence[Path]): Non-BIDS stats JSON paths.
        label_by (str): Label source mode (``auto``, ``metadata_date``, ``filename_date``).
    Returns:
        List[Dict[str, object]]: Rows shaped for existing aggregation helpers.
    """
    rows: List[Dict[str, object]] = []
    for stats_path in stats_paths:
        try:
            payload = _read_json(stats_path)
            qa_session = resolve_non_bids_session_label(payload, stats_path=stats_path, label_by=label_by)
            run_id = re.sub(r"_tsnr_stats\\.json$", "", stats_path.name)
            row: Dict[str, object] = {
                "stats_path": str(stats_path),
                "sub": "non-bids",
                "ses": "session-series",
                "task": run_id,
                "echo": qa_session,
                "qa_session": qa_session,
                "run_id": run_id,
            }
            for metric in METRICS:
                value = payload.get(metric)
                if value is None:
                    raise ValueError(f"Missing required metric {metric} in {stats_path}")
                fv = float(value)
                if not math.isfinite(fv):
                    raise ValueError(f"Non-finite metric {metric} in {stats_path}")
                row[metric] = fv
            tsnr_std = payload.get("tsnr_std")
            if tsnr_std is None:
                raise ValueError(f"Missing required metric tsnr_std in {stats_path}")
            tsnr_std_f = float(tsnr_std)
            if not math.isfinite(tsnr_std_f):
                raise ValueError(f"Non-finite metric tsnr_std in {stats_path}")
            row["tsnr_std"] = tsnr_std_f
            spike_block = payload.get("roi_mean_tr_spike_metrics")
            if isinstance(spike_block, dict):
                missing = [k for k in SPIKE_METRICS if spike_block.get(k) is None]
                if not missing:
                    for key in SPIKE_METRICS:
                        fv = float(spike_block[key])
                        if not math.isfinite(fv):
                            raise ValueError(f"Non-finite spike metric {key} in {stats_path}")
                        row[key] = fv
                    row["has_spike_metrics"] = True
                else:
                    row["has_spike_metrics"] = False
            else:
                row["has_spike_metrics"] = False
            slice_block = payload.get("slice_ftsnr_metrics")
            if isinstance(slice_block, dict):
                missing_slice_keys = [k for k in SLICE_METRICS if slice_block.get(k) is None]
                if not missing_slice_keys:
                    for key in SLICE_METRICS:
                        fv = float(slice_block[key])
                        if not math.isfinite(fv):
                            raise ValueError(f"Non-finite slice metric {key} in {stats_path}")
                        row[key] = fv
                    for key, value in slice_block.items():
                        if key in SLICE_METRICS or key in SLICE_NON_SCALAR_KEYS or isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)):
                            fv = float(value)
                            if not math.isfinite(fv):
                                raise ValueError(f"Non-finite slice metric {key} in {stats_path}")
                            row[key] = fv
                    row["has_slice_metrics"] = True
                else:
                    row["has_slice_metrics"] = False
            else:
                row["has_slice_metrics"] = False
            rows.append(row)
        except (ValueError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"Warning: skipping {stats_path}: {exc}")
    return rows


def discover_phantom_stats_files(stats_dir: Path) -> List[Path]:
    """Backward-compatible alias for ``discover_non_bids_stats_files``."""
    return discover_non_bids_stats_files(stats_dir)


def default_out_dir_for_phantom_stats_dir(stats_dir: Path) -> Path:
    """Backward-compatible alias for ``default_out_dir_for_stats_dir``."""
    return default_out_dir_for_stats_dir(stats_dir)


def resolve_phantom_session_label(
    payload: Dict[str, object],
    stats_path: Path,
    label_by: str,
) -> str:
    """Backward-compatible alias for ``resolve_non_bids_session_label``."""
    return resolve_non_bids_session_label(payload=payload, stats_path=stats_path, label_by=label_by)


def load_phantom_metric_rows(
    stats_paths: Sequence[Path],
    label_by: str,
) -> List[Dict[str, object]]:
    """Backward-compatible alias for ``load_non_bids_metric_rows``."""
    return load_non_bids_metric_rows(stats_paths=stats_paths, label_by=label_by)


def _raise_if_mixed_slice_metrics(rows: Sequence[Dict[str, object]]) -> None:
    """Fail fast when some stats JSON include slice summaries and others do not.

    Uniform absence (no runs have slice metrics) is allowed. Uniform presence is allowed.
    Mixed schemas make slice panels ambiguous; callers should recompute derivatives.

    Args:
        rows (Sequence[Dict[str, object]]): Loaded rows from ``load_metric_rows`` /
            ``load_phantom_metric_rows``.

    Raises:
        ValueError: If any but not all rows have ``has_slice_metrics``.
    """
    with_slice = [str(r["stats_path"]) for r in rows if r.get("has_slice_metrics")]
    without = [str(r["stats_path"]) for r in rows if not r.get("has_slice_metrics")]
    if not with_slice or not without:
        return
    n_show = 8
    msg = (
        "Mixed slice_ftsnr_metrics schema in this report scope: some *_tsnr_stats.json files "
        f"include complete worst-slice summaries ({len(with_slice)} file(s)), others do not "
        f"({len(without)} file(s)). Slice panels and slice CSV rows require every run to "
        "include slice_ftsnr_metrics with non-null worst_slice_spike_pct_tr_abs_robust_z_gt_4 and "
        "worst_slice_spike_max_abs_robust_z.\n"
        "Examples with complete slice metrics:\n"
    )
    for p in with_slice[:n_show]:
        msg += f"  - {p}\n"
    if len(with_slice) > n_show:
        msg += f"  - ... and {len(with_slice) - n_show} more\n"
    msg += "Examples missing those fields:\n"
    for p in without[:n_show]:
        msg += f"  - {p}\n"
    if len(without) > n_show:
        msg += f"  - ... and {len(without) - n_show} more\n"
    msg += (
        "Fix: regenerate all BOLD stats for this scope with one tSNR version, for example:\n"
        "  uv run tsnr.py /path/to/sub-<label>/ses-<label>/func brain"
    )
    raise ValueError(msg.strip())


def _raise_if_mixed_spike_metrics(rows: Sequence[Dict[str, object]]) -> None:
    """Fail fast when ROI spike summaries are present on some runs only (``--spike-metrics-panels``).

    Args:
        rows (Sequence[Dict[str, object]]): Loaded metric rows.

    Raises:
        ValueError: If any but not all rows have ``has_spike_metrics``.
    """
    with_spike = [str(r["stats_path"]) for r in rows if r.get("has_spike_metrics")]
    without = [str(r["stats_path"]) for r in rows if not r.get("has_spike_metrics")]
    if not with_spike or not without:
        return
    n_show = 8
    msg = (
        "Mixed roi_mean_tr_spike_metrics schema in this report scope: some files have full "
        f"TR spike summaries ({len(with_spike)} file(s)), others do not ({len(without)} file(s)). "
        "Spike panels require max_abs_robust_z, pct_tr_abs_robust_z_gt_4, and "
        "n_tr_abs_robust_z_gt_4 on every run.\n"
        "Examples with complete spike metrics:\n"
    )
    for p in with_spike[:n_show]:
        msg += f"  - {p}\n"
    if len(with_spike) > n_show:
        msg += f"  - ... and {len(with_spike) - n_show} more\n"
    msg += "Examples missing those fields:\n"
    for p in without[:n_show]:
        msg += f"  - {p}\n"
    if len(without) > n_show:
        msg += f"  - ... and {len(without) - n_show} more\n"
    msg += (
        "Fix: regenerate all BOLD stats for this scope with one tSNR version, for example:\n"
        "  uv run tsnr.py /path/to/sub-<label>/ses-<label>/func brain"
    )
    raise ValueError(msg.strip())


def filter_rows(
    rows: Sequence[Dict[str, object]],
    subject: Optional[str],
    session: Optional[str],
) -> List[Dict[str, object]]:
    """Filter rows by optional subject and session labels.
    Args:
        rows (Sequence[Dict[str, object]]): Run-level rows.
        subject (Optional[str]): Subject label like ``sub-3334``.
        session (Optional[str]): Session label like ``ses-1a``.
    Returns:
        List[Dict[str, object]]: Filtered rows.
    """
    out = list(rows)
    if subject is not None:
        out = [row for row in out if str(row["sub"]) == subject]
    if session is not None:
        out = [row for row in out if str(row["ses"]) == session]
    return out


def _echo_order_key(echo_label: str) -> Tuple[int, str]:
    """Sort key for echo labels.
    Args:
        echo_label (str): Echo label such as ``echo-2``.
    Returns:
        Tuple[int, str]: Numeric-first sorting key.
    """
    match = re.search(r"echo-(\d+)$", echo_label)
    if match is None:
        return (10_000, echo_label)
    return (int(match.group(1)), echo_label)


def _group_key(row: Dict[str, object], group_by_task: bool) -> Tuple[str, ...]:
    """Build grouping key for aggregation."""
    if group_by_task:
        return (
            str(row["ses"]),
            str(row["echo"]),
            str(row["task"]),
        )
    return (
        str(row["ses"]),
        str(row["echo"]),
    )


def _error_value(values: np.ndarray, mode: str) -> float:
    """Compute one-sided error bar size for a vector.
    Args:
        values (np.ndarray): Numeric values.
        mode (str): ``sd``, ``sem``, or ``ci95``.
    Returns:
        float: Error magnitude.
    Raises:
        ValueError: If mode is invalid.
    """
    n = int(values.size)
    if n <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    if mode == "sd":
        return std
    if mode == "sem":
        return std / math.sqrt(float(n))
    if mode == "ci95":
        return 1.96 * (std / math.sqrt(float(n)))
    raise ValueError(f"Invalid error mode: {mode}")


def aggregate_metric_rows(
    rows: Sequence[Dict[str, object]],
    metric: str,
    error_mode: str,
    group_by_task: bool,
) -> List[Dict[str, object]]:
    """Aggregate run-level rows into session/echo groups.
    Args:
        rows (Sequence[Dict[str, object]]): Run-level rows.
        metric (str): Metric key to aggregate.
        error_mode (str): Error bar mode.
        group_by_task (bool): Whether task is part of grouping.
    Returns:
        List[Dict[str, object]]: Aggregated summary rows.
    """
    grouped: Dict[Tuple[str, ...], List[float]] = {}
    for row in rows:
        key = _group_key(row, group_by_task)
        value = float(row[metric])
        grouped.setdefault(key, []).append(value)

    out: List[Dict[str, object]] = []
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        summary: Dict[str, object] = {
            "ses": key[0],
            "echo": key[1],
            "n_runs": int(arr.size),
            f"{metric}_mean": float(np.mean(arr)),
            f"{metric}_error": float(_error_value(arr, error_mode)),
        }
        if group_by_task:
            summary["task"] = key[2]
        out.append(summary)
    out.sort(
        key=lambda r: (
            str(r["ses"]),
            _echo_order_key(str(r["echo"])),
            str(r.get("task", "")),
        )
    )
    return out


def write_summary_csv(summary_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Write aggregated rows to CSV.
    Args:
        summary_rows (Sequence[Dict[str, object]]): Combined summary rows.
        output_path (Path): Destination CSV path.
    Returns:
        None: This function returns nothing.
    """
    if not summary_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in summary_rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _tidy_csv_rows(
    aggregated_by_metric: Dict[str, Sequence[Dict[str, object]]],
    subject_filter: Optional[str],
    session_filter: Optional[str],
    metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Convert metric-specific summaries into tidy long CSV rows.
    Args:
        aggregated_by_metric (Dict[str, Sequence[Dict[str, object]]]): Aggregated rows keyed by metric.
        subject_filter (Optional[str]): CLI subject filter if used.
        session_filter (Optional[str]): CLI session filter if used.
    Returns:
        List[Dict[str, object]]: Long-format rows with one metric per record.
    """
    out: List[Dict[str, object]] = []
    for metric in metric_names:
        metric_rows = aggregated_by_metric.get(metric, [])
        mean_key = f"{metric}_mean"
        error_key = f"{metric}_error"
        for row in metric_rows:
            out.append(
                {
                    "metric": metric,
                    "sub": subject_filter if subject_filter is not None else "all",
                    "ses": str(row["ses"]) if session_filter is None else session_filter,
                    "echo": str(row["echo"]),
                    "task": str(row.get("task", "pooled")),
                    "n_runs": int(row["n_runs"]),
                    "mean": float(row[mean_key]),
                    "error": float(row[error_key]),
                }
            )
    out.sort(key=lambda r: (str(r["metric"]), str(r["ses"]), _echo_order_key(str(r["echo"])), str(r["task"])))
    return out


def _session_label(summary_row: Dict[str, object], group_by_task: bool) -> str:
    """Build a legend label for one line."""
    ses = str(summary_row["ses"])
    if group_by_task:
        return f"{ses} | {summary_row['task']}"
    return ses


def _plot_metric_on_axis(
    axis: plt.Axes,
    aggregated_rows: Sequence[Dict[str, object]],
    metric: str,
    error_mode: str,
    group_by_task: bool,
    show_error_bars: bool,
    point_annotations: Optional[Dict[Tuple[str, str], str]] = None,
    subtitle_note: Optional[str] = None,
    x_axis_label: str = "Echo",
    axis_group_label: str = "echo",
    include_error_mode_in_title: bool = True,
) -> None:
    """Render one metric on a provided matplotlib axis.
    Args:
        axis (plt.Axes): Destination axis.
        aggregated_rows (Sequence[Dict[str, object]]): Aggregated rows for one metric.
        metric (str): Metric name.
        error_mode (str): Error mode label (for subplot title when error bars are shown).
        group_by_task (bool): Whether lines split by task.
        show_error_bars (bool): When False, draw lines with markers only (no y error).
    Returns:
        None: This function returns nothing.
    """
    if not aggregated_rows:
        axis.set_visible(False)
        return
    metric_mean_key = f"{metric}_mean"
    metric_err_key = f"{metric}_error"

    line_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in aggregated_rows:
        line_groups.setdefault(_session_label(row, group_by_task), []).append(dict(row))

    x_labels_all = sorted({str(row["echo"]) for row in aggregated_rows}, key=_echo_order_key)
    x_index = {label: idx for idx, label in enumerate(x_labels_all)}
    n_values_all = [int(row["n_runs"]) for row in aggregated_rows]
    n_unique = sorted(set(n_values_all))
    if len(n_unique) == 1:
        n_label = f"n per {axis_group_label}={n_unique[0]}"
    else:
        n_label = f"n per {axis_group_label} range={n_unique[0]}..{n_unique[-1]}"

    for label, group_rows in sorted(line_groups.items(), key=lambda kv: kv[0]):
        group_rows.sort(key=lambda row: _echo_order_key(str(row["echo"])))
        x_pos = np.asarray([x_index[str(row["echo"])] for row in group_rows], dtype=np.float64)
        y_vals = np.asarray([float(row[metric_mean_key]) for row in group_rows], dtype=np.float64)
        y_errs = np.asarray([float(row[metric_err_key]) for row in group_rows], dtype=np.float64)
        if show_error_bars:
            axis.errorbar(
                x_pos,
                y_vals,
                yerr=y_errs,
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=label,
            )
        else:
            axis.plot(
                x_pos,
                y_vals,
                marker="o",
                linewidth=1.8,
                label=label,
            )
        if point_annotations is not None:
            y_span = max(float(np.max(y_vals) - np.min(y_vals)), 1e-9)
            offset = 0.02 * y_span
            for row, x_val, y_val in zip(group_rows, x_pos, y_vals):
                ann = point_annotations.get((label, str(row["echo"])))
                if ann:
                    axis.text(
                        float(x_val),
                        float(y_val) + offset,
                        ann,
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )

    axis.set_xticks(np.arange(len(x_labels_all), dtype=np.float64))
    axis.set_xticklabels(x_labels_all)
    axis.set_xlabel(x_axis_label)
    axis.set_ylabel(METRIC_YLABEL.get(metric, metric))
    ylabel = METRIC_YLABEL.get(metric, metric)
    if show_error_bars and include_error_mode_in_title:
        base_title = f"{ylabel} ({error_mode} error bars; {n_label})"
    else:
        base_title = f"{ylabel} ({n_label})"
    if subtitle_note:
        axis.set_title(f"{base_title}; {subtitle_note}")
    else:
        axis.set_title(base_title)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()


def _panel_suptitle(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build panel title with optional dataset filters."""
    filters: List[str] = []
    if subject is not None:
        filters.append(subject)
    if session is not None:
        filters.append(session)
    scope = " | ".join(filters) if filters else "all subjects/sessions"
    split = "session+task lines" if group_by_task else "session lines"
    if show_error_bars:
        return f"tSNR/fTSNR summary across echoes ({scope}; {split}; with error bars)"
    return f"tSNR/fTSNR summary across echoes ({scope}; {split})"


def _spike_panel_suptitle(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build spike-panel title with optional dataset filters."""
    filters: List[str] = []
    if subject is not None:
        filters.append(subject)
    if session is not None:
        filters.append(session)
    scope = " | ".join(filters) if filters else "all subjects/sessions"
    split = "session+task lines" if group_by_task else "session lines"
    if show_error_bars:
        return f"ROI mean TR spike metrics (robust z) across echoes ({scope}; {split}; with error bars)"
    return f"ROI mean TR spike metrics (robust z) across echoes ({scope}; {split})"


def _slice_panel_suptitle(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build slice-panel title with optional dataset filters."""
    filters: List[str] = []
    if subject is not None:
        filters.append(subject)
    if session is not None:
        filters.append(session)
    scope = " | ".join(filters) if filters else "all subjects/sessions"
    split = "session+task lines" if group_by_task else "session lines"
    if show_error_bars:
        return f"Slice-level spike summary across echoes ({scope}; {split}; with error bars)"
    return f"Slice-level spike summary across echoes ({scope}; {split})"


def _panel_suptitle_non_bids_sessions(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build core-panel title for non-BIDS session comparisons."""
    _ = (subject, session, group_by_task)
    return "tSNR/fTSNR summary across QA sessions"


def _spike_panel_suptitle_non_bids_sessions(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build spike-panel title for non-BIDS session comparisons."""
    _ = (subject, session, group_by_task)
    return "ROI mean TR spike metrics (robust z) across QA sessions"


def _slice_panel_suptitle_non_bids_sessions(
    subject: Optional[str],
    session: Optional[str],
    group_by_task: bool,
    show_error_bars: bool = True,
) -> str:
    """Build slice-panel title for non-BIDS session comparisons."""
    _ = (subject, session, group_by_task)
    return "Slice-level spike summary across QA sessions"


def _slice_annotation_lookup(
    aggregated_rows: Sequence[Dict[str, object]],
    annotation_metric: str,
    group_by_task: bool,
) -> Dict[Tuple[str, str], str]:
    """Build per-point slice-index labels ``Z=12`` (capital Z), keyed by (line_label, echo)."""
    out: Dict[Tuple[str, str], str] = {}
    key = f"{annotation_metric}_mean"
    for row in aggregated_rows:
        if key not in row:
            continue
        slice_index = int(round(float(row[key])))
        out[(_session_label(row, group_by_task), str(row["echo"]))] = f"Z={slice_index}"
    return out


def plot_slice_metric_panel(
    aggregated_by_metric: Dict[str, Sequence[Dict[str, object]]],
    error_mode: str,
    output_path: Path,
    group_by_task: bool,
    subject: Optional[str],
    session: Optional[str],
    show_error_bars: bool,
    x_axis_label: str = "Echo",
    axis_group_label: str = "echo",
    suptitle_fn: Callable[..., str] = _slice_panel_suptitle,
    include_error_mode_in_title: bool = True,
) -> None:
    """Render two-panel slice summary with Z-index annotations on points."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)
    top_ann = _slice_annotation_lookup(
        aggregated_by_metric.get("worst_slice_spike_pct_slice_index", []),
        annotation_metric="worst_slice_spike_pct_slice_index",
        group_by_task=group_by_task,
    )
    bottom_ann = _slice_annotation_lookup(
        aggregated_by_metric.get("worst_slice_spike_max_abs_slice_index", []),
        annotation_metric="worst_slice_spike_max_abs_slice_index",
        group_by_task=group_by_task,
    )
    _plot_metric_on_axis(
        axes[0],
        aggregated_rows=aggregated_by_metric.get("worst_slice_spike_pct_tr_abs_robust_z_gt_4", []),
        metric="worst_slice_spike_pct_tr_abs_robust_z_gt_4",
        error_mode=error_mode,
        group_by_task=group_by_task,
        show_error_bars=show_error_bars,
        point_annotations=top_ann,
        subtitle_note="Z=slice for worst % (|robust z|>4)",
        x_axis_label=x_axis_label,
        axis_group_label=axis_group_label,
        include_error_mode_in_title=include_error_mode_in_title,
    )
    _plot_metric_on_axis(
        axes[1],
        aggregated_rows=aggregated_by_metric.get("worst_slice_spike_max_abs_robust_z", []),
        metric="worst_slice_spike_max_abs_robust_z",
        error_mode=error_mode,
        group_by_task=group_by_task,
        show_error_bars=show_error_bars,
        point_annotations=bottom_ann,
        subtitle_note="Z=slice for max |robust z| (y-axis)",
        x_axis_label=x_axis_label,
        axis_group_label=axis_group_label,
        include_error_mode_in_title=include_error_mode_in_title,
    )
    fig.suptitle(
        suptitle_fn(
            subject=subject,
            session=session,
            group_by_task=group_by_task,
            show_error_bars=show_error_bars,
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_panel(
    aggregated_by_metric: Dict[str, Sequence[Dict[str, object]]],
    metric_order: Tuple[str, ...],
    error_mode: str,
    output_path: Path,
    group_by_task: bool,
    subject: Optional[str],
    session: Optional[str],
    suptitle_fn: Callable[..., str],
    figsize: Tuple[float, float] = (11, 14),
    show_error_bars: bool = True,
    x_axis_label: str = "Echo",
    axis_group_label: str = "echo",
    include_error_mode_in_title: bool = True,
) -> None:
    """Render metrics as one PNG with stacked subplots.
    Args:
        aggregated_by_metric (Dict[str, Sequence[Dict[str, object]]]): Aggregated rows by metric name.
        metric_order (Tuple[str, ...]): Subplot order (top to bottom).
        error_mode (str): Error mode label.
        output_path (Path): Plot output path.
        group_by_task (bool): Whether lines split by task.
        subject (Optional[str]): Optional subject filter.
        session (Optional[str]): Optional session filter.
        suptitle_fn: Callable taking (subject, session, group_by_task, show_error_bars) for figure title.
        figsize (Tuple[float, float]): Figure size.
        show_error_bars (bool): When False, line plots without y error bars.
    Returns:
        None: This function returns nothing.
    """
    fig, axes = plt.subplots(len(metric_order), 1, figsize=figsize, sharex=False)
    if len(metric_order) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metric_order):
        _plot_metric_on_axis(
            axis,
            aggregated_rows=aggregated_by_metric.get(metric, []),
            metric=metric,
            error_mode=error_mode,
            group_by_task=group_by_task,
            show_error_bars=show_error_bars,
            x_axis_label=x_axis_label,
            axis_group_label=axis_group_label,
            include_error_mode_in_title=include_error_mode_in_title,
        )
    fig.suptitle(
        suptitle_fn(
            subject=subject,
            session=session,
            group_by_task=group_by_task,
            show_error_bars=show_error_bars,
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _linear_detrend_1d(y: np.ndarray) -> np.ndarray:
    """Remove an OLS straight line over TR indices ``0..n-1``.
    Args:
        y (np.ndarray): 1D ROI-mean (or residual) series.
    Returns:
        np.ndarray: Residuals; for ``n < 3`` returns a copy of ``y`` (no detrend).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 3:
        return y.copy()
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, y, 1)
    return y - np.polyval(coef, t)


def _signed_robust_z_median_mad(y: np.ndarray) -> np.ndarray:
    """Signed robust z per sample (median and MAD scaled by 1.4826, same as ``tsnr.py``).
    Args:
        y (np.ndarray): 1D series (for example linear-detrended ROI mean).
    Returns:
        np.ndarray: Per-index robust z, same length as ``y`` (empty if invalid).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n_t = int(y.size)
    if n_t < 1 or not np.all(np.isfinite(y)):
        return np.array([], dtype=np.float64)
    if n_t < 2:
        return np.zeros(1, dtype=np.float64)
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    rs = 1.4826 * mad
    if rs <= 1e-12:
        rs = float(np.std(y, ddof=0))
    if rs <= 1e-12:
        return np.zeros_like(y)
    return (y - median) / rs


def _robust_z_series_for_tr_plot(raw_roi_mean: Sequence[float]) -> List[float]:
    """Linear detrend on ROI mean, then robust z (TR-index figure only; not echo summaries).
    Args:
        raw_roi_mean (Sequence[float]): ROI-mean signal per TR after ``timepoint_selection``.
    Returns:
        List[float]: Signed robust z per TR for plotting.
    """
    y = np.asarray(list(raw_roi_mean), dtype=np.float64).ravel()
    if y.size >= 3:
        resid = _linear_detrend_1d(y)
    else:
        resid = y.copy()
    z = _signed_robust_z_median_mad(resid)
    return [float(x) for x in z]


def _count_abs_robust_z_gt_4(zr: Sequence[float]) -> int:
    """Count TRs with absolute robust z strictly greater than 4 (plot convention).
    Args:
        zr (Sequence[float]): Per-TR robust z values shown in the figure.
    Returns:
        int: Number of TRs with ``|z| > 4``.
    """
    return sum(1 for z in zr if abs(float(z)) > 4.0)


def roi_mean_signal_series_from_spike_block(spike: object) -> Optional[List[float]]:
    """Raw ROI-mean signal per TR (from JSON only; for signal-vs-TR figures).
    Args:
        spike (object): ``roi_mean_tr_spike_metrics`` dict from stats JSON.
    Returns:
        Optional[List[float]]: Per-TR ROI mean signal, or ``None`` if missing.
    """
    if not isinstance(spike, dict):
        return None
    raw = spike.get("roi_mean_signal_per_tr")
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def per_slice_metrics_row_or_empty(slice_metrics: object, slice_index: int) -> Dict[str, object]:
    """Return one per-slice row, defaulting to an empty row when missing.
    Missing rows are treated as empty/not-applicable slices for backward
    compatibility with compact stats JSON output.
    Args:
        slice_metrics (object): ``slice_ftsnr_metrics`` object from stats JSON.
        slice_index (int): Z slice index to query.
    Returns:
        Dict[str, object]: Existing row or an empty-equivalent default row.
    """
    empty: Dict[str, object] = {
        "slice_index": int(slice_index),
        "n_voxels": 0,
        "eligible": False,
        "slice_n_tr_abs_robust_z_gt_4": 0,
        "slice_pct_tr_abs_robust_z_gt_4": 0.0,
        "slice_max_abs_robust_z": 0.0,
    }
    if not isinstance(slice_metrics, dict):
        return empty
    rows = slice_metrics.get("per_slice")
    if not isinstance(rows, list):
        return empty
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int(row.get("slice_index", -1)) == int(slice_index):
            return row
    return empty


def tr_plot_z_from_spike_block(spike: object) -> Optional[List[float]]:
    """Z series for the TR-index grid: detrended from ``roi_mean_signal_per_tr`` when present.
    Args:
        spike (object): ``roi_mean_tr_spike_metrics`` dict from stats JSON.
    Returns:
        Optional[List[float]]: Per-TR z for plotting, or ``None`` if unavailable.
    """
    if not isinstance(spike, dict):
        return None
    raw = spike.get("roi_mean_signal_per_tr")
    if isinstance(raw, list) and len(raw) > 0:
        try:
            return _robust_z_series_for_tr_plot([float(x) for x in raw])
        except (TypeError, ValueError):
            pass
    zr = spike.get("robust_z_per_tr")
    if isinstance(zr, list) and len(zr) > 0:
        return [float(x) for x in zr]
    return None


def discover_session_stats_files(bids_root: Path, subject: str, session: str) -> List[Path]:
    """List stats JSON files for one subject/session (sorted by filename).
    Args:
        bids_root (Path): BIDS dataset root.
        subject (str): Subject id, e.g. ``sub-3334``.
        session (str): Session id, e.g. ``ses-1a``.
    Returns:
        List[Path]: Sorted ``*_tsnr_stats.json`` paths under that session's derivatives/tsnr.
    """
    session_dir = bids_root / subject / session / "derivatives" / "tsnr"
    if not session_dir.is_dir():
        return []
    return sorted(p for p in session_dir.glob("*_tsnr_stats.json") if p.is_file())


def plot_robust_z_tr_session_grid(
    stats_paths: Sequence[Path],
    output_path: Path,
    subject: str,
    session: str,
) -> bool:
    """One multi-panel figure: robust z (signed) vs TR index for each run's stats file.
    Args:
        stats_paths (Sequence[Path]): One stats JSON per BOLD run, sorted for display order.
        output_path (Path): Output PNG path.
        subject (str): Subject label for the title.
        session (str): Session label for the title.
    Returns:
        bool: True if the figure was written, False if per-TR plot data were missing anywhere.
    """
    n = len(stats_paths)
    if n == 0:
        return False

    series_list: List[Tuple[Path, List[float], str]] = []
    any_raw_for_detrend = False
    for stats_path in stats_paths:
        payload = _read_json(stats_path)
        spike = payload.get("roi_mean_tr_spike_metrics")
        if not isinstance(spike, dict):
            print(f"Warning: missing roi_mean_tr_spike_metrics in {stats_path}")
            return False
        zr = tr_plot_z_from_spike_block(spike)
        if zr is None or len(zr) == 0:
            print(
                f"Warning: need roi_mean_signal_per_tr or robust_z_per_tr in {stats_path}; "
                "re-run tsnr.py brain mode to refresh stats."
            )
            return False
        raw = spike.get("roi_mean_signal_per_tr")
        if isinstance(raw, list) and len(raw) >= 3:
            any_raw_for_detrend = True
        stem = stats_path.name.replace("_tsnr_stats.json", "")
        series_list.append((stats_path, zr, stem))

    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    for idx, (_, zr, stem) in enumerate(series_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        x = np.arange(len(zr), dtype=np.float64)
        ax.axhline(4.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.85)
        ax.axhline(-4.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.85)
        ax.axhline(0.0, color="lightgray", linestyle="-", linewidth=0.65)
        ax.plot(x, zr, linewidth=0.95, color="C0")
        ax.set_xlim(0, max(len(zr) - 1, 0))
        n_gt4 = _count_abs_robust_z_gt_4(zr)
        line1 = stem if len(stem) <= 44 else stem[:41] + "..."
        title = f"{line1}\n|z|>4: {n_gt4} TRs"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("TR index", fontsize=8)
        ax.set_ylabel("robust z", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    detrend_note = (
        "linear detrend on ROI mean then robust z. "
        if any_raw_for_detrend
        else "robust z from JSON (re-run tsnr for linear detrend on new plots). "
    )
    fig.suptitle(
        f"ROI-mean robust z vs TR ({subject} | {session}; {detrend_note}one panel per BOLD run)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_roi_mean_signal_tr_session_grid(
    stats_paths: Sequence[Path],
    output_path: Path,
    subject: str,
    session: str,
) -> bool:
    """One multi-panel figure: ROI-mean signal vs TR index for each run (raw series from JSON).
    Args:
        stats_paths (Sequence[Path]): One stats JSON per BOLD run, sorted for display order.
        output_path (Path): Output PNG path.
        subject (str): Subject label for the title.
        session (str): Session label for the title.
    Returns:
        bool: True if the figure was written, False if ``roi_mean_signal_per_tr`` was missing anywhere.
    """
    n = len(stats_paths)
    if n == 0:
        return False

    series_list: List[Tuple[Path, List[float], str]] = []
    for stats_path in stats_paths:
        payload = _read_json(stats_path)
        spike = payload.get("roi_mean_tr_spike_metrics")
        if not isinstance(spike, dict):
            print(f"Warning: missing roi_mean_tr_spike_metrics in {stats_path}")
            return False
        sig = roi_mean_signal_series_from_spike_block(spike)
        if sig is None or len(sig) == 0:
            print(
                f"Warning: missing roi_mean_signal_per_tr in {stats_path}; "
                "re-run tsnr.py brain mode to refresh stats."
            )
            return False
        stem = stats_path.name.replace("_tsnr_stats.json", "")
        series_list.append((stats_path, sig, stem))

    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    for idx, (_, y, stem) in enumerate(series_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        x = np.arange(len(y), dtype=np.float64)
        arr = np.asarray(y, dtype=np.float64)
        ax.plot(x, arr, linewidth=0.95, color="C0")
        ax.set_xlim(0, max(len(y) - 1, 0))
        line1 = stem if len(stem) <= 40 else stem[:37] + "..."
        ymin, ymax = float(np.min(arr)), float(np.max(arr))
        title = f"{line1}\n{len(y)} TRs; signal min/max {ymin:.2f} / {ymax:.2f}"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("TR index", fontsize=8)
        ax.set_ylabel("ROI mean signal", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(
        f"ROI-mean signal vs TR ({subject} | {session}; raw mean over ROI voxels; one panel per BOLD run)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def run_report(
    bids_root: Optional[Path],
    out_dir: Path,
    error_mode: str,
    group_by_task: bool,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    robust_z_tr_panels: bool = False,
    roi_mean_signal_tr_panels: bool = False,
    show_error_bars: bool = True,
    stats_dir: Optional[Path] = None,
    phantom_stats_dir: Optional[Path] = None,
    label_by: str = "auto",
    spike_metrics_panels: bool = False,
) -> List[Path]:
    """Generate PNG summary plots and CSV.
    Args:
        bids_root (Optional[Path]): BIDS root with derivatives (BIDS mode).
        out_dir (Path): Output directory for plots and CSV.
        error_mode (str): Error bar mode.
        group_by_task (bool): Include task in grouping when True.
        subject (Optional[str]): Optional subject filter (for example ``sub-3334``).
        session (Optional[str]): Optional session filter (for example ``ses-1a``).
        robust_z_tr_panels (bool): When True, write ``robust_z_vs_tr_{subject}_{session}.png`` (requires subject and session).
        roi_mean_signal_tr_panels (bool): When True, write ``roi_mean_signal_vs_tr_{subject}_{session}.png`` (raw ROI mean vs TR).
        show_error_bars (bool): When False, echo panels use lines without y error bars.
        stats_dir (Optional[Path]): Directory with non-BIDS ``*_tsnr_stats.json`` files.
        phantom_stats_dir (Optional[Path]): Deprecated alias for ``stats_dir``.
        label_by (str): Non-BIDS label mode: ``auto``, ``metadata_date``, or ``filename_date``.
        spike_metrics_panels (bool): When True and all inputs carry ROI spike metrics, write the
            spike-metrics PNG and add spike fields to the summary CSV (default off).
    Returns:
        List[Path]: Paths generated by this run.
    Raises:
        ValueError: If no valid stats rows are found.
    """
    effective_stats_dir = stats_dir if stats_dir is not None else phantom_stats_dir
    if stats_dir is not None and phantom_stats_dir is not None and stats_dir != phantom_stats_dir:
        raise ValueError("Use only one of stats_dir or phantom_stats_dir")
    if effective_stats_dir is not None:
        stats_files = discover_non_bids_stats_files(effective_stats_dir)
        rows = load_non_bids_metric_rows(stats_files, label_by=label_by)
        # Non-BIDS mode compares session labels directly; ignore BIDS filters and task grouping.
        subject = "non-bids"
        session = "session-series"
        group_by_task = False
        x_axis_label = "By session"
        axis_group_label = "session"
        panel_suffix = "session"
        include_error_mode_in_title = False
        core_suptitle_fn = _panel_suptitle_non_bids_sessions
        spike_suptitle_fn = _spike_panel_suptitle_non_bids_sessions
        slice_suptitle_fn = _slice_panel_suptitle_non_bids_sessions
    else:
        if bids_root is None:
            raise ValueError("bids_root is required when stats_dir is not provided")
        stats_files = discover_stats_files(bids_root)
        rows = filter_rows(load_metric_rows(stats_files), subject=subject, session=session)
        x_axis_label = "Echo"
        axis_group_label = "echo"
        panel_suffix = "echo"
        include_error_mode_in_title = True
        core_suptitle_fn = _panel_suptitle
        spike_suptitle_fn = _spike_panel_suptitle
        slice_suptitle_fn = _slice_panel_suptitle
    if not rows:
        raise ValueError(
            "No valid stats rows discovered under "
            f"{effective_stats_dir if effective_stats_dir is not None else bids_root} "
            f"for filters subject={subject!r}, session={session!r}"
        )

    _raise_if_mixed_slice_metrics(rows)
    if spike_metrics_panels:
        _raise_if_mixed_spike_metrics(rows)

    generated: List[Path] = []
    aggregated_by_metric: Dict[str, Sequence[Dict[str, object]]] = {}
    for metric in METRICS:
        aggregated = aggregate_metric_rows(rows, metric=metric, error_mode=error_mode, group_by_task=group_by_task)
        aggregated_by_metric[metric] = aggregated
    panel_path = out_dir / f"metrics_panel_by_{panel_suffix}_{error_mode}.png"
    plot_metric_panel(
        aggregated_by_metric=aggregated_by_metric,
        metric_order=METRICS,
        error_mode=error_mode,
        output_path=panel_path,
        group_by_task=group_by_task,
        subject=subject,
        session=session,
        suptitle_fn=core_suptitle_fn,
        show_error_bars=show_error_bars,
        x_axis_label=x_axis_label,
        axis_group_label=axis_group_label,
        include_error_mode_in_title=include_error_mode_in_title,
    )
    generated.append(panel_path)

    has_all_spikes = bool(rows) and all(bool(r.get("has_spike_metrics")) for r in rows)
    has_all_slice_metrics = bool(rows) and all(bool(r.get("has_slice_metrics")) for r in rows)
    csv_metric_names: List[str] = list(METRICS)
    if spike_metrics_panels and has_all_spikes:
        for metric in SPIKE_METRICS:
            aggregated_by_metric[metric] = aggregate_metric_rows(
                rows, metric=metric, error_mode=error_mode, group_by_task=group_by_task
            )
        spike_path = out_dir / f"spike_metrics_panel_by_{panel_suffix}_{error_mode}.png"
        plot_metric_panel(
            aggregated_by_metric=aggregated_by_metric,
            metric_order=SPIKE_METRICS,
            error_mode=error_mode,
            output_path=spike_path,
            group_by_task=group_by_task,
            subject=subject,
            session=session,
            suptitle_fn=spike_suptitle_fn,
            figsize=(11, 12),
            show_error_bars=show_error_bars,
            x_axis_label=x_axis_label,
            axis_group_label=axis_group_label,
            include_error_mode_in_title=include_error_mode_in_title,
        )
        generated.append(spike_path)
        csv_metric_names.extend(SPIKE_METRICS)
    elif spike_metrics_panels and not has_all_spikes:
        print(
            "Warning: skipping spike metric plots and CSV rows (no stats JSON includes "
            "roi_mean_tr_spike_metrics with all required keys for every run in this scope)."
        )
    if has_all_slice_metrics:
        for metric in SLICE_METRICS + SLICE_INDEX_METRICS:
            aggregated_by_metric[metric] = aggregate_metric_rows(
                rows, metric=metric, error_mode=error_mode, group_by_task=group_by_task
            )
        slice_path = out_dir / f"slice_metrics_panel_by_{panel_suffix}_{error_mode}.png"
        plot_slice_metric_panel(
            aggregated_by_metric=aggregated_by_metric,
            error_mode=error_mode,
            output_path=slice_path,
            group_by_task=group_by_task,
            subject=subject,
            session=session,
            show_error_bars=show_error_bars,
            x_axis_label=x_axis_label,
            axis_group_label=axis_group_label,
            suptitle_fn=slice_suptitle_fn,
            include_error_mode_in_title=include_error_mode_in_title,
        )
        generated.append(slice_path)
        csv_metric_names.extend(SLICE_METRICS + SLICE_INDEX_METRICS)

    csv_path = out_dir / "aggregated_metric_summary.csv"
    csv_rows = _tidy_csv_rows(
        aggregated_by_metric,
        subject_filter=subject,
        session_filter=session,
        metric_names=csv_metric_names,
    )
    write_summary_csv(csv_rows, csv_path)
    generated.append(csv_path)

    if robust_z_tr_panels:
        if effective_stats_dir is not None:
            raise ValueError("--robust-z-tr-panels is available only in BIDS mode")
        if subject is None or session is None:
            raise ValueError("--robust-z-tr-panels requires --subject and --session")
        session_paths = discover_session_stats_files(bids_root, subject, session)
        if not session_paths:
            raise ValueError(f"No stats JSON files under {bids_root}/{subject}/{session}/derivatives/tsnr")
        rz_path = out_dir / f"robust_z_vs_tr_{subject}_{session}.png"
        if plot_robust_z_tr_session_grid(session_paths, rz_path, subject=subject, session=session):
            generated.append(rz_path)
    if roi_mean_signal_tr_panels:
        if effective_stats_dir is not None:
            raise ValueError("--roi-mean-signal-tr-panels is available only in BIDS mode")
        if subject is None or session is None:
            raise ValueError("--roi-mean-signal-tr-panels requires --subject and --session")
        session_paths = discover_session_stats_files(bids_root, subject, session)
        if not session_paths:
            raise ValueError(f"No stats JSON files under {bids_root}/{subject}/{session}/derivatives/tsnr")
        sig_path = out_dir / f"roi_mean_signal_vs_tr_{subject}_{session}.png"
        if plot_roi_mean_signal_tr_session_grid(session_paths, sig_path, subject=subject, session=session):
            generated.append(sig_path)
    return generated


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser.
    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot tSNR/fTSNR stats across echo and session. "
            "If every input stats JSON includes slice_ftsnr_metrics with complete worst-slice fields, "
            "also writes slice_metrics_panel_by_*_<error>.png and corresponding CSV rows."
        )
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="BIDS dataset root containing sub-*/ses-*/derivatives/tsnr stats JSON files.",
    )
    parser.add_argument(
        "--stats-dir",
        dest="stats_dir",
        type=Path,
        default=None,
        help="Directory containing *_tsnr_stats.json files for non-BIDS workflow.",
    )
    parser.add_argument(
        "--phantom-stats-dir",
        dest="stats_dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--label-by",
        type=str,
        default="auto",
        choices=LABEL_BY_CHOICES,
        help="In non-BIDS mode, resolve session labels from metadata date, filename date, or auto.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "tsnr_plots",
        help="Directory for generated plots and summary CSV.",
    )
    parser.add_argument(
        "--error-bar",
        type=str,
        default="sem",
        choices=ERROR_BAR_CHOICES,
        help="Error bar mode for grouped summaries.",
    )
    parser.add_argument(
        "--group-by-task",
        action="store_true",
        help="Separate lines by session+task (default when both --subject and --session are set).",
    )
    parser.add_argument(
        "--pool-across-tasks",
        action="store_true",
        help="Pool all tasks per session/echo (overrides default split-by-task when --subject and --session are set).",
    )
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Draw echo panels as lines only (no y error bars). Default off when curves are pooled; "
        "default on when using default split-by-task for a single subject/session.",
    )
    parser.add_argument(
        "--show-error-bars",
        action="store_true",
        help="Force error bars on echo panels (overrides --no-error-bars and per-task defaults).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject filter, e.g. sub-3334.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional session filter, e.g. ses-1a.",
    )
    parser.add_argument(
        "--robust-z-tr-panels",
        action="store_true",
        help="Also write one multi-panel PNG of robust z vs TR index (requires --subject and --session). "
        "Uses linear detrend on roi_mean_signal_per_tr when present; else robust_z_per_tr. "
        "Re-run tsnr.py brain mode if per-TR fields are missing.",
    )
    parser.add_argument(
        "--roi-mean-signal-tr-panels",
        action="store_true",
        help="Also write roi_mean_signal_vs_tr_<subject>_<session>.png: raw ROI-mean fMRI signal vs TR "
        "(requires --subject and --session; needs roi_mean_signal_per_tr in each stats JSON).",
    )
    parser.add_argument(
        "--spike-metrics-panels",
        action="store_true",
        help="Also write spike_metrics_panel_by_*_<error>.png and add ROI spike metrics to the CSV "
        "(requires roi_mean_tr_spike_metrics with max_abs_robust_z, pct/n TR |z|>4 in every stats JSON). "
        "Default reports omit this figure.",
    )
    return parser


def cli(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.
    Args:
        argv (Optional[List[str]]): Optional argument vector.
    Returns:
        int: Exit code.
    """
    args = build_parser().parse_args(argv)
    bids_root = args.bids_root.expanduser() if args.bids_root is not None else None
    stats_dir = args.stats_dir.expanduser() if args.stats_dir is not None else None
    if (bids_root is None) == (stats_dir is None):
        print("Error: provide exactly one of --bids-root or --stats-dir")
        return 1
    if args.no_error_bars and args.show_error_bars:
        print("Error: use only one of --no-error-bars and --show-error-bars")
        return 1
    out_dir = args.out_dir.expanduser()
    if stats_dir is not None and args.out_dir == Path("reports") / "tsnr_plots":
        out_dir = default_out_dir_for_stats_dir(stats_dir)
    has_sub_ses = args.subject is not None and args.session is not None
    if args.pool_across_tasks:
        group_by_task = False
    elif has_sub_ses:
        group_by_task = True
    else:
        group_by_task = bool(args.group_by_task)
    if args.show_error_bars:
        show_error_bars = True
    elif args.no_error_bars:
        show_error_bars = False
    elif group_by_task:
        show_error_bars = False
    else:
        show_error_bars = True
    try:
        generated = run_report(
            bids_root=bids_root,
            out_dir=out_dir,
            error_mode=args.error_bar,
            group_by_task=group_by_task,
            subject=args.subject,
            session=args.session,
            robust_z_tr_panels=bool(args.robust_z_tr_panels),
            roi_mean_signal_tr_panels=bool(args.roi_mean_signal_tr_panels),
            show_error_bars=show_error_bars,
            stats_dir=stats_dir,
            label_by=str(args.label_by),
            spike_metrics_panels=bool(args.spike_metrics_panels),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print("Generated outputs:")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
