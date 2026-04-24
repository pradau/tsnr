# Author: Perry Radau
# Date: 2026-04-20
# Batch-generate robust z vs TR index grids for every subject/session in a BIDS tree.
# Dependencies: Python 3.10+, matplotlib, numpy (same as plot_tsnr_stats.py)
# Usage: uv run plot_robust_z_tr_all_sessions.py --bids-root /path/to/bids

"""Emit one multi-panel robust-z-vs-TR figure per subject/session (all BOLD stats files)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_tsnr_stats import (
    discover_session_stats_files,
    discover_subject_sessions,
    plot_roi_mean_signal_tr_session_grid,
    plot_robust_z_tr_session_grid,
    roi_mean_signal_series_from_spike_block,
    tr_plot_z_from_spike_block,
)


def _stats_files_missing_roi_mean_series(session_paths: List[Path]) -> List[Path]:
    """Return stats JSON paths missing ``roi_mean_signal_per_tr`` for signal-vs-TR plots.
    Args:
        session_paths (List[Path]): Candidate ``*_tsnr_stats.json`` paths for one session.
    Returns:
        List[Path]: Paths that lack usable ``roi_mean_signal_per_tr``.
    """
    missing: List[Path] = []
    for path in session_paths:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append(path)
            continue
        block = data.get("roi_mean_tr_spike_metrics")
        if roi_mean_signal_series_from_spike_block(block) is None:
            missing.append(path)
    return missing


def _stats_files_missing_tr_plot(session_paths: List[Path]) -> List[Path]:
    """Return stats JSON paths that cannot be used for the TR-index figure.
    Args:
        session_paths (List[Path]): Candidate ``*_tsnr_stats.json`` paths for one session.
    Returns:
        List[Path]: Paths that lack data for ``tr_plot_z_from_spike_block``.
    """
    missing: List[Path] = []
    for path in session_paths:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append(path)
            continue
        block = data.get("roi_mean_tr_spike_metrics")
        if tr_plot_z_from_spike_block(block) is None:
            missing.append(path)
    return missing


def _print_recompute_hints(bids_root: Path, failed_pairs: List[Tuple[str, str]]) -> None:
    """Print uv commands to refresh stats so TR-index plots have required spike fields.
    Args:
        bids_root (Path): BIDS root used for this run.
        failed_pairs (List[Tuple[str, str]]): Subject/session pairs that were skipped.
    Returns:
        None: This function returns nothing.
    """
    print()
    print(
        "These sessions need brain-mode stats recomputed (current JSON lacks "
        "roi_mean_signal_per_tr or robust_z_per_tr under roi_mean_tr_spike_metrics)."
    )
    print("From the tSNR repo, run one command per session (adjust if your layout differs):")
    repo_hint = _REPO_ROOT
    for sub, ses in failed_pairs:
        func_dir = bids_root / sub / ses / "func"
        if func_dir.is_dir():
            print(f'  cd "{repo_hint}" && uv run tsnr.py "{func_dir}" brain')
        else:
            print(f"  (missing {func_dir}; re-run brain mode for {sub} {ses} func data.)")
    print()
    print("Then re-run this plotting script.")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Write robust_z_vs_tr_<sub>_<ses>.png for each subject/session that has "
            "derivatives/tsnr/*_tsnr_stats.json (one subplot per stats file)."
        )
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        required=True,
        help="BIDS dataset root (required).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <bids-root>/reports/robust_z_tr_sessions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List subject/sessions and planned output paths without writing files.",
    )
    parser.add_argument(
        "--roi-mean-signal-tr-panels",
        action="store_true",
        help="Also write roi_mean_signal_vs_tr_<sub>_<ses>.png per session (raw ROI mean vs TR).",
    )
    return parser.parse_args()


def main() -> int:
    """Discover all subject/sessions and write one robust-z grid per session.
    Returns:
        int: 0 if every session produced a figure, 1 if none found or any session failed/skipped.
    """
    args = _parse_args()
    bids_root = args.bids_root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (bids_root / "reports" / "robust_z_tr_sessions")
    )

    try:
        pairs: List[Tuple[str, str]] = discover_subject_sessions(bids_root)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if not pairs:
        print(f"Error: no *_tsnr_stats.json files under {bids_root}")
        return 1

    if args.dry_run:
        print(f"Would write {len(pairs)} figure(s) to {out_dir}:")
        for sub, ses in pairs:
            n = len(discover_session_stats_files(bids_root, sub, ses))
            extra = f" and roi_mean_signal_vs_tr_{sub}_{ses}.png" if args.roi_mean_signal_tr_panels else ""
            print(f"  {sub} {ses}  ({n} stats file(s)) -> robust_z_vs_tr_{sub}_{ses}.png{extra}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    failed_pairs: List[Tuple[str, str]] = []
    for sub, ses in pairs:
        session_paths = discover_session_stats_files(bids_root, sub, ses)
        out_path = out_dir / f"robust_z_vs_tr_{sub}_{ses}.png"
        if not session_paths:
            failures += 1
            failed_pairs.append((sub, ses))
            print(f"Skipped {sub} {ses}: no *_tsnr_stats.json under derivatives/tsnr")
            continue
        bad = _stats_files_missing_tr_plot(session_paths)
        if bad:
            failures += 1
            failed_pairs.append((sub, ses))
            print(
                f"Skipped {sub} {ses}: {len(bad)} of {len(session_paths)} stats file(s) "
                f"lack TR-index plot data (example: {bad[0].name})"
            )
            continue
        ok = plot_robust_z_tr_session_grid(session_paths, out_path, subject=sub, session=ses)
        if ok:
            print(f"Wrote {out_path}")
        else:
            failures += 1
            failed_pairs.append((sub, ses))
            print(f"Warning: plot failed for {sub} {ses}")
            continue
        if args.roi_mean_signal_tr_panels:
            bad_roi = _stats_files_missing_roi_mean_series(session_paths)
            if bad_roi:
                failures += 1
                failed_pairs.append((sub, ses))
                print(
                    f"Skipped ROI-mean plot {sub} {ses}: {len(bad_roi)} of {len(session_paths)} file(s) "
                    f"lack roi_mean_signal_per_tr (example: {bad_roi[0].name})"
                )
                continue
            roi_path = out_dir / f"roi_mean_signal_vs_tr_{sub}_{ses}.png"
            if plot_roi_mean_signal_tr_session_grid(session_paths, roi_path, subject=sub, session=ses):
                print(f"Wrote {roi_path}")
            else:
                failures += 1
                failed_pairs.append((sub, ses))
                print(f"Warning: ROI-mean signal plot failed for {sub} {ses}")

    if failures:
        print(f"Summary: {failures} of {len(pairs)} session(s) had no figure written.")
        _print_recompute_hints(bids_root, failed_pairs)
        return 1
    print(f"Success: {len(pairs)} figure(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
