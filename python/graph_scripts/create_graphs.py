#!/usr/bin/env python3
"""Create benchmark plots from a gmmslam benchmark run directory.

Usage:
    python3 create_graphs.py
    python3 create_graphs.py /path/to/gmmslam/logs/gmmslam_cpp_YYYYMMDD_HHMMSS
    python3 create_graphs.py gmmslam/logs/gmmslam_cpp_YYYYMMDD_HHMMSS

With no run_dir, the newest gmmslam/logs/gmmslam_cpp_* directory is used.
Relative paths are resolved from the workspace root when possible. The script
writes PNG figures and a small summary file into benchmark_graphs/<run_name>
by default.
It only depends on matplotlib and the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titlepad": 10,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15,
    }
)

Row = Dict[str, str]

XLABEL_TIME = "Time since run start (s)"


def read_csv(path: Path) -> List[Row]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_tsv(path: Path) -> List[Row]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_tum(path: Path) -> List[Tuple[float, float, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    poses = []
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                poses.append(
                    (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                )
            except ValueError:
                continue
    return poses


def read_frame_pruning_console(path: Path) -> List[Row]:
    """Parse raw gmm_utils frame-to-frame pruning summaries from console.txt."""
    if not path.exists() or path.stat().st_size == 0:
        return []

    stamp_pattern = re.compile(r"\[[^\]]+,\s*([0-9]+(?:\.[0-9]+)?)\]")
    summary_pattern = re.compile(
        r"frame-to-frame GMM prune summary:\s*"
        r"(\d+)\s*->\s*(\d+)\s*component\(s\)\s*"
        r"\((\d+)\s*removed\),\s*"
        r"(\d+)\s*merge\(s\)\s*in\s*(\d+)\s*pass\(es\),\s*"
        r"D_B\s*<\s*([0-9.eE+-]+),\s*"
        r"rtree=([^,]+),\s*chi_sq=([0-9.eE+-]+),\s*"
        r"margin_m=([0-9.eE+-]+),\s*"
        r"between_frames=([^,]+),\s*prefer_older=([^\s]+)"
    )

    rows: List[Row] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if "frame-to-frame GMM prune summary" not in line:
                continue
            stamp_match = stamp_pattern.search(line)
            summary_match = summary_pattern.search(line)
            if not stamp_match or not summary_match:
                continue
            (
                before,
                after,
                removed,
                merges,
                passes,
                db,
                rtree,
                chi_sq,
                margin,
                between,
                prefer,
            ) = summary_match.groups()
            rows.append(
                {
                    "stamp": stamp_match.group(1),
                    "event": "frame_to_frame_raw",
                    "before_total": before,
                    "after_total": after,
                    "removed_total": removed,
                    "merges": merges,
                    "passes": passes,
                    "bhatt_threshold": db,
                    "rtree": rtree,
                    "chi_sq": chi_sq,
                    "margin_m": margin,
                    "between_frames": between,
                    "prefer_older": prefer,
                }
            )
    return rows


def f(row: Row, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row: Row, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rel_time(values: Sequence[float]) -> List[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return list(values)
    t0 = finite[0]
    return [v - t0 if math.isfinite(v) else v for v in values]


def finite_pairs(
    xs: Iterable[float], ys: Iterable[float]
) -> Tuple[List[float], List[float]]:
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def valid_time_rows(rows: List[Row]) -> List[Row]:
    """Rows with real ROS stamps, sorted by stamp."""
    valid = [r for r in rows if math.isfinite(f(r, "stamp")) and f(r, "stamp") >= 0.0]
    return sorted(valid, key=lambda r: f(r, "stamp"))


def min_stamp(rows: List[Row]) -> Optional[float]:
    """Minimum finite stamp across a list of rows, or None if empty."""
    stamps = [f(r, "stamp") for r in rows if math.isfinite(f(r, "stamp"))]
    return min(stamps) if stamps else None


def save_fig(out_dir: Path, name: str) -> Path:
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def describe(values: Sequence[float]) -> str:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return "n=0"
    clean_sorted = sorted(clean)
    p95 = clean_sorted[min(len(clean_sorted) - 1, int(0.95 * (len(clean_sorted) - 1)))]
    return (
        f"n={len(clean)}, mean={mean(clean):.3f}, median={median(clean):.3f}, "
        f"p95={p95:.3f}, max={max(clean):.3f}"
    )


def workspace_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "gmmslam" / "logs").exists():
            return parent
    return script_path.parents[3]


def resolve_input_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    root_candidate = (root / path).resolve()
    if root_candidate.exists():
        return root_candidate

    gmmslam_candidate = (root / "gmmslam" / path).resolve()
    if gmmslam_candidate.exists():
        return gmmslam_candidate

    logs_candidate = (root / "gmmslam" / "logs" / path).resolve()
    if logs_candidate.exists():
        return logs_candidate

    return root_candidate


def resolve_output_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def newest_run_dir(root: Path) -> Optional[Path]:
    logs_dir = root / "gmmslam" / "logs"
    if not logs_dir.exists():
        return None
    runs = [p for p in logs_dir.glob("gmmslam_cpp_*") if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def has_map_columns(frames: List[Row]) -> bool:
    """True when frames.csv was written by a build that logs map_x/y/z."""
    return bool(frames) and "map_x" in frames[0]


def has_odom_columns(frames: List[Row]) -> bool:
    """True when frames.csv includes odom_x/y/z (smoother odom-frame pose)."""
    return bool(frames) and "odom_x" in frames[0]


def loop_closure_segments(
    frames: List[Row], registration_events: List[Row]
) -> List[Tuple[float, float, float, float]]:
    """Return (x0,y0,x1,y1) pairs for staged loop closures."""
    use_map = has_map_columns(frames)
    x_key = "map_x" if use_map else "est_x"
    y_key = "map_y" if use_map else "est_y"

    poses_by_odom = {}
    for r in frames:
        odom_idx = i(r, "odom_idx", -1)
        x = f(r, x_key)
        y = f(r, y_key)
        if odom_idx >= 0 and math.isfinite(x) and math.isfinite(y):
            poses_by_odom[odom_idx] = (x, y)

    segments = []
    seen = set()
    for r in registration_events:
        if r.get("kind") != "loop" or r.get("event") != "staged":
            continue
        prev_idx = i(r, "prev_idx", -1)
        curr_idx = i(r, "curr_idx", -1)
        edge = (prev_idx, curr_idx)
        if edge in seen:
            continue
        seen.add(edge)
        if prev_idx in poses_by_odom and curr_idx in poses_by_odom:
            x0, y0 = poses_by_odom[prev_idx]
            x1, y1 = poses_by_odom[curr_idx]
            segments.append((x0, y0, x1, y1))
    return segments


# ── Trajectory ───────────────────────────────────────────────────────────────

def plot_trajectory(
    run_dir: Path,
    out_dir: Path,
    frames: List[Row],
    registration_events: List[Row],
) -> Optional[Path]:
    est_lpf = read_tum(run_dir / "estimated_trajectory.tum")
    est_map = read_tum(run_dir / "estimated_trajectory_map.tum")
    gt = read_tum(run_dir / "groundtruth_trajectory.tum")
    if not est_lpf and not est_map and not gt:
        return None

    plt.figure(figsize=(9, 8))
    if gt:
        plt.plot(
            [p[1] for p in gt],
            [p[2] for p in gt],
            label="Ground truth",
            linewidth=2,
            zorder=3,
        )
    if est_map:
        plt.plot(
            [p[1] for p in est_map],
            [p[2] for p in est_map],
            label="SLAM estimate (map frame, raw)",
            linewidth=1.5,
            color="tab:orange",
            zorder=2,
        )
    if est_lpf:
        plt.plot(
            [p[1] for p in est_lpf],
            [p[2] for p in est_lpf],
            label="SLAM estimate (map frame, smoothed)",
            linewidth=1.0,
            color="tab:blue",
            alpha=0.75,
            zorder=1,
        )

    segments = loop_closure_segments(frames, registration_events)
    for idx, (x0, y0, x1, y1) in enumerate(segments):
        plt.plot(
            [x0, x1],
            [y0, y1],
            color="red",
            linewidth=3.0,
            alpha=0.95,
            label="Loop closure edge" if idx == 0 else None,
            zorder=4,
        )
    plt.axis("equal")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Top-Down Trajectory with Loop Closure Edges")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return save_fig(out_dir, "trajectory_xy.png")


# ── Absolute trajectory error ─────────────────────────────────────────────────

def plot_ate(frames: List[Row], out_dir: Path) -> Optional[Path]:
    rows = [r for r in frames if i(r, "has_gt", 0) == 1]
    if not rows:
        return None

    stamps = [f(r, "stamp") for r in rows]
    times = rel_time(stamps)

    plt.figure(figsize=(11, 5))

    if has_odom_columns(frames):
        errors_odom = []
        for r in rows:
            dx = f(r, "odom_x") - f(r, "gt_x")
            dy = f(r, "odom_y") - f(r, "gt_y")
            dz = f(r, "odom_z") - f(r, "gt_z")
            errors_odom.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        xs, ys = finite_pairs(times, errors_odom)
        if xs:
            plt.plot(xs, ys, linewidth=1.4, label="SLAM vs ground truth", color="tab:green")
            plt.legend()

    plt.xlabel(XLABEL_TIME)
    plt.ylabel("Position error (m)")
    plt.title("Absolute Trajectory Error Over Time")
    plt.grid(True, alpha=0.3)
    return save_fig(out_dir, "ate_over_time.png")


# ── Correction snaps ──────────────────────────────────────────────────────────

def plot_correction_snaps(
    frames: List[Row], registration_events: List[Row], out_dir: Path
) -> Optional[Path]:
    """Map-frame position snapping, ATE, and loop closure events."""
    if not has_map_columns(frames):
        return None

    rows = valid_time_rows(frames)
    if not rows:
        return None

    stamps = [f(r, "stamp") for r in rows]
    t0 = stamps[0] if stamps else 0.0
    times = [s - t0 for s in stamps]

    map_x = [f(r, "map_x") for r in rows]
    map_y = [f(r, "map_y") for r in rows]

    gt_rows = [r for r in rows if i(r, "has_gt", 0) == 1]
    gt_stamps = [f(r, "stamp") for r in gt_rows]
    gt_times = [s - t0 for s in gt_stamps]
    gt_x = [f(r, "gt_x") for r in gt_rows]
    gt_y = [f(r, "gt_y") for r in gt_rows]

    loop_times = sorted(
        f(r, "stamp") - t0
        for r in registration_events
        if r.get("kind") == "loop" and r.get("event") == "staged"
        and math.isfinite(f(r, "stamp")) and (f(r, "stamp") - t0) >= 0.0
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for axis_label, map_vals, g_times, g_vals in [
        ("X", map_x, gt_times, gt_x),
        ("Y", map_y, gt_times, gt_y),
    ]:
        xs_m, ys_m = finite_pairs(times, map_vals)
        xs_g, ys_g = finite_pairs(g_times, g_vals)
        if xs_m:
            axes[0].plot(xs_m, ys_m, linewidth=0.9, alpha=0.85, label=f"SLAM {axis_label} (map frame)")
        if xs_g:
            axes[0].plot(xs_g, ys_g, linewidth=0.9, alpha=0.65, linestyle="--",
                         label=f"Ground truth {axis_label}")

    for idx, lt in enumerate(loop_times):
        axes[0].axvline(lt, color="red", linewidth=0.8, alpha=0.5,
                        label="Loop closure" if idx == 0 else None)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Map-frame position: SLAM output vs ground truth")
    axes[0].legend(ncol=4)
    axes[0].grid(True, alpha=0.3)

    if has_odom_columns(rows):
        odom_delta = []
        for r in rows:
            ox, oy = f(r, "odom_x"), f(r, "odom_y")
            gx, gy = f(r, "gt_x"), f(r, "gt_y")
            if i(r, "has_gt", 0) == 1 and all(math.isfinite(v) for v in (ox, oy, gx, gy)):
                odom_delta.append(math.sqrt((ox - gx) ** 2 + (oy - gy) ** 2))
            else:
                odom_delta.append(math.nan)
        xs_o, ys_o = finite_pairs(times, odom_delta)
        if xs_o:
            axes[1].plot(xs_o, ys_o, linewidth=1.3, color="tab:green",
                         label="SLAM output error")
    for idx, lt in enumerate(loop_times):
        axes[1].axvline(lt, color="red", linewidth=0.8, alpha=0.5,
                        label="Loop closure" if idx == 0 else None)
    axes[1].set_ylabel("Position error (m)")
    axes[1].set_title("Absolute trajectory error over time")
    axes[1].set_xlabel(XLABEL_TIME)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    return save_fig(out_dir, "correction_snaps.png")


# ── Frame timing ──────────────────────────────────────────────────────────────

def plot_frame_timing(frames: List[Row], out_dir: Path) -> Optional[Path]:
    if not frames:
        return None
    stamps = [f(r, "stamp") for r in frames]
    times = rel_time(stamps)
    preprocess = [f(r, "preprocess_ms") for r in frames]
    callback = [f(r, "callback_ms") for r in frames]
    plt.figure(figsize=(11, 5))
    xs, ys = finite_pairs(times, preprocess)
    if xs:
        plt.plot(xs, ys, label="Point cloud preprocessing", linewidth=1.1)
    xs, ys = finite_pairs(times, callback)
    if xs:
        plt.plot(xs, ys, label="Full callback (end-to-end)", linewidth=1.1)
    plt.xlabel(XLABEL_TIME)
    plt.ylabel("Processing time (ms)")
    plt.title("Frame Processing Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return save_fig(out_dir, "frame_timing.png")


# ── GMM fitting ───────────────────────────────────────────────────────────────

def plot_gmm_fits(rows: List[Row], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    stamps = [f(r, "stamp") for r in rows]
    times = rel_time(stamps)
    elapsed = [f(r, "elapsed_ms") for r in rows]
    components = [f(r, "component_count") for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    xs, ys = finite_pairs(times, elapsed)
    if xs:
        ax1.plot(xs, ys, label="Fit time", color="tab:blue", linewidth=1.1)
    ax1.set_xlabel(XLABEL_TIME)
    ax1.set_ylabel("GMM fitting time (ms)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    xs, ys = finite_pairs(times, components)
    if xs:
        ax2.plot(
            xs, ys, label="Component count", color="tab:orange", linewidth=1.0, alpha=0.8
        )
    ax2.set_ylabel("Gaussian components per keyframe", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("GMM Fitting Time and Component Count per Keyframe")
    return save_fig(out_dir, "gmm_fit_timing.png")


# ── D2D registration ──────────────────────────────────────────────────────────

def plot_d2d(rows: List[Row], out_dir: Path) -> List[Path]:
    if not rows:
        return []
    paths = []
    kinds = sorted({r.get("kind", "unknown") or "unknown" for r in rows})

    all_finite = [f(r, "stamp") for r in rows if math.isfinite(f(r, "stamp"))]
    t0 = min(all_finite) if all_finite else 0.0

    # Clip y-axis at 99th percentile to suppress startup/GC outliers.
    all_elapsed = sorted(
        f(r, "elapsed_ms") for r in rows if math.isfinite(f(r, "elapsed_ms"))
    )
    n_clip = max(1, int(len(all_elapsed) * 0.99))
    y_clip = all_elapsed[n_clip - 1] * 1.05 if all_elapsed else None
    n_clipped = sum(1 for v in all_elapsed if y_clip is not None and v > y_clip)

    plt.figure(figsize=(11, 5))
    for kind in kinds:
        subset = [r for r in rows if (r.get("kind", "unknown") or "unknown") == kind]
        times = [f(r, "stamp") - t0 for r in subset]
        elapsed = [f(r, "elapsed_ms") for r in subset]
        xs, ys = finite_pairs(times, elapsed)
        if xs:
            plt.scatter(xs, ys, s=14, label=kind.replace("_", " ").title(), alpha=0.75)
    plt.xlabel(XLABEL_TIME)
    plt.ylabel("Registration time (ms)")
    title = "D2D Registration Processing Time by Type"
    if n_clipped > 0:
        title += f"\n(y-axis clipped at 99th percentile — {n_clipped} outlier(s) hidden)"
    plt.title(title)
    if y_clip is not None:
        plt.ylim(bottom=0, top=y_clip)
    plt.legend()
    plt.grid(True, alpha=0.3)
    paths.append(save_fig(out_dir, "d2d_timing_by_type.png"))

    success_rates = []
    counts = []
    for kind in kinds:
        subset = [r for r in rows if (r.get("kind", "unknown") or "unknown") == kind]
        n = len(subset)
        ok = sum(i(r, "success", 0) for r in subset)
        counts.append(n)
        success_rates.append(ok / n if n else 0.0)
    plt.figure(figsize=(8, 5))
    bar_labels = [k.replace("_", " ").title() for k in kinds]
    plt.bar(bar_labels, success_rates)
    for idx, (rate, count) in enumerate(zip(success_rates, counts)):
        plt.text(idx, rate + 0.01, f"{rate * 100:.1f}%\nn={count}", ha="center",
                 va="bottom", fontsize=12)
    plt.ylim(0, 1.15)
    plt.xlabel("Registration type")
    plt.ylabel("Success rate (fraction)")
    plt.title("D2D Registration Success Rate by Type")
    plt.grid(True, axis="y", alpha=0.3)
    paths.append(save_fig(out_dir, "d2d_success_rate.png"))
    return paths


# ── Loop closure quality — split into two figures ─────────────────────────────

def _loop_closure_t0(loop_rows: List[Row], global_graph_opt: List[Row]) -> float:
    """Single time reference: minimum stamp across loop events and global graph rows."""
    candidates: List[float] = []
    for r in loop_rows:
        s = f(r, "stamp")
        if math.isfinite(s):
            candidates.append(s)
    for r in global_graph_opt:
        s = f(r, "stamp")
        if math.isfinite(s):
            candidates.append(s)
    return min(candidates) if candidates else 0.0


def plot_loop_closure_scores(
    registration_events: List[Row], out_dir: Path
) -> Optional[Path]:
    """Loop closure registration score over time (log scale, full-size panel).

    Shows failed-at-worker candidates (sentinel score → plotted at a visual
    floor) and staged-to-global-graph candidates. Accepted-but-not-staged rows
    are omitted to keep the plot focused.
    """
    loop_rows = [r for r in registration_events if r.get("kind") == "loop"]
    if not loop_rows:
        return None

    all_stamps = [f(r, "stamp") for r in loop_rows if math.isfinite(f(r, "stamp"))]
    if not all_stamps:
        return None
    t0 = min(all_stamps)

    failed = [r for r in loop_rows if r.get("event") == "worker_failed"]
    staged = [r for r in loop_rows if r.get("event") == "staged"]

    # Detect threshold: minimum positive score among staged closures.
    staged_scores = [
        f(r, "score") for r in staged
        if math.isfinite(f(r, "score")) and f(r, "score") > 0
    ]
    detect_thresh = min(staged_scores) if staged_scores else None

    # Floor for failed candidates whose score is a sentinel (negative / zero).
    # Placed one decade below the minimum real score so they are visible but
    # clearly below the operating range.
    all_pos = [f(r, "score") for r in loop_rows
               if math.isfinite(f(r, "score")) and f(r, "score") > 0]
    score_floor = min(all_pos) * 0.1 if all_pos else 1e-5

    fig, ax = plt.subplots(figsize=(13, 6))

    # Failed at worker — sentinel scores mapped to floor.
    xs_f, ys_f = [], []
    for r in failed:
        t = f(r, "stamp") - t0
        s = f(r, "score")
        if not math.isfinite(t):
            continue
        xs_f.append(t)
        ys_f.append(s if (math.isfinite(s) and s > 0) else score_floor)
    if xs_f:
        ax.scatter(xs_f, ys_f, s=25, color="lightgrey", marker="x", alpha=0.8,
                   label="Failed at worker (score unavailable → shown at floor)",
                   zorder=1)

    # Staged to global graph.
    xs_s, ys_s = [], []
    for r in staged:
        t = f(r, "stamp") - t0
        s = f(r, "score")
        if math.isfinite(t) and math.isfinite(s) and s > 0:
            xs_s.append(t)
            ys_s.append(s)
    if xs_s:
        ax.scatter(xs_s, ys_s, s=35, color="tab:orange", marker="o", alpha=0.85,
                   edgecolors="red", linewidths=1.0,
                   label="Staged to global graph", zorder=3)

    if detect_thresh is not None:
        ax.axhline(detect_thresh, color="red", linestyle="--", linewidth=1.2,
                   label=f"Detect threshold ≈ {detect_thresh:.3f}")

    # Dashed line showing where failed candidates are placed.
    if xs_f:
        ax.axhline(score_floor, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_TIME)
    ax.set_ylabel("Registration score (log scale)")
    ax.set_title("Loop Closure Candidate Score Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return save_fig(out_dir, "loop_closure_scores.png")


def plot_loop_closure_diagnostics(
    registration_events: List[Row], global_graph_opt: List[Row], out_dir: Path
) -> Optional[Path]:
    """Translation shift by D2D + global graph optimization time."""
    loop_rows = [r for r in registration_events if r.get("kind") == "loop"]
    if not loop_rows:
        return None

    # Unified time reference — avoids negative x values when gg starts before
    # the first loop event.
    t0 = _loop_closure_t0(loop_rows, valid_time_rows(global_graph_opt))

    accepted = [r for r in loop_rows if r.get("event") == "accepted"]
    staged   = [r for r in loop_rows if r.get("event") == "staged"]

    HIGH_ROT = 20.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1 — how much D2D moved the position (accepted only)
    xs_d, ys_d, colors_d = [], [], []
    for r in accepted:
        t = f(r, "stamp") - t0
        dt = f(r, "delta_t_m")
        ir = f(r, "init_r_deg")
        if not math.isfinite(t) or not math.isfinite(dt) or t < 0:
            continue
        xs_d.append(t)
        ys_d.append(dt)
        colors_d.append("red" if (math.isfinite(ir) and ir > HIGH_ROT) else "tab:green")
    if xs_d:
        axes[0].scatter(xs_d, ys_d, s=25, c=colors_d, alpha=0.8, zorder=2)
        axes[0].scatter([], [], s=25, color="red",
                        label=f"Initial rotation > {HIGH_ROT:.0f}° (uncertain)")
        axes[0].scatter([], [], s=25, color="tab:green",
                        label=f"Initial rotation ≤ {HIGH_ROT:.0f}°")
    axes[0].axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axes[0].set_ylabel("Translation correction (m)")
    axes[0].set_title("Translation Moved by D2D Registration (accepted loop closures)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — global graph iSAM2 optimization time (instability proxy)
    gg = valid_time_rows(global_graph_opt)
    if gg:
        gg_times  = [f(r, "stamp") - t0 for r in gg]
        gg_est_ms = [f(r, "estimate_ms") for r in gg]
        xs_g, ys_g = finite_pairs(gg_times, gg_est_ms)
        if xs_g:
            axes[1].plot(xs_g, ys_g, linewidth=0.9, color="tab:purple",
                         label="Global graph optimization time")
            staged_stamps = {f(r, "stamp") for r in staged if math.isfinite(f(r, "stamp"))}
            spike_xs, spike_ys = [], []
            for t_g, ms in zip(xs_g, ys_g):
                abs_t = t_g + t0
                if any(abs(abs_t - ls) < 0.5 for ls in staged_stamps):
                    spike_xs.append(t_g)
                    spike_ys.append(ms)
            if spike_xs:
                axes[1].scatter(spike_xs, spike_ys, s=35, color="red", zorder=3,
                                label="Commit after loop closure")
        axes[1].legend()
    axes[1].set_xlabel(XLABEL_TIME)
    axes[1].set_ylabel("Graph optimization time (ms)")
    axes[1].set_title(
        "Global Pose Graph Optimization Time (spikes indicate large corrections)"
    )
    axes[1].grid(True, alpha=0.3)

    return save_fig(out_dir, "loop_closure_diagnostics.png")


# ── Loop closure quality — combined 3-panel figure ───────────────────────────

def plot_loop_closure_quality(
    registration_events: List[Row], global_graph_opt: List[Row], out_dir: Path
) -> Optional[Path]:
    """Three-panel combined figure: score (log), translation shift, gg timing."""
    loop_rows = [r for r in registration_events if r.get("kind") == "loop"]
    if not loop_rows:
        return None

    t0 = _loop_closure_t0(loop_rows, valid_time_rows(global_graph_opt))

    failed = [r for r in loop_rows if r.get("event") == "worker_failed"]
    staged  = [r for r in loop_rows if r.get("event") == "staged"]

    staged_scores = [
        f(r, "score") for r in staged
        if math.isfinite(f(r, "score")) and f(r, "score") > 0
    ]
    detect_thresh = min(staged_scores) if staged_scores else None

    all_pos = [f(r, "score") for r in loop_rows
               if math.isfinite(f(r, "score")) and f(r, "score") > 0]
    score_floor = min(all_pos) * 0.1 if all_pos else 1e-5

    HIGH_ROT = 20.0
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # ── Panel 1: score (log scale, no Accepted) ───────────────────────────────
    xs_f, ys_f = [], []
    for r in failed:
        t = f(r, "stamp") - t0
        s = f(r, "score")
        if not math.isfinite(t):
            continue
        xs_f.append(t)
        ys_f.append(s if (math.isfinite(s) and s > 0) else score_floor)
    if xs_f:
        axes[0].scatter(xs_f, ys_f, s=25, color="lightgrey", marker="x", alpha=0.8,
                        label="Failed at worker (score unavailable → shown at floor)",
                        zorder=1)

    xs_s, ys_s = [], []
    for r in staged:
        t = f(r, "stamp") - t0
        s = f(r, "score")
        if math.isfinite(t) and math.isfinite(s) and s > 0:
            xs_s.append(t)
            ys_s.append(s)
    if xs_s:
        axes[0].scatter(xs_s, ys_s, s=35, color="tab:orange", marker="o", alpha=0.85,
                        edgecolors="red", linewidths=1.0,
                        label="Staged to global graph", zorder=3)

    if detect_thresh is not None:
        axes[0].axhline(detect_thresh, color="red", linestyle="--", linewidth=1.2,
                        label=f"Detect threshold ≈ {detect_thresh:.3f}")
    if xs_f:
        axes[0].axhline(score_floor, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Registration score (log scale)")
    axes[0].set_title("Loop Closure Candidate Score Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # ── Panel 2: translation moved by D2D (accepted only) ────────────────────
    accepted = [r for r in loop_rows if r.get("event") == "accepted"]
    xs_d, ys_d, colors_d = [], [], []
    for r in accepted:
        t = f(r, "stamp") - t0
        dt = f(r, "delta_t_m")
        ir = f(r, "init_r_deg")
        if not math.isfinite(t) or not math.isfinite(dt) or t < 0:
            continue
        xs_d.append(t)
        ys_d.append(dt)
        colors_d.append("red" if (math.isfinite(ir) and ir > HIGH_ROT) else "tab:green")
    if xs_d:
        axes[1].scatter(xs_d, ys_d, s=25, c=colors_d, alpha=0.8, zorder=2)
        axes[1].scatter([], [], s=25, color="red",
                        label=f"Initial rotation > {HIGH_ROT:.0f}° (uncertain)")
        axes[1].scatter([], [], s=25, color="tab:green",
                        label=f"Initial rotation ≤ {HIGH_ROT:.0f}°")
    axes[1].axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axes[1].set_ylabel("Translation correction (m)")
    axes[1].set_title("Translation Moved by D2D Registration (accepted loop closures)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: global graph iSAM2 optimization time ────────────────────────
    gg = valid_time_rows(global_graph_opt)
    if gg:
        gg_times  = [f(r, "stamp") - t0 for r in gg]
        gg_est_ms = [f(r, "estimate_ms") for r in gg]
        xs_g, ys_g = finite_pairs(gg_times, gg_est_ms)
        if xs_g:
            axes[2].plot(xs_g, ys_g, linewidth=0.9, color="tab:purple",
                         label="Global graph optimization time")
            staged_stamps = {f(r, "stamp") for r in staged if math.isfinite(f(r, "stamp"))}
            spike_xs, spike_ys = [], []
            for t_g, ms in zip(xs_g, ys_g):
                if any(abs((t_g + t0) - ls) < 0.5 for ls in staged_stamps):
                    spike_xs.append(t_g)
                    spike_ys.append(ms)
            if spike_xs:
                axes[2].scatter(spike_xs, spike_ys, s=35, color="red", zorder=3,
                                label="Commit after loop closure")
        axes[2].legend()
    axes[2].set_xlabel(XLABEL_TIME)
    axes[2].set_ylabel("Graph optimization time (ms)")
    axes[2].set_title(
        "Global Pose Graph Optimization Time (spikes indicate large corrections)"
    )
    axes[2].grid(True, alpha=0.3)

    return save_fig(out_dir, "loop_closure_quality.png")


# ── iSAM2 optimization timings ────────────────────────────────────────────────

def plot_isam_timings(
    smoother: List[Row], global_graph: List[Row], out_dir: Path
) -> Optional[Path]:
    smoother = valid_time_rows(smoother)
    global_graph = valid_time_rows(global_graph)
    if not smoother and not global_graph:
        return None
    all_stamps = [f(r, "stamp") for r in smoother] + [
        f(r, "stamp") for r in global_graph
    ]
    t0 = min(s for s in all_stamps if math.isfinite(s))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    if smoother:
        times = [f(r, "stamp") - t0 for r in smoother]
        total = [f(r, "total_ms") for r in smoother]
        xs, ys = finite_pairs(times, total)
        if xs:
            axes[0].plot(xs, ys, label="Fixed-lag smoother", linewidth=0.9)
            axes[0].legend()
    axes[0].set_ylabel("Optimization time (ms)")
    axes[0].set_title("Fixed-Lag Smoother Optimization Time per Solve")
    axes[0].grid(True, alpha=0.3)

    if global_graph:
        times = [f(r, "stamp") - t0 for r in global_graph]
        total = [f(r, "total_ms") for r in global_graph]
        xs, ys = finite_pairs(times, total)
        if xs:
            axes[1].scatter(xs, ys, label="Global pose graph", s=14)
            axes[1].legend()
    axes[1].set_xlabel(XLABEL_TIME)
    axes[1].set_ylabel("Optimization time (ms)")
    axes[1].set_title("Global Pose Graph Optimization Time per Commit")
    axes[1].grid(True, alpha=0.3)
    return save_fig(out_dir, "isam2_optimization_timing.png")


# ── Global map — split into two figures ───────────────────────────────────────

def _dedup_global_map(rows: List[Row]) -> List[Row]:
    """Keep only the last row per timestamp (multiple events share a stamp)."""
    seen: dict = {}
    for r in rows:
        s = f(r, "stamp")
        if math.isfinite(s) and s >= 0.0:
            seen[s] = r
    return [seen[s] for s in sorted(seen)]


def plot_global_map_gaussians(rows: List[Row], out_dir: Path) -> Optional[Path]:
    """Gaussian count growth and pruning deltas."""
    if not rows:
        return None

    # Compute per-event deltas BEFORE deduplication so that co-timestamped
    # events (e.g. submap_finalized + internal_submap_prune at the same stamp)
    # each get their own correct delta rather than being collapsed into one.
    t0 = f(rows[0], "stamp")
    TRACKED = {"submap_finalized", "internal_submap_prune", "cross_submap_prune"}
    delta_rows = []
    prev_total: Optional[float] = None
    for r in rows:
        total = f(r, "total_gaussians")
        if not math.isfinite(total):
            continue
        t = f(r, "stamp") - t0
        delta = 0.0 if prev_total is None else total - prev_total
        prev_total = total
        event = r.get("event", "unknown")
        if event in TRACKED:
            delta_rows.append((t, delta, event))

    # Dedup only for the cumulative top curve (last value per timestamp)
    deduped = _dedup_global_map(rows)
    stamps = [f(r, "stamp") for r in deduped]
    times = [s - t0 for s in stamps]
    gaussians = [f(r, "total_gaussians") for r in deduped]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    xs, ys = finite_pairs(times, gaussians)
    if xs:
        axes[0].step(xs, ys, where="post", label="Total Gaussians", color="tab:blue")
    axes[0].set_ylabel("Total Gaussian components in map")
    axes[0].set_title("Global Map: Total Gaussian Count Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    label_map = {
        "submap_finalized": "Submap finalized",
        "internal_submap_prune": "Internal submap pruning",
        "cross_submap_prune": "Cross-submap pruning",
    }
    for event, color in [
        ("submap_finalized", "tab:green"),
        ("internal_submap_prune", "tab:orange"),
        ("cross_submap_prune", "tab:red"),
    ]:
        subset = [(x, d) for x, d, e in delta_rows if e == event and math.isfinite(d)]
        if subset:
            xs_event = [x for x, _ in subset]
            ys_event = [d for _, d in subset]
            axes[1].scatter(xs_event, ys_event, s=20, alpha=0.8, color=color,
                            label=label_map[event])
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(XLABEL_TIME)
    axes[1].set_ylabel("Change in Gaussian count")
    axes[1].set_title("Map Size Change per Event (positive = growth, negative = pruning)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    return save_fig(out_dir, "global_map_gaussians.png")


def plot_global_map_submaps(rows: List[Row], out_dir: Path) -> Optional[Path]:
    """Submap counts and pending operation backlog."""
    rows = _dedup_global_map(rows)
    if not rows:
        return None
    stamps = [f(r, "stamp") for r in rows]
    t0 = stamps[0]
    times = [s - t0 for s in stamps]
    finalized = [f(r, "finalized_submaps") for r in rows]
    optimized = [f(r, "optimized_submaps") for r in rows]
    pending_overlap = [f(r, "pending_overlap_registrations") for r in rows]
    pending_finalize = [f(r, "pending_submap_finalizations") for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    xs, ys = finite_pairs(times, finalized)
    if xs:
        axes[0].step(xs, ys, where="post", label="Finalized submaps", color="tab:orange")
    xs, ys = finite_pairs(times, optimized)
    if xs:
        axes[0].step(xs, ys, where="post", label="Optimized submaps", color="tab:green")
    axes[0].set_ylabel("Number of submaps")
    axes[0].set_title("Global Map: Submap Count Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    xs, ys = finite_pairs(times, pending_overlap)
    if xs:
        axes[1].step(xs, ys, where="post", label="Pending overlap registrations",
                     color="tab:purple")
    xs, ys = finite_pairs(times, pending_finalize)
    if xs:
        axes[1].step(xs, ys, where="post", label="Pending submap finalizations",
                     color="tab:brown")
    axes[1].set_xlabel(XLABEL_TIME)
    axes[1].set_ylabel("Pending operations")
    axes[1].set_title("Global Map: Pending Operation Backlog")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    return save_fig(out_dir, "global_map_submaps.png")


# ── Pruning stats ─────────────────────────────────────────────────────────────

def plot_pruning_stats(
    rows: List[Row], frame_rows: List[Row], out_dir: Path
) -> Optional[Path]:
    rows = valid_time_rows(rows)
    frame_rows = valid_time_rows(frame_rows)
    if not rows and not frame_rows:
        return None

    first_stamps = []
    if rows:
        first_stamps.append(f(rows[0], "stamp"))
    if frame_rows:
        first_stamps.append(f(frame_rows[0], "stamp"))
    t0 = min(s for s in first_stamps if math.isfinite(s))
    times = [f(r, "stamp") - t0 for r in rows]
    elapsed = [f(r, "elapsed_ms") for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if rows:
        event_labels = {
            "submap_finalized": "Submap finalized",
            "internal_submap_prune": "Internal pruning",
            "cross_submap_prune": "Cross-submap pruning",
        }
        for event in sorted({r.get("event", "unknown") for r in rows}):
            subset = [r for r in rows if r.get("event", "unknown") == event]
            xs = [f(r, "stamp") - t0 for r in subset]
            ys = [f(r, "removed_total") for r in subset]
            xs, ys = finite_pairs(xs, ys)
            if xs:
                label = event_labels.get(event, event.replace("_", " ").title())
                axes[0].scatter(xs, ys, s=20, label=label, alpha=0.75)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("Gaussians removed")
    axes[0].set_title("Global Map Pruning: Gaussians Removed per Event")
    if rows:
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    xs, ys = finite_pairs(times, elapsed)
    if xs:
        axes[1].plot(xs, ys, linewidth=1.1)
    axes[1].set_ylabel("Pruning time (ms)")
    axes[1].set_title("Global Map Pruning: Time Spent per Event")
    axes[1].grid(True, alpha=0.3)

    if frame_rows:
        xs = [f(r, "stamp") - t0 for r in frame_rows]
        removed = [f(r, "removed_total") for r in frame_rows]
        merges = [f(r, "merges") for r in frame_rows]
        xs_removed, ys_removed = finite_pairs(xs, removed)
        xs_merges, ys_merges = finite_pairs(xs, merges)
        if xs_removed:
            axes[2].scatter(xs_removed, ys_removed, s=16, alpha=0.75,
                            label="Components removed")
        if xs_merges:
            axes[2].plot(xs_merges, ys_merges, linewidth=0.9, alpha=0.7,
                         label="Merge operations")
        axes[2].legend()
    axes[2].set_xlabel(XLABEL_TIME)
    axes[2].set_ylabel("Component count")
    axes[2].set_title("Frame-to-Frame GMM Pruning: Removed Components and Merges")
    axes[2].grid(True, alpha=0.3)
    return save_fig(out_dir, "gaussian_pruning_stats.png")


# ── System resource usage ─────────────────────────────────────────────────────

def plot_processes(rows: List[Row], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    by_time: Dict[float, Dict[str, float]] = defaultdict(
        lambda: {"cpu": 0.0, "rss": 0.0}
    )
    for r in rows:
        t = f(r, "elapsed_s")
        if not math.isfinite(t):
            continue
        by_time[t]["cpu"] += f(r, "cpu_percent", 0.0)
        by_time[t]["rss"] += f(r, "rss_kb", 0.0) / 1024.0
    if not by_time:
        return None
    times = sorted(by_time)
    cpu = [by_time[t]["cpu"] for t in times]
    rss = [by_time[t]["rss"] for t in times]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(times, cpu, label="CPU", color="tab:blue")
    ax1.set_xlabel("Wall-clock time (s)")
    ax1.set_ylabel("Total CPU usage (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(times, rss, label="Memory (RSS)", color="tab:orange")
    ax2.set_ylabel("Total memory usage (MiB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("System Resource Usage (All Processes Combined)")
    return save_fig(out_dir, "process_resource_usage.png")


def plot_gpu(rows: List[Row], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    samples = list(range(len(rows)))
    util = [f(r, "gpu_util_percent") for r in rows]
    mem = [f(r, "mem_used_mib") for r in rows]
    if not samples:
        return None
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(samples, util, color="tab:blue")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("GPU utilization (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(samples, mem, color="tab:orange")
    ax2.set_ylabel("GPU memory used (MiB)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("GPU Utilization and Memory Usage")
    return save_fig(out_dir, "gpu_usage.png")


# ── Summary text ──────────────────────────────────────────────────────────────

def write_summary(
    out_dir: Path,
    frames: List[Row],
    gmm: List[Row],
    d2d: List[Row],
    registration_events: List[Row],
    smoother: List[Row],
    global_graph: List[Row],
    global_map: List[Row],
    pruning: List[Row],
    frame_pruning: List[Row],
) -> Path:
    lines = ["GMM-SLAM benchmark graph summary", ""]
    if frames:
        lines.append(f"Frames: {len(frames)}")
        lines.append(
            f"Callback latency ms: {describe([f(r, 'callback_ms') for r in frames])}"
        )
    if gmm:
        lines.append(f"GMM fits: {len(gmm)}")
        lines.append(
            f"GMM fit elapsed ms: {describe([f(r, 'elapsed_ms') for r in gmm])}"
        )
    if d2d:
        lines.append(f"D2D results: {len(d2d)}")
        for kind in sorted({r.get("kind", "unknown") for r in d2d}):
            subset = [r for r in d2d if r.get("kind", "unknown") == kind]
            ok = sum(i(r, "success", 0) for r in subset)
            lines.append(
                f"D2D {kind}: success={ok}/{len(subset)} "
                f"({(ok / len(subset) * 100.0) if subset else 0.0:.1f}%), "
                f"elapsed_ms {describe([f(r, 'elapsed_ms') for r in subset])}"
            )
    loop_edges = loop_closure_segments(frames, registration_events)
    if loop_edges:
        lines.append(f"Loop closure trajectory segments plotted: {len(loop_edges)}")
    if smoother:
        lines.append(f"Fixed-lag smoother solves: {len(smoother)}")
        lines.append(
            f"Fixed-lag total ms: {describe([f(r, 'total_ms') for r in smoother])}"
        )
    if global_graph:
        lines.append(f"Global graph commits: {len(global_graph)}")
        lines.append(
            f"Global graph total ms: {describe([f(r, 'total_ms') for r in global_graph])}"
        )
    if global_map:
        last = global_map[-1]
        lines.append(
            "Final global map: "
            f"optimized_submaps={last.get('optimized_submaps')}, "
            f"finalized_submaps={last.get('finalized_submaps')}, "
            f"total_gaussians={last.get('total_gaussians')}"
        )
    if pruning:
        applied = [r for r in pruning if i(r, "success", 0) == 1]
        removed = [f(r, "removed_total") for r in applied]
        elapsed = [f(r, "elapsed_ms") for r in pruning]
        lines.append(
            f"Gaussian pruning events: {len(pruning)} (applied={len(applied)})"
        )
        lines.append(f"Gaussian pruning removed components: {describe(removed)}")
        lines.append(f"Gaussian pruning elapsed ms: {describe(elapsed)}")
    if frame_pruning:
        lines.append(f"Frame-to-frame pruning summaries: {len(frame_pruning)}")
        lines.append(
            "Frame-to-frame pruning removed components: "
            f"{describe([f(r, 'removed_total') for r in frame_pruning])}"
        )
        lines.append(
            "Frame-to-frame pruning merge counts: "
            f"{describe([f(r, 'merges') for r in frame_pruning])}"
        )
    path = out_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", help="Benchmark run directory")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory for graphs"
    )
    args = parser.parse_args()

    root = workspace_root()
    if args.run_dir is None:
        run_dir = newest_run_dir(root)
        if run_dir is None:
            raise SystemExit(
                f"no gmmslam_cpp_* runs found under {root / 'gmmslam' / 'logs'}"
            )
    else:
        run_dir = resolve_input_path(args.run_dir, root)
    if not run_dir.exists():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    out_dir = (
        resolve_output_path(args.out_dir, root)
        if args.out_dir
        else (root / "benchmark_graphs" / run_dir.name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read_csv(run_dir / "frames.csv")
    gmm = read_csv(run_dir / "gmm_fits.csv")
    d2d = read_csv(run_dir / "d2d_timings.csv")
    smoother = read_csv(run_dir / "smoother_optimization.csv")
    global_graph = read_csv(run_dir / "global_graph_optimization.csv")
    global_map = read_csv(run_dir / "global_map_stats.csv")
    pruning = read_csv(run_dir / "global_pruning_stats.csv")
    frame_pruning = read_frame_pruning_console(run_dir / "console.txt")
    registration_events = read_csv(run_dir / "registration_events.csv")
    processes = read_tsv(run_dir / "processes.tsv")
    gpu = read_csv(run_dir / "gpu.csv")

    generated: List[Path] = []
    for path in [
        plot_trajectory(run_dir, out_dir, frames, registration_events),
        plot_ate(frames, out_dir),
        plot_correction_snaps(frames, registration_events, out_dir),
        plot_loop_closure_scores(registration_events, out_dir),
        plot_loop_closure_diagnostics(registration_events, global_graph, out_dir),
        plot_loop_closure_quality(registration_events, global_graph, out_dir),
        plot_frame_timing(frames, out_dir),
        plot_gmm_fits(gmm, out_dir),
        plot_isam_timings(smoother, global_graph, out_dir),
        plot_global_map_gaussians(global_map, out_dir),
        plot_global_map_submaps(global_map, out_dir),
        plot_pruning_stats(pruning, frame_pruning, out_dir),
        plot_processes(processes, out_dir),
        plot_gpu(gpu, out_dir),
    ]:
        if path is not None:
            generated.append(path)
    generated.extend(plot_d2d(d2d, out_dir))
    generated.append(
        write_summary(
            out_dir,
            frames,
            gmm,
            d2d,
            registration_events,
            smoother,
            global_graph,
            global_map,
            pruning,
            frame_pruning,
        )
    )

    print(f"Wrote {len(generated)} files to {out_dir}")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
