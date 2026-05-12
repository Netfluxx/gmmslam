#!/usr/bin/env python3
"""
Plot SOLiD descriptor benchmark CSV written by solid_descriptor_benchmark.

CSV header:
  t_sec,yaw_rad,delta_yaw_deg,cos_range,solid_ms,n_pts,ref_valid

Examples:
  python3 plot_solid_bench.py /tmp/solid_bench.csv -o bench.png
  python3 plot_solid_bench.py yaw.csv arc.csv --labels "yaw only" "arc" -o compare.png --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_solid_csv(path: Path) -> dict[str, np.ndarray]:
    """Load benchmark CSV into float arrays (ref_valid as float 0/1)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        expected = {
            "t_sec",
            "yaw_rad",
            "delta_yaw_deg",
            "cos_range",
            "solid_ms",
            "n_pts",
            "ref_valid",
        }
        cols = set(reader.fieldnames)
        if not expected <= cols:
            missing = expected - cols
            raise ValueError(f"{path}: missing columns {missing}, have {reader.fieldnames}")

        buckets: dict[str, list[float]] = {k: [] for k in expected}
        for row in reader:
            for k in expected:
                buckets[k].append(float(row[k]))

    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def plot_runs(
    runs: list[tuple[str, dict[str, np.ndarray]]],
    out_path: Path | None,
    show: bool,
    t0_relative: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    ax_cos_t, ax_yaw_t, ax_scatter = axes[0]
    ax_lat, ax_npts, ax_hist = axes[1]

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(runs), 1)))
    xlab = "t (s)" + (" from start" if t0_relative else "")

    for i, (label, d) in enumerate(runs):
        c = colors[i]
        t = d["t_sec"].copy()
        if t0_relative and t.size:
            t = t - t[0]
        valid = d["ref_valid"] > 0.5
        if not np.any(valid):
            valid = np.ones_like(valid, dtype=bool)

        ax_cos_t.plot(t[valid], d["cos_range"][valid], label=label, color=c, lw=1.2)
        ax_yaw_t.plot(t[valid], d["delta_yaw_deg"][valid], label=label, color=c, lw=1.2)

        dy = d["delta_yaw_deg"][valid]
        cr = d["cos_range"][valid]
        ax_scatter.scatter(dy, cr, s=10, alpha=0.5, label=label, color=c, edgecolors="none")

        ax_lat.plot(t[valid], d["solid_ms"][valid], label=label, color=c, lw=0.9, alpha=0.85)
        ax_npts.plot(t[valid], d["n_pts"][valid], label=label, color=c, lw=0.9, alpha=0.85)

        ax_hist.hist(
            d["cos_range"][valid],
            bins=32,
            alpha=0.35,
            label=label,
            color=c,
            density=True,
        )

    ax_cos_t.set_ylabel("cos_range")
    ax_cos_t.set_title("Descriptor similarity vs time")
    ax_cos_t.grid(True, alpha=0.3)
    ax_cos_t.legend(loc="best", fontsize=8)

    ax_yaw_t.set_ylabel("delta_yaw_deg")
    ax_yaw_t.set_title("Yaw offset vs time (vs reference pose)")
    ax_yaw_t.grid(True, alpha=0.3)
    ax_yaw_t.legend(loc="best", fontsize=8)

    ax_scatter.set_xlabel("delta_yaw_deg")
    ax_scatter.set_ylabel("cos_range")
    ax_scatter.set_title("Similarity vs yaw offset")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc="best", fontsize=8)

    ax_lat.set_xlabel(xlab)
    ax_lat.set_ylabel("solid_ms")
    ax_lat.set_title("SOLiD compute time")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="best", fontsize=8)

    ax_npts.set_xlabel(xlab)
    ax_npts.set_ylabel("n_pts")
    ax_npts.set_title("Points after preprocess")
    ax_npts.grid(True, alpha=0.3)
    ax_npts.legend(loc="best", fontsize=8)

    ax_hist.set_xlabel("cos_range")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("cos_range distribution")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc="best", fontsize=8)

    fig.suptitle("SOLiD benchmark", fontsize=12)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path, help="One or more benchmark CSV files")
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (default: file stems)",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Save figure to this path (png/pdf/...)")
    p.add_argument("--show", action="store_true", help="Open an interactive window")
    p.add_argument(
        "--no-relative-time",
        action="store_true",
        help="Use absolute sim time on x-axis instead of t - t0 per run",
    )
    args = p.parse_args()

    if args.labels is not None and len(args.labels) != len(args.csv):
        p.error(f"Need {len(args.csv)} --labels, got {len(args.labels)}")

    labels = args.labels if args.labels else [path.stem for path in args.csv]
    runs: list[tuple[str, dict[str, np.ndarray]]] = []
    for path, lab in zip(args.csv, labels):
        if not path.is_file():
            p.error(f"Not a file: {path}")
        runs.append((lab, load_solid_csv(path)))

    t0_relative = not args.no_relative_time
    out = args.output
    if out is None and not args.show:
        out = Path("solid_bench_plot.png")

    plot_runs(runs, out_path=out, show=args.show, t0_relative=t0_relative)


if __name__ == "__main__":
    main()
