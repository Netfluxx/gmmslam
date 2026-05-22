#!/usr/bin/env python3
"""
Plot SOLiD descriptor similarity for the arc benchmark.

The script parses benchmark log lines containing position, heading difference,
and similarity channels. It computes displacement relative to the first logged
pose, then plots similarity against both XY displacement and heading change.
"""

import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# =====================================================================
#  CONFIG -- edit freely
# =====================================================================
INPUT_LOG = "solid_arc_60fov.log"
OUTPUT_PNG = "solid_arc_60fov.png"

TITLE = "SOLiD descriptor robustness during arc motion (60 deg FOV)"
SIM_YLABEL = "Cosine similarity to reference descriptor"
DIST_XLABEL = "XY displacement from reference pose [m]"
YAW_XLABEL = "Heading difference from reference [deg]"

CHANNELS = [
    ("solid", "Full SOLiD", "#8e44ad"),
    ("range", "R-SOLiD (range, rotation-invariant)", "#2980b9"),
    ("angle", "A-SOLiD (azimuth)", "#27ae60"),
    ("polar", "Polar (rotation-sensitive)", "#c0392b"),
]

SHOW_SCATTER = True
SHOW_TRENDLINE = True
N_BINS = 36
YLIM = (0.30, 1.01)
FIGSIZE = (11.0, 8.0)
DPI = 200
# =====================================================================


def parse_log(path):
    """Return arrays for time, pose, heading difference, and similarity."""
    pat = re.compile(
        r"t=([-\d.]+).*?"
        r"p=\(([-\d.]+),([-\d.]+),([-\d.]+)\)\s+"
        r"d_yaw=([-\d.]+)deg\s+"
        r"solid=([\d.]+)\s+range=([\d.]+)\s+angle=([\d.]+)\s+polar=([\d.]+)"
    )
    rows = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append(tuple(map(float, m.groups())))

    if not rows:
        sys.exit(f"No data rows parsed from {path!r} -- check the format.")

    arr = np.array(rows)
    x0, y0, z0 = arr[0, 1], arr[0, 2], arr[0, 3]
    dx = arr[:, 1] - x0
    dy = arr[:, 2] - y0
    dz = arr[:, 3] - z0

    data = {
        "t": arr[:, 0],
        "x": arr[:, 1],
        "y": arr[:, 2],
        "z": arr[:, 3],
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "disp_xy": np.hypot(dx, dy),
        "disp_3d": np.sqrt(dx * dx + dy * dy + dz * dz),
        "d_yaw": arr[:, 4],
        "solid": arr[:, 5],
        "range": arr[:, 6],
        "angle": arr[:, 7],
        "polar": arr[:, 8],
    }

    print(
        f"parsed {len(arr)} rows; "
        f"XY displacement {data['disp_xy'].min():.2f} to {data['disp_xy'].max():.2f} m; "
        f"heading {data['d_yaw'].min():.1f} to {data['d_yaw'].max():.1f} deg"
    )
    return data


def binned_mean(x, y, n_bins):
    edges = np.linspace(x.min() - 1e-6, x.max() + 1e-6, n_bins + 1)
    idx = np.digitize(x, edges)
    cx, cy = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.any():
            cx.append(0.5 * (edges[b - 1] + edges[b]))
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


def plot_channels(ax, x, data, xlabel):
    for key, label, colour in CHANNELS:
        y = data[key]
        if SHOW_SCATTER:
            ax.scatter(x, y, s=12, alpha=0.20, color=colour, zorder=2)
        if SHOW_TRENDLINE:
            bx, by = binned_mean(x, y, N_BINS)
            ax.plot(bx, by, "-o", color=colour, lw=2, ms=4, label=label, zorder=4)
        else:
            ax.scatter([], [], color=colour, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(SIM_YLABEL)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.3)


def main():
    in_log = sys.argv[1] if len(sys.argv) > 1 else INPUT_LOG
    out_png = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PNG

    data = parse_log(in_log)

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharey=True)
    fig.suptitle(TITLE)

    plot_channels(axes[0], data["disp_xy"], data, DIST_XLABEL)
    axes[0].set_xlim(data["disp_xy"].min() - 0.1, data["disp_xy"].max() + 0.1)

    plot_channels(axes[1], data["d_yaw"], data, YAW_XLABEL)
    axes[1].axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_xlim(data["d_yaw"].min() - 5, data["d_yaw"].max() + 5)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", framealpha=0.95, fontsize=9, ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_png, dpi=DPI)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
