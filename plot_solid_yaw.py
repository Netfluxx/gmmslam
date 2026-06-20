#!/usr/bin/env python3
"""
Plot SOLiD descriptor similarity vs. yaw rotation from a benchmark log.

Usage:
    python plot_solid_yaw.py                      # uses INPUT_LOG / OUTPUT_PNG below
    python plot_solid_yaw.py my.log out.png       # override paths on the command line

The script reads lines of the form
    ... t=12.3 p=(...) d_yaw=45.6deg solid=0.97 range=0.98 angle=0.99 polar=0.65 ...
and plots each similarity channel against d_yaw.
"""

import re
import sys
import numpy as np
import matplotlib

# Use a headless renderer. Some desktop backends (Qt/GTK via system GL/fonts)
# can segfault after parsing succeeds, especially when launched from terminals
# attached to Docker/X11 sessions.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# =====================================================================
#  CONFIG  --  edit freely
# =====================================================================
INPUT_LOG  = "solid_descriptor_benchmark.log"  # path to the log file
OUTPUT_PNG = "solid_descriptor_benchmark.png"  # where to save the figure

TITLE   = "SOLiD descriptor robustness to yaw rotation (60 deg FOV)"
XLABEL  = "Yaw rotation relative to reference [deg]"
YLABEL  = "Cosine similarity to reference descriptor"

# Which channels to plot, with their legend label and colour.
# Comment out a line to drop that channel from the plot.
CHANNELS = [
    # key,      legend label,                              colour
    ("solid", "Full SOLiD",                              "#8e44ad"),
    ("range", "R-SOLiD (range, rotation-invariant)",     "#2980b9"),
    ("angle", "A-SOLiD  (azimuth)",                       "#27ae60"),
    ("polar", "Polar  (rotation-sensitive)",              "#c0392b"),
]

SHOW_SCATTER   = True     # faint raw points behind the trend lines
SHOW_TRENDLINE = True     # binned-mean trend line
N_BINS         = 40       # number of yaw bins for the trend line
YLIM           = (0.30, 1.01)   # set to None for auto
FIGSIZE        = (9.0, 5.4)
DPI            = 200
# =====================================================================


def parse_log(path):
    """Return arrays for t, d_yaw and each similarity channel."""
    pat = re.compile(
        r't=([-\d.]+).*?d_yaw=([-\d.]+)deg\s+'
        r'solid=([\d.]+)\s+range=([\d.]+)\s+angle=([\d.]+)\s+polar=([\d.]+)'
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
    data = {
        "t":     arr[:, 0],
        "d_yaw": arr[:, 1],
        "solid": arr[:, 2],
        "range": arr[:, 3],
        "angle": arr[:, 4],
        "polar": arr[:, 5],
    }
    print(f"parsed {len(arr)} rows; "
          f"yaw {data['d_yaw'].min():.1f} to {data['d_yaw'].max():.1f} deg")
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


def main():
    in_log  = sys.argv[1] if len(sys.argv) > 1 else INPUT_LOG
    out_png = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PNG

    data = parse_log(in_log)
    yaw = data["d_yaw"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for key, label, colour in CHANNELS:
        y = data[key]
        if SHOW_SCATTER:
            ax.scatter(yaw, y, s=12, alpha=0.20, color=colour, zorder=2)
        if SHOW_TRENDLINE:
            bx, by = binned_mean(yaw, y, N_BINS)
            ax.plot(bx, by, "-o", color=colour, lw=2, ms=4,
                    label=label, zorder=4)
        else:
            ax.scatter([], [], color=colour, label=label)  # legend proxy

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(TITLE)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_xlim(yaw.min() - 5, yaw.max() + 5)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
