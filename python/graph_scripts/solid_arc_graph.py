#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def parse_solid_log(path):
    """
    Extract:
      t      from t=...
      x,y,z  from p=(x,y,z)
      d_yaw  from d_yaw=...deg
      solid  from lowercase solid=...

    Important:
      This ignores uppercase SOLiD=...ms, which is runtime, not descriptor score.
    """
    pattern = re.compile(
        r"t=(?P<t>-?\d+(?:\.\d+)?)"
        r".*?p=\((?P<x>-?\d+(?:\.\d+)?),(?P<y>-?\d+(?:\.\d+)?),(?P<z>-?\d+(?:\.\d+)?)\)"
        r".*?d_yaw=(?P<yaw>-?\d+(?:\.\d+)?)deg"
        r".*?\bsolid=(?P<solid>-?\d+(?:\.\d+)?)"
    )

    rows = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append(
                    [
                        float(m.group("t")),
                        float(m.group("x")),
                        float(m.group("y")),
                        float(m.group("z")),
                        float(m.group("yaw")),
                        float(m.group("solid")),
                    ]
                )

    if not rows:
        raise RuntimeError("No valid solid_bench lines found.")

    data = np.array(rows)
    return {
        "t": data[:, 0],
        "x": data[:, 1],
        "y": data[:, 2],
        "z": data[:, 3],
        "yaw": data[:, 4],
        "solid": data[:, 5],
    }


def deduplicate_points(x, y, tol=1e-4):
    """
    Keep only spatially distinct points for circle fitting.
    This avoids overweighting a long stationary period at the beginning.
    """
    keep = [0]

    for i in range(1, len(x)):
        dx = x[i] - x[keep[-1]]
        dy = y[i] - y[keep[-1]]

        if np.hypot(dx, dy) > tol:
            keep.append(i)

    return np.array(keep, dtype=int)


def fit_circle(x, y):
    """
    Least-squares circle fit:
      (x - cx)^2 + (y - cy)^2 = r^2

    Rewritten as:
      x^2 + y^2 = 2*cx*x + 2*cy*y + c
    """
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x**2 + y**2

    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    if rank < 3:
        raise RuntimeError("Circle fit failed: points may be almost collinear.")

    cx, cy, c = sol
    r = np.sqrt(c + cx**2 + cy**2)

    return cx, cy, r


def crop_forward_arc_leg(x, y, move_tol=1e-4):
    """
    Keep only the forward sweep from arc start to arc end.

    The Webots arc supervisor runs start -> end -> start; this drops the
    return leg so plots show a single monotonic arc.
    """
    dist = np.hypot(x - x[0], y - y[0])

    start_idx = 0
    for i in range(len(x)):
        if dist[i] > move_tol:
            start_idx = i
            break

    moving = dist[start_idx:] > move_tol
    if not np.any(moving):
        return slice(0, len(x))

    end_idx = start_idx + int(np.argmax(dist[start_idx:]))
    if end_idx <= start_idx:
        end_idx = len(x) - 1

    return slice(start_idx, end_idx + 1)


def compute_arc_angle_deg(x, y, cx, cy):
    """
    Signed arc angle on the fitted circle, zero at the sweep midpoint.

    Matches the Webots arc supervisor: progress 0 -> 1 maps to
    [-sweep/2, +sweep/2] around the angular center of the forward leg,
    not 0 at the first sample.
    """
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    theta_mid = 0.5 * (theta[0] + theta[-1])
    return np.rad2deg(theta - theta_mid)


def crop_by_arc_angle(arc_angle_deg, max_abs_deg):
    """Keep samples whose |arc angle| <= max_abs_deg."""
    mask = np.abs(arc_angle_deg) <= max_abs_deg
    if not np.any(mask):
        raise RuntimeError(
            f"No samples within +/-{max_abs_deg} deg arc angle after cropping."
        )
    return mask


def colored_line(ax, x, y, values, linewidth=3.0):
    """
    Draw a trajectory line colored by the SOLiD descriptor score.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    segment_values = 0.5 * (values[:-1] + values[1:])

    lc = LineCollection(segments, cmap="viridis", linewidth=linewidth)
    lc.set_array(segment_values)
    lc.set_clim(values.min(), values.max())

    ax.add_collection(lc)
    return lc


def add_direction_arrows(ax, x, y, n_arrows=8):
    """
    Add small arrows along the arc so the motion direction is obvious.
    """
    if len(x) < 3:
        return

    idxs = np.linspace(1, len(x) - 2, min(n_arrows, len(x) - 2), dtype=int)

    for i in idxs:
        dx = x[i + 1] - x[i - 1]
        dy = y[i + 1] - y[i - 1]

        norm = np.hypot(dx, dy)
        if norm < 1e-9:
            continue

        dx /= norm
        dy /= norm

        ax.annotate(
            "",
            xy=(x[i] + 0.05 * dx, y[i] + 0.05 * dy),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", lw=1.0),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="Path to the solid_bench log file")
    parser.add_argument(
        "--plane",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Plane used for the arc plot. Default: xy",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output image path (default: <logfile_stem>_arc.png next to the log)",
    )
    parser.add_argument(
        "--max-arc-angle-deg",
        type=float,
        default=70.0,
        help="Keep |arc angle| <= this value (deg). Default: 70",
    )
    args = parser.parse_args()

    if args.save is None:
        log_path = Path(args.logfile)
        args.save = str(log_path.with_name(log_path.stem + "_arc.png"))

    data = parse_solid_log(args.logfile)

    if args.plane == "xy":
        a = data["x"]
        b = data["y"]
        xlabel = "x relative to start [m]"
        ylabel = "y relative to start [m]"
    elif args.plane == "xz":
        a = data["x"]
        b = data["z"]
        xlabel = "x relative to start [m]"
        ylabel = "z relative to start [m]"
    else:
        a = data["y"]
        b = data["z"]
        xlabel = "y relative to start [m]"
        ylabel = "z relative to start [m]"

    solid = data["solid"]

    # Express position relative to initial position
    a_rel = a - a[0]
    b_rel = b - b[0]

    leg = crop_forward_arc_leg(a_rel, b_rel)
    a_rel = a_rel[leg]
    b_rel = b_rel[leg]
    solid = solid[leg]

    # Fit circle using spatially distinct points only
    fit_idx = deduplicate_points(a_rel, b_rel, tol=1e-4)

    if len(fit_idx) < 5:
        raise RuntimeError(
            "Not enough movement in the selected plane to show an arc. "
            "Try another plane with --plane xz or --plane yz, or check that p=(...) changes."
        )

    cx, cy, radius = fit_circle(a_rel[fit_idx], b_rel[fit_idx])
    arc_angle_deg = compute_arc_angle_deg(a_rel, b_rel, cx, cy)

    angle_mask = crop_by_arc_angle(arc_angle_deg, args.max_arc_angle_deg)
    a_rel = a_rel[angle_mask]
    b_rel = b_rel[angle_mask]
    solid = solid[angle_mask]
    arc_angle_deg = arc_angle_deg[angle_mask]

    mid_idx = int(np.argmin(np.abs(arc_angle_deg)))

    fig, (ax_path, ax_score) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # Left: drone path, colored by cosine similarity vs mid-arc descriptor
    lc = colored_line(ax_path, a_rel, b_rel, solid, linewidth=3.0)
    sc = ax_path.scatter(a_rel, b_rel, c=solid, cmap="viridis", s=18)

    # Draw fitted circle lightly so the arc geometry is obvious
    theta_circle = np.linspace(0, 2 * np.pi, 400)
    ax_path.plot(
        cx + radius * np.cos(theta_circle),
        cy + radius * np.sin(theta_circle),
        "--",
        alpha=0.25,
        linewidth=1.0,
        label="fitted circle",
    )

    add_direction_arrows(ax_path, a_rel, b_rel)

    ax_path.scatter(a_rel[0], b_rel[0], marker="o", s=80, label="arc start")
    ax_path.scatter(
        a_rel[mid_idx],
        b_rel[mid_idx],
        marker="*",
        s=140,
        c="red",
        edgecolors="k",
        label="mid-arc (ref)",
    )
    ax_path.scatter(a_rel[-1], b_rel[-1], marker="x", s=80, label="arc end")

    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel(xlabel)
    ax_path.set_ylabel(ylabel)
    ax_path.set_title("Arc trajectory (cosine vs mid-arc descriptor)")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend()

    cbar = fig.colorbar(sc, ax=ax_path)
    cbar.set_label("Cosine similarity vs mid-arc")

    # Right: true cosine(descriptor, descriptor_mid) when logged with reference_mode=mid_arc
    ax_score.plot(arc_angle_deg, solid, marker="o", markersize=3, linewidth=1.5)
    ax_score.scatter(
        [arc_angle_deg[mid_idx]],
        [solid[mid_idx]],
        marker="*",
        s=80,
        c="red",
        edgecolors="k",
        zorder=5,
        label="mid-arc (ref)",
    )
    lim = args.max_arc_angle_deg
    ax_score.set_xlim(-lim, lim)
    ax_score.axhline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    ax_score.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    ax_score.set_xlabel("Arc angle relative to sweep center [deg]")
    ax_score.set_ylabel("Cosine similarity vs mid-arc descriptor")
    ax_score.set_title("Viewpoint sensitivity along the arc")
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(loc="lower left", fontsize=8)
    ax_score.set_ylim(0.0, 1.05)

    fig.suptitle("SOLiD cosine similarity vs arc angle (mid-arc reference)")
    fig.tight_layout()

    plt.savefig(args.save, dpi=300)
    print(f"Saved plot to {args.save}")
    plt.close(fig)


if __name__ == "__main__":
    main()
