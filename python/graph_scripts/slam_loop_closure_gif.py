#!/usr/bin/env python3
"""Generate an animated GIF showing a SLAM loop closure demo from a ROS bag.

Usage:
    python3 slam_loop_closure_gif.py slam_demo.bag
    python3 slam_loop_closure_gif.py slam_demo.bag -o output.gif --fps 15 --step 3
    python3 slam_loop_closure_gif.py slam_demo.bag --map      # include map cloud

Topics read:
    /gmmslam_node/global_graph_path    nav_msgs/Path       – global optimised trajectory
    /gmmslam_node/gt_path              nav_msgs/Path       – ground truth
    /gmmslam_node/loop_closure_markers visualization_msgs/MarkerArray
    /gmmslam_node/map_cloud            sensor_msgs/PointCloud2  (opt, --map flag)

Dependencies: rosbag, matplotlib, Pillow, numpy
    pip install matplotlib Pillow numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

try:
    import rosbag
except ImportError:
    print("ERROR: rosbag not found — source your ROS workspace first.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def path_to_xy(msg) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array([p.pose.position.x for p in msg.poses])
    ys = np.array([p.pose.position.y for p in msg.poses])
    return xs, ys


def parse_map_cloud(msg) -> Optional[np.ndarray]:
    """Return Nx2 array of (x, y) from a PointCloud2 message, or None."""
    try:
        from sensor_msgs import point_cloud2
        pts = list(point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True))
        if not pts:
            return None
        arr = np.array(pts, dtype=np.float32)
        return arr
    except Exception:
        return None


def interpolate_map(map_msgs, t_sec: float) -> Optional[np.ndarray]:
    """Return the most recent map cloud at or before t_sec."""
    result = None
    for t, pts in map_msgs:
        if t <= t_sec:
            result = pts
        else:
            break
    return result


def loop_edges_at(loop_msgs, loop_times, t_sec) -> List[Tuple]:
    """Return list of (x0, y0, x1, y1) loop closure edges active at t_sec."""
    if len(loop_times) == 0:
        return []
    idx = int(np.searchsorted(loop_times, t_sec, side="right")) - 1
    if idx < 0:
        return []
    _, ma = loop_msgs[idx]
    edges = []
    for marker in ma.markers:
        if marker.action == 2:  # DELETE
            continue
        pts = marker.points
        for i in range(0, len(pts) - 1, 2):
            edges.append((pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y))
    return edges


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SLAM loop closure GIF generator")
    parser.add_argument("bag", help="Path to the .bag file")
    parser.add_argument("-o", "--output", default="slam_loop_closure.gif",
                        help="Output GIF path (default: slam_loop_closure.gif)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate (default: 10)")
    parser.add_argument("--step", type=int, default=3,
                        help="Use every N-th path message (default: 3)")
    parser.add_argument("--dpi", type=int, default=130, help="Output DPI (default: 130)")
    parser.add_argument("--map", action="store_true",
                        help="Include map cloud points (slower, larger GIF)")
    parser.add_argument("--map-interval", type=float, default=30.0,
                        help="Keep one map cloud sample every N seconds (default: 30)")
    parser.add_argument("--map-max-pts", type=int, default=50_000,
                        help="Downsample each map cloud to at most this many points (default: 50000)")
    parser.add_argument("--topic-est", default="/gmmslam_node/global_graph_path")
    parser.add_argument("--topic-gt",  default="/gmmslam_node/gt_path")
    parser.add_argument("--topic-loop", default="/gmmslam_node/loop_closure_markers")
    parser.add_argument("--topic-map",  default="/gmmslam_node/map_cloud")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: bag not found: {bag_path}")
        sys.exit(1)

    topics = [args.topic_est, args.topic_gt, args.topic_loop]
    if args.map:
        topics.append(args.topic_map)

    print(f"Reading: {bag_path}")
    est_msgs, gt_msgs, loop_msgs, map_msgs = [], [], [], []
    est_counter = 0
    map_last_t = -1e9

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()
            if topic == args.topic_est:
                # Apply --step during reading to avoid storing all O(N^2) poses
                if est_counter % args.step == 0:
                    est_msgs.append((ts, msg))
                est_counter += 1
            elif topic == args.topic_gt:
                # Only keep the last GT message — it contains the full path
                gt_msgs = [(ts, msg)]
            elif topic == args.topic_loop:
                loop_msgs.append((ts, msg))
            elif topic == args.topic_map and args.map:
                # Sparse sampling: one map cloud every --map-interval seconds
                if ts - map_last_t >= args.map_interval:
                    pts = parse_map_cloud(msg)
                    if pts is not None:
                        # Downsample to cap RAM usage
                        if len(pts) > args.map_max_pts:
                            idx = np.random.choice(len(pts), args.map_max_pts, replace=False)
                            pts = pts[idx]
                        map_msgs.append((ts, pts))
                        map_last_t = ts

    print(f"  estimated path : {len(est_msgs)} frames (every {args.step} msgs)")
    print(f"  ground truth   : {len(gt_msgs)} msgs")
    print(f"  loop closures  : {len(loop_msgs)} msgs")
    if args.map:
        print(f"  map cloud      : {len(map_msgs)} samples (1 per {args.map_interval:.0f}s, max {args.map_max_pts} pts each)")

    if not est_msgs:
        print(f"ERROR: no messages on {args.topic_est}")
        sys.exit(1)

    loop_times = np.array([t for t, _ in loop_msgs]) if loop_msgs else np.array([])

    # Ground truth: use the last (most complete) message
    gt_x, gt_y = (path_to_xy(gt_msgs[-1][1]) if gt_msgs else (None, None))

    # Compute axis bounds from all trajectories
    all_x, all_y = [], []
    for _, msg in est_msgs:
        xs, ys = path_to_xy(msg)
        all_x.extend(xs.tolist()); all_y.extend(ys.tolist())
    if gt_x is not None:
        all_x.extend(gt_x.tolist()); all_y.extend(gt_y.tolist())
    if not all_x:
        print("ERROR: no pose data found.")
        sys.exit(1)

    pad = 1.5
    xmin, xmax = min(all_x) - pad, max(all_x) + pad
    ymin, ymax = min(all_y) - pad, max(all_y) + pad

    # ── Figure setup ──────────────────────────────────────────────────────────
    BG = "#0d0d0d"
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlabel("x [m]", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y [m]", color="#aaaaaa", fontsize=9)
    ax.set_title("GMM-SLAM — Loop Closure Demo", color="white", fontsize=12, pad=8)

    # Static ground truth
    if gt_x is not None:
        ax.plot(gt_x, gt_y, color="#ff5555", linewidth=1.2, alpha=0.5,
                label="Ground truth", zorder=2)

    # Dynamic elements
    map_scatter = ax.scatter([], [], s=0.5, c="#555555", alpha=0.4, zorder=1,
                             rasterized=True) if args.map else None
    est_line, = ax.plot([], [], color="#4daaff", linewidth=1.5, zorder=3)
    drone_dot, = ax.plot([], [], "o", color="white", markersize=5, zorder=5)
    loop_line_cache: list = []

    time_text = ax.text(0.98, 0.02, "", transform=ax.transAxes,
                        color="#888888", fontsize=8, ha="right", va="bottom",
                        fontfamily="monospace")
    loop_count_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                               color="#cc66ff", fontsize=8, ha="left", va="bottom")

    legend_handles = [
        Line2D([0], [0], color="#ff5555", linewidth=1.5, label="Ground truth"),
        Line2D([0], [0], color="#4daaff", linewidth=1.5, label="SLAM estimate"),
        Line2D([0], [0], color="#cc66ff", linewidth=1.2, label="Loop closures"),
    ]
    if args.map:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markersize=5, label="Map cloud", linestyle="None"))
    ax.legend(handles=legend_handles, facecolor="#1a1a1a", edgecolor="#333333",
              labelcolor="white", fontsize=8, loc="upper left")

    start_t = est_msgs[0][0]

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame_idx: int):
        t_sec, msg = est_msgs[frame_idx]
        xs, ys = path_to_xy(msg)

        est_line.set_data(xs, ys)
        if len(xs):
            drone_dot.set_data([xs[-1]], [ys[-1]])

        # Loop closure edges
        for ln in loop_line_cache:
            ln.remove()
        loop_line_cache.clear()
        edges = loop_edges_at(loop_msgs, loop_times, t_sec)
        for x0, y0, x1, y1 in edges:
            ln, = ax.plot([x0, x1], [y0, y1], color="#cc66ff",
                          linewidth=0.8, alpha=0.65, zorder=4)
            loop_line_cache.append(ln)

        # Map cloud
        if map_scatter is not None and map_msgs:
            pts = interpolate_map(map_msgs, t_sec)
            if pts is not None:
                map_scatter.set_offsets(pts)

        time_text.set_text(f"t = {t_sec - start_t:6.1f} s")
        n_loops = len(edges) // 1  # each edge = one loop pair line
        loop_count_text.set_text(f"loop edges: {len(edges)}" if edges else "")

        return [est_line, drone_dot, time_text, loop_count_text] + loop_line_cache

    # ── Render ────────────────────────────────────────────────────────────────
    n_frames = len(est_msgs)
    if n_frames > 1:
        real_dur = est_msgs[-1][0] - est_msgs[0][0]
        gif_dur  = (n_frames - 1) / args.fps
        print(f"  playback speed : {real_dur / gif_dur:.1f}× real time  "
              f"({real_dur:.0f} s → {gif_dur:.1f} s GIF)")
    print(f"Rendering {n_frames} frames at {args.fps} fps → {args.output}")

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // args.fps, blit=False)

    ani.save(args.output, writer="pillow", fps=args.fps, dpi=args.dpi)
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
