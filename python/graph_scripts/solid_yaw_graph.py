#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_solid_log(path):
    """
    Extracts:
      - t: timestamp from t=...
      - yaw: yaw difference from d_yaw=...deg
      - solid: descriptor score from solid=...

    Ignores SOLiD=...ms runtime.
    """
    pattern = re.compile(
        r"t=(?P<t>-?\d+(?:\.\d+)?)"
        r".*?d_yaw=(?P<yaw>-?\d+(?:\.\d+)?)deg"
        r".*?\bsolid=(?P<solid>-?\d+(?:\.\d+)?)"
    )

    t_vals = []
    yaw_vals = []
    solid_vals = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                t_vals.append(float(match.group("t")))
                yaw_vals.append(float(match.group("yaw")))
                solid_vals.append(float(match.group("solid")))

    return np.array(t_vals), np.array(yaw_vals), np.array(solid_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="Path to the solid_bench log file")
    parser.add_argument(
        "--x",
        choices=["yaw", "time"],
        default="yaw",
        help="X axis: yaw or time. Default: yaw",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output image path (default: <logfile_stem>_yaw.png next to the log)",
    )
    args = parser.parse_args()

    if args.save is None:
        log_path = Path(args.logfile)
        args.save = str(log_path.with_name(log_path.stem + "_yaw.png"))

    t, yaw, solid = parse_solid_log(args.logfile)

    if len(solid) == 0:
        raise RuntimeError("No solid_bench lines found. Check the log format.")

    x = yaw if args.x == "yaw" else t - t[0]
    xlabel = "Yaw change [deg]" if args.x == "yaw" else "Time [s]"

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, solid, marker="o", markersize=3, linewidth=1.5)

    plt.xlabel(xlabel)
    plt.ylabel("SOLiD descriptor score")
    plt.title("SOLiD descriptor score during in-place yaw rotation")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()

    plt.savefig(args.save, dpi=300)
    print(f"Saved plot to {args.save}")
    plt.close()


if __name__ == "__main__":
    main()
