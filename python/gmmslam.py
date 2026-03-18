#!/usr/bin/env python3
"""GMM-SLAM ROS1 node — entry point (backward-compatible with launch files).

All logic lives in gmmslam_node.py and its submodules.
"""

from gmmslam_node import GMMSLAMNode, main

if __name__ == "__main__":
    main()
