# GMM-SLAM

GMM-SLAM is a lidar / depth–driven SLAM stack that represents geometry with **Gaussian mixture models (GMMs)**, runs a **fixed-lag smoother** for high-rate poses, and maintains a **global submap pose graph** optimized with **GTSAM iSAM2**. Loop closure and submap alignment use **distribution-to-distribution (D2D)** GMM registration from the GIRA3D ecosystem.

---

## SLAM architecture

High-level data flow:

1. **Input** — Depth point clouds (ROS), optional ground truth pose, optional IMU.
2. **Preprocessing** — Range filtering, voxel downsampling.
3. **Local odometry / smoothing** — A fixed-lag factor graph (`FixedLagBackend`) integrates relative motion (e.g. noisy GT–based odometry) at a high rate and keeps a sliding window of poses and keyframes.
4. **Keyframe GMM fitting** — On selected keyframes, a **self-organizing GMM (SOGMM)** is fit asynchronously (GPU path where available) and stored per keyframe for registration and map building.
5. **Registration** — Sequential and loop-closure requests use **SOLiD**-style descriptors for candidates where configured, and **D2D GMM registration** to estimate relative transforms and scores for factors.
6. **Global pose graph** — Submaps are anchored on the smoother; when a submap closes, its keyframe GMMs are merged in the submap frame, optionally **pruned** (R-tree candidate search, Bhattacharyya gate, then keeping the Gaussian with lower associated pose uncertainty), saved to disk, and used for overlap-based submap registration. **iSAM2** refines submap poses when new between / loop factors arrive.
7. **Visualization** — RViz markers for trajectory, latest GMM, **global per-submap GMM map**, submap graph, and loop-closure debug markers.

<img width="2445" height="1418" alt="image(1)" src="https://github.com/user-attachments/assets/00e01e80-a79f-4379-97a0-99e9175d9f6b" />


<img width="1817" height="1289" alt="Screenshot from 2026-05-12 16-45-49" src="https://github.com/user-attachments/assets/1f8a086c-b764-4b56-8edc-c76e757a6f80" />


<img width="1652" height="1047" alt="gmmslam" src="https://github.com/user-attachments/assets/771e7bf9-8e91-4372-87c7-3dbf097edb54" />

---

## Docker image

### What the image is for

The Dockerfile targets **Linux x86_64** with an **NVIDIA GPU** and the **NVIDIA Container Toolkit** (GPU passthrough). The image is based on:

- **`nvidia/cuda:11.8.0-devel-ubuntu20.04`** — CUDA 11.8 development image on **Ubuntu 20.04**
- **ROS 1 Noetic**
- **GTSAM 4.2**, **Eigen 3.4**, **GIRA3D** reconstruction & registration (Open3D, SOGMM, D2D), **Webots** + `webots_ros` for simulation-style workflows

It is **not** intended for ARM-only or CPU-only production (the default build enables CUDA paths in the reconstruction stack). CPU-only or other arches would require Dockerfile changes.

### Image size

After a full build, the Docker image is on the order of **~24.2 GB** (layers include CUDA, ROS, Webots, colcon-built GIRA3D + Open3D, and tooling). Plan disk space accordingly.

### Where it was tested

The following **host** setup was used for builds and runs (container OS remains Ubuntu 20.04 inside the image):

| Item | Value |
|------|--------|
| Host OS | **Ubuntu 22.04.5** Desktop |
| CPU | **Intel Core Ultra 9 275HX** |
| GPU | **NVIDIA GeForce RTX 5070 Ti** (12 GB) |
| Driver / CUDA (host) | **CUDA 13.2** (driver stack on host; container ships CUDA **11.8** from the base image) |
| RAM | **32 GB** |

Other Linux hosts with a supported NVIDIA driver and Docker + Compose should work, but only the configuration above is explicitly documented as tested.

### How to build and run

From the **`gmmslam/docker`** directory (compose **build context** is the `gmmslam` package root `..`):

```bash
cd docker
docker compose up -d --build
```

Attach a shell:

```bash
docker exec -it disal_slam bash
```

Stop the container:

```bash
docker stop disal_slam
```

Rebuild without cache (heavy):

```bash
docker compose build --no-cache
```

The compose file (`docker/compose.yaml`) mounts the repo into the workspace and passes `DISPLAY` and GPU devices for RViz and CUDA workloads.

---

## Running GMM-SLAM

### Start the SLAM node

```bash
docker exec -it disal_slam run_gmmslam
```

### Visualize in RViz

In a separate terminal:

```bash
docker exec -it disal_slam run_rviz
```

RViz will show:

- **DepthCloud**: Live depth point cloud from drone
- **GmmMap**: Per-submap GMM map (3D ellipsoids, color-coded per submap, re-anchored on every loop closure)
- **GmmLatest** *(off by default)*: GMM fit of the latest keyframe in white
- **SubmapGraph**: Submap nodes (green) and loop-closure edges (magenta)
- **KeyframeNodes** *(off by default)*: Per-keyframe pose markers (red)
- **LoopClosures**: Detected loop-closure correspondences
- **Path**: Trajectory estimate
- **Odometry**: Current pose
- **TF**: Transform tree (world → m500_1_base_link)

### Topics

**Subscribed:**

- `/m500_1/mpa/depth/points` — Depth point cloud

**Published (visualization):**

- `/gmmslam_node/path` — Trajectory
- `/gmmslam_node/odom` — Odometry
- `/gmmslam_node/gmm_global_markers` — **The map**: per-submap GMMs as 3D ellipsoids, globally aligned via iSAM2, refreshed every loop closure
- `/gmmslam_node/gmm_markers` — Latest-keyframe GMM
- `/gmmslam_node/global_graph_markers` — Submap pose graph + loop edges
- `/gmmslam_node/graph_nodes` — Per-keyframe pose markers
- `/gmmslam_node/loop_closure_markers` — Loop-closure correspondences
- `/gmmslam_node/latest_frame_cloud` — Most recent depth scan in world frame
- `/gmmslam_node/map_cloud` — Accumulated depth scans (one chunk per smoother frame), reprojected with current `pose_by_idx` so fixed-lag and **keyframe-level** loop updates move the cloud; latched; rate set by `map_cloud_publish_hz` (default 0.5 Hz). Pure **submap–submap** graph edges alone do not move these points (they follow the smoother, not the global submap graph).

The map is stored as one merged GMM per submap on disk (`gmm_dir`/`submap_XXXX.gmm`). Near-duplicate components produced by overlapping observations are pruned at submap finalization using optional R-tree spatial indexing, a Bhattacharyya-distance gate, and a keep-lower-pose-uncertainty rule; see the `map:` block in `config/params.yaml` to tune.

---

## Parameters

Edit the SOGMM bandwidth in `python/gmmslam.py` (line 375):

```python
self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.2)
```

- Larger bandwidth (0.3–0.5) → more stable D2D registration  
- Smaller bandwidth (0.1) → more precise GMM fitting  

---

## Third-party and open-source components

This repository **vendors or depends on** the following (versions may follow the Docker image or your host install):

| Component | Role | Upstream / license |
|-----------|------|--------------------|
| **ROS Noetic** | Middleware, messages, RViz, `tf`, Catkin | [Open Robotics](https://www.openrobotics.org/) |
| **GTSAM** | Smoothing and **iSAM2** global submap graph | [borglab/gtsam](https://github.com/borglab/gtsam) |
| **Eigen** | Linear algebra | [libeigen/eigen](https://eigen.tuxfamily.org/) |
| **yaml-cpp** | YAML configuration loading | [jbeder/yaml-cpp](https://github.com/jbeder/yaml-cpp) |
| **nlohmann/json** | JSON in C++ (FetchContent if not found on system) | [nlohmann/json](https://github.com/nlohmann/json) |
| **dlib** | Pulled in via GIRA3D / registration stack | [davisking/dlib](https://github.com/davisking/dlib) |
| **GIRA3D reconstruction** | SOGMM / Open3D colcon workspace (`self_organizing_gmm`, etc.) | [gira3d/gira3d-reconstruction](https://github.com/gira3d/gira3d-reconstruction) |
| **GIRA3D registration** | GMM I/O, **D2D registration** (`gmm`, `gmm_d2d_registration`) | [gira3d/gira3d-registration](https://github.com/gira3d/gira3d-registration) |
| **SOLiD** | **Place recognition for loop closure** — spatially organized LiDAR global descriptor (cosine gate on radius candidates, optional yaw prior for D2D init); see Kim et al., *IEEE RA-L* 2024 | [sparolab/SOLiD](https://github.com/sparolab/SOLiD/tree/main) (official implementation; BSD-3-Clause) |
| **Open3D** | Built as part of the GIRA3D reconstruction image flow | [isl-org/Open3D](https://github.com/isl-org/Open3D) |
| **nanoflann** | KD-tree headers in the GIRA3D dry install | [jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) |
| **NVIDIA CUDA** (base image) | GPU builds for reconstruction / SOGMM path | NVIDIA EULA |
| **CMake** (Kitware tarball in Docker) | Build system for newer Open3D | [Kitware/CMake](https://github.com/Kitware/CMake) |
| **Webots** + **webots_ros** | Simulator integration in the Docker image | [cyberbotics/webots](https://github.com/cyberbotics/webots) |
| **R-tree** (`src/util/rtree.h`) | Spatial indexing for GMM pruning candidates | Vendored from **[nushoin/RTree](https://github.com/nushoin/RTree)** (header-only `RTree.h` lineage: Guttman & Stonebraker’s algorithm, C++ templated port by Greg Douglas, community forks; see that repo’s README and [LICENSE](https://github.com/nushoin/RTree/blob/master/LICENSE) for authorship and licensing) |


### Related SLAM / odometry packages which inspired this project:

Design ideas were informed by public systems such as:

- [koide3/glim](https://github.com/koide3/glim)  
- [superxslam/SuperOdom](https://github.com/superxslam/SuperOdom)  
- [TixiaoShan/LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)  


---

## ROS wrappers

- **ROS 1 Noetic** — Primary path (Docker + Catkin).  
- **ROS 2 Humble** — TODO.
