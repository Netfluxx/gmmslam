# GMM-SLAM

GMM-SLAM is a ROS 2 Humble lidar/depth SLAM stack that represents geometry with Gaussian mixture models, runs a fixed-lag smoother for high-rate poses, and maintains a global submap pose graph optimized with GTSAM iSAM2. Registration uses GMM Distribution-to-Distribution registration from the GIRA3D ecosystem. Loop candidate search uses SOLiD descriptors.

## Architecture

The runtime flow is:

1. Input depth point clouds, optional external odometry, optional ground-truth pose, and optional IMU.
2. Preprocess point clouds with range filtering, voxel downsampling, and target point limits from `config/params.yaml`.
3. Maintain local odometry and fixed-lag smoothing in `FixedLagBackend`.
4. Fit keyframe GMMs asynchronously and persist them under `gmm_dir`.
5. Dispatch D2D registration requests as `std_msgs::msg::String` JSON between `gmmslam_node` and `d2d_registration_node`.
6. Use SOLiD descriptor ranking and yaw priors for loop candidates when enabled.
7. Maintain and visualize the global submap pose graph, GMM map, latest scan, path, odometry, and registration markers.

## ROS 2 Humble Runtime

This package is ROS 2 Humble-only and uses `ament_cmake`.

Build from a ROS 2 workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select gmmslam
source install/setup.bash
```

Run the default launch file:

```bash
ros2 launch gmmslam gmmslam.launch.py
```

Useful launch arguments:

```bash
ros2 launch gmmslam gmmslam.launch.py rviz:=true
ros2 launch gmmslam gmmslam.launch.py ext_odom_publisher:=true
ros2 launch gmmslam gmmslam.launch.py config_file:=/path/to/params.yaml
```

The launch file starts:

- `gmmslam_node`
- `d2d_registration_node`
- optional `ext_odom_publisher_node`
- optional RViz
- optional static sensor transform publisher

## Docker

The supported container path is the Humble image in `docker/Dockerfile.humble`.

Build and start the container:

```bash
cd docker
docker compose up -d --build
```

Attach a shell:

```bash
docker exec -it gmmslam_humble bash
```

Inside the container, build and launch with the same ROS 2 commands:

```bash
cd /workspaces/gmmslam
source /opt/ros/humble/setup.bash
colcon build --packages-select gmmslam
source install/setup.bash
ros2 launch gmmslam gmmslam.launch.py
```

The helper script `docker/run_slam.sh` starts the Humble launch path in the container environment.

## Topics

Default subscribed topics:

- `/m500_1/mpa/depth/points` as `sensor_msgs::msg::PointCloud2`
- `/m500_1/mavros/local_position/pose` as `geometry_msgs::msg::PoseStamped`
- `/gmmslam_node/ext_odom_pose` as optional external odometry input
- optional IMU topic from `ros.imu_topic`

Default published topics:

- `/gmmslam_node/path`
- `/gmmslam_node/odom`
- `/gmmslam_node/odom_lpf`
- `/gmmslam_node/latest_frame_cloud`
- `/gmmslam_node/map_cloud`
- `/gmmslam_node/gmm_markers`
- `/gmmslam_node/gmm_global_markers`
- `/gmmslam_node/global_graph_markers`
- `/gmmslam_node/graph_nodes`
- `/gmmslam_node/loop_closure_markers`
- `/gmmslam_node/registration/request`
- `/gmmslam_node/registration/result`

The D2D request and result topics intentionally remain `std_msgs::msg::String` JSON for compatibility with the existing request flow.

## Parameters

`config/params.yaml` is the source of truth for normal runtime configuration. Launch arguments override ROS interface fields such as topic names and frame IDs after YAML loading.

Important sections:

- `ros`: input/output topics, frame names, and restart state path
- `preprocess`: point cloud filtering and downsampling
- `sogmm`: GMM fitting backend and component controls
- `registration`: D2D dispatch, scoring, queue, and factor settings
- `loop_closure`: loop search and acceptance gates
- `solid`: SOLiD descriptor and yaw prior settings
- `global_graph`: submap graph settings
- `map`: GMM pruning controls
- `visualization`: marker and cloud publication controls
- `ext_odom`: optional external odometry publisher and factor settings
- `imu`: optional IMU preintegration settings

By default, standalone external odometry factors are disabled in `params.yaml`. The optional external odometry publisher can be started with `ext_odom_publisher:=true`.

## Executables

- `gmmslam_node`: main SLAM frontend, smoother, global graph, and visualization.
- `d2d_registration_node`: worker node for D2D request/result JSON.
- `ext_odom_publisher_node`: optional noisy external odometry publisher from a reference pose topic.
- `solid_descriptor_benchmark`: SOLiD descriptor timing and similarity metrics tool.

## Third-Party Components

This package depends on:

- ROS 2 Humble middleware, messages, RViz, and TF2
- GTSAM
- Eigen
- yaml-cpp
- nlohmann/json
- dlib
- GIRA3D reconstruction and registration components
- SOLiD descriptor ideas and descriptor math
- OpenMP
- GMMap

The vendored `src/util/rtree.h` is used for spatial indexing in GMM pruning.
