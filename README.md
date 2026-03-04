
Installation: ![slam_architecture](https://github.com/user-attachments/assets/a036ef1a-17e7-49a4-bdb6-ec223bd4a179)



step 1: go to the gmmslam/docker folder then build the docker container with:
`docker compose up -d --build`
then go inside the container with :
`docker exec -it disal_slam bash`

To stop the container run : `docker stop disal_slam`

## Running GMM-SLAM

### Start the SLAM Node

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
- **MapCloud**: Accumulated SLAM map
- **Path**: Trajectory estimate
- **Odometry**: Current pose
- **DepthImage**: Depth camera visualization
- **TF**: Transform tree (world → m500_1_base_link)

### Topics

**Subscribed:**
- `/m500_1/mpa/depth/points` - Depth point cloud

**Published:**
- `/gmmslam_node/path` - Trajectory
- `/gmmslam_node/odom` - Odometry
- `/gmmslam_node/map_cloud` - Accumulated map

## Parameters

Edit the SOGMM bandwidth in `python/gmmslam.py` (line 375):
```python
self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.2)
```

Larger bandwidth (0.3-0.5) → more stable D2D registration
Smaller bandwidth (0.1) → more precise GMM fitting


ROS1 Noetic Wrapper: 

ROS2 Humble Wrapper:


State of the art Lidar-Inertial Odometry and SLAM packages used for inspiration:
- https://github.com/koide3/glim
- https://github.com/superxslam/SuperOdom
- https://github.com/TixiaoShan/LIO-SAM
