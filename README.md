
Installation: ![slam_architecture](https://github.com/user-attachments/assets/a036ef1a-17e7-49a4-bdb6-ec223bd4a179)



step 1: go to the gmmslam/docker folder then build the docker container with:
`docker compose up -d --build`
then go inside the container with :
`docker exec -it disal_slam bash`

To stop the container run : `docker stop disal_slam`


ROS1 Noetic Wrapper: 

ROS2 Humble Wrapper:


State of the art Lidar-Inertial Odometry and SLAM packages used for inspiration:
- https://github.com/koide3/glim
- https://github.com/superxslam/SuperOdom
- https://github.com/TixiaoShan/LIO-SAM
