<img width="691" height="291" alt="slam_architecture" src="https://github.com/user-attachments/assets/602eab3c-24a1-4e21-971d-5ad0d61d7cbc" />


Installation:

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
