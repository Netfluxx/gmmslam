#pragma once
#include <cstdint>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

#include "gmmslam/types.hpp"

namespace gmmslam {

double stampToSec(const ros::Time& stamp);

geometry_msgs::TransformStamped poseToTransformStamped(
    const Matrix4d& T,
    const ros::Time& stamp,
    const std::string& parent_frame,
    const std::string& child_frame);

geometry_msgs::PoseStamped poseToPoseStamped(
    const Matrix4d& T,
    const ros::Time& stamp,
    const std::string& frame_id);

Eigen::MatrixXf pc2ToEigen(const sensor_msgs::PointCloud2& msg);

sensor_msgs::PointCloud2 eigenToPc2Rgb(
    const Eigen::MatrixXf& pts,
    const ros::Time& stamp,
    const std::string& frame_id,
    uint8_t r,
    uint8_t g,
    uint8_t b);

Eigen::MatrixXf preprocess(
    const Eigen::MatrixXf& pts,
    double min_range,
    double max_range,
    double voxel_size);

Matrix4d poseMsgToMatrix(const geometry_msgs::Pose& pose_msg);

Eigen::MatrixXd makePcld4d(const Eigen::MatrixXf& pts);

} // namespace gmmslam
