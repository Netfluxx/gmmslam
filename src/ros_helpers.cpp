#include "gmmslam/ros_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <sensor_msgs/point_cloud2_iterator.h>

namespace gmmslam {

namespace {

// Centroid per voxel; leaf_size <= 0 returns pts unchanged (same idea as SOLiD).
Eigen::MatrixXf voxelDownsample(const Eigen::MatrixXf& pts, double leaf_size) {
    if (leaf_size <= 0.0 || pts.rows() == 0) {
        return pts;
    }

    const float inv = 1.0f / static_cast<float>(leaf_size);
    struct Acc {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        int n = 0;
    };
    std::unordered_map<std::int64_t, Acc> bins;
    bins.reserve(static_cast<std::size_t>(pts.rows()));

    auto pack = [](std::int32_t a, std::int32_t b, std::int32_t c) {
        std::int64_t h = static_cast<std::int64_t>(a) * 73856093LL;
        h ^= static_cast<std::int64_t>(b) * 19349663LL;
        h ^= static_cast<std::int64_t>(c) * 83492791LL;
        return h;
    };

    for (int i = 0; i < pts.rows(); ++i) {
        const auto ix = static_cast<std::int32_t>(std::floor(pts(i, 0) * inv));
        const auto iy = static_cast<std::int32_t>(std::floor(pts(i, 1) * inv));
        const auto iz = static_cast<std::int32_t>(std::floor(pts(i, 2) * inv));
        const auto key = pack(ix, iy, iz);
        auto& acc = bins[key];
        acc.sx += pts(i, 0);
        acc.sy += pts(i, 1);
        acc.sz += pts(i, 2);
        acc.n += 1;
    }

    Eigen::MatrixXf out(static_cast<int>(bins.size()), 3);
    int r = 0;
    for (const auto& kv : bins) {
        const auto& a = kv.second;
        out(r, 0) = static_cast<float>(a.sx / static_cast<double>(a.n));
        out(r, 1) = static_cast<float>(a.sy / static_cast<double>(a.n));
        out(r, 2) = static_cast<float>(a.sz / static_cast<double>(a.n));
        ++r;
    }
    return out;
}

Eigen::MatrixXf subsampleToMax(const Eigen::MatrixXf& pts, int target) {
    const int n = static_cast<int>(pts.rows());
    if (target <= 0 || n <= target) {
        return pts;
    }
    std::vector<int> pop(static_cast<std::size_t>(n));
    std::iota(pop.begin(), pop.end(), 0);
    std::vector<int> pick(static_cast<std::size_t>(target));
    thread_local std::mt19937 rng{std::random_device{}()};
    std::sample(pop.begin(), pop.end(), pick.begin(),
                static_cast<std::ptrdiff_t>(target), rng);
    Eigen::MatrixXf out(target, 3);
    for (int i = 0; i < target; ++i) {
        out.row(i) = pts.row(pick[static_cast<std::size_t>(i)]);
    }
    return out;
}

} // namespace

double stampToSec(const ros::Time& stamp) {
    return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nsec);
}

geometry_msgs::TransformStamped poseToTransformStamped(
    const Matrix4d& T,
    const ros::Time& stamp,
    const std::string& parent_frame,
    const std::string& child_frame) {

    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();

    geometry_msgs::TransformStamped ts;
    ts.header.stamp = stamp;
    ts.header.frame_id = parent_frame;
    ts.child_frame_id = child_frame;
    ts.transform.translation.x = T(0, 3);
    ts.transform.translation.y = T(1, 3);
    ts.transform.translation.z = T(2, 3);
    ts.transform.rotation.x = q.x();
    ts.transform.rotation.y = q.y();
    ts.transform.rotation.z = q.z();
    ts.transform.rotation.w = q.w();
    return ts;
}

geometry_msgs::PoseStamped poseToPoseStamped(
    const Matrix4d& T,
    const ros::Time& stamp,
    const std::string& frame_id) {

    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();

    geometry_msgs::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = frame_id;
    ps.pose.position.x = T(0, 3);
    ps.pose.position.y = T(1, 3);
    ps.pose.position.z = T(2, 3);
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    return ps;
}

Eigen::MatrixXf pc2ToEigen(const sensor_msgs::PointCloud2& msg) {
    const int n_points = static_cast<int>(msg.height * msg.width);
    if (n_points == 0) {
        return Eigen::MatrixXf(0, 3);
    }

    sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");

    Eigen::MatrixXf result(n_points, 3);
    int valid_count = 0;

    for (int i = 0; i < n_points; ++i, ++it_x, ++it_y, ++it_z) {
        const float x = *it_x;
        const float y = *it_y;
        const float z = *it_z;
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            result(valid_count, 0) = x;
            result(valid_count, 1) = y;
            result(valid_count, 2) = z;
            ++valid_count;
        }
    }

    result.conservativeResize(valid_count, 3);
    return result;
}

sensor_msgs::PointCloud2 eigenToPc2Rgb(
    const Eigen::MatrixXf& pts,
    const ros::Time& stamp,
    const std::string& frame_id,
    uint8_t r,
    uint8_t g,
    uint8_t b) {

    const auto n = static_cast<uint32_t>(pts.rows());

    // Pack RGB into a float via uint32 reinterpretation
    const uint32_t rgb_uint = (static_cast<uint32_t>(r) << 16) |
                              (static_cast<uint32_t>(g) << 8) |
                              static_cast<uint32_t>(b);
    float rgb_float;
    std::memcpy(&rgb_float, &rgb_uint, sizeof(float));

    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = frame_id;
    cloud.height = 1;
    cloud.width = n;
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4,
        "x", 1, sensor_msgs::PointField::FLOAT32,
        "y", 1, sensor_msgs::PointField::FLOAT32,
        "z", 1, sensor_msgs::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(n);

    sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> out_rgb(cloud, "rgb");

    for (uint32_t i = 0; i < n; ++i, ++out_x, ++out_y, ++out_z, ++out_rgb) {
        *out_x = pts(i, 0);
        *out_y = pts(i, 1);
        *out_z = pts(i, 2);
        *out_rgb = rgb_float;
    }

    return cloud;
}

Eigen::MatrixXf preprocess(
    const Eigen::MatrixXf& pts,
    double min_range,
    double max_range,
    double voxel_size,
    int target_points) {

    if (pts.rows() == 0) return pts;

    const Eigen::VectorXf ranges = pts.rowwise().norm();
    const auto min_r = static_cast<float>(min_range);
    const auto max_r = static_cast<float>(max_range);

    std::vector<int> keep;
    keep.reserve(pts.rows());
    for (int i = 0; i < pts.rows(); ++i) {
        if (ranges(i) >= min_r && ranges(i) <= max_r) {
            keep.push_back(i);
        }
    }

    Eigen::MatrixXf result(static_cast<int>(keep.size()), 3);
    for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
        result.row(i) = pts.row(keep[i]);
    }
    result = voxelDownsample(result, voxel_size);
    return subsampleToMax(result, target_points);
}

Matrix4d poseMsgToMatrix(const geometry_msgs::Pose& pose_msg) {
    Eigen::Quaterniond q(
        pose_msg.orientation.w,
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z);

    const double q_norm = q.norm();
    if (q_norm < 1e-12) {
        throw std::runtime_error("Invalid quaternion norm in ground-truth pose");
    }
    q.normalize();

    Matrix4d T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = pose_msg.position.x;
    T(1, 3) = pose_msg.position.y;
    T(2, 3) = pose_msg.position.z;
    return T;
}

Eigen::MatrixXd makePcld4d(const Eigen::MatrixXf& pts) {
    const int n = static_cast<int>(pts.rows());
    Eigen::MatrixXd result(n, 4);
    result.leftCols(3) = pts.cast<double>();
    result.col(3) = result.leftCols(3).rowwise().norm();
    return result;
}

} // namespace gmmslam
