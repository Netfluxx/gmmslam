#include "gmmslam/ros_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

std::optional<OrganizedDepthImage> pc2ToOrganizedDepth(
    const sensor_msgs::PointCloud2& msg,
    double min_range,
    double max_range,
    bool estimate_intrinsics,
    double fx,
    double fy,
    double cx,
    double cy,
    double horizontal_fov_deg) {

    if (msg.height <= 1 || msg.width <= 1) {
        return std::nullopt;
    }

    OrganizedDepthImage organized;
    organized.depth =
        Eigen::MatrixXf::Zero(static_cast<int>(msg.height),
                              static_cast<int>(msg.width));

    sensor_msgs::PointCloud2ConstIterator<float> it_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(msg, "z");

    const double min_r = std::max(0.0, min_range);
    const double max_r = max_range > 0.0 ? max_range
                                         : std::numeric_limits<double>::infinity();

    double sum_u = 0.0, sum_u2 = 0.0, sum_xz = 0.0, sum_u_xz = 0.0;
    double sum_v = 0.0, sum_v2 = 0.0, sum_yz = 0.0, sum_v_yz = 0.0;
    int n_fit = 0;

    for (uint32_t v = 0; v < msg.height; ++v) {
        for (uint32_t u = 0; u < msg.width; ++u, ++it_x, ++it_y, ++it_z) {
            const float x = *it_x;
            const float y = *it_y;
            const float z = *it_z;
            // Webots RangeFinder point clouds are published in the camera body
            // frame used by the autopilot: X forward, Y left, Z up.  GMMap
            // wants the original pinhole depth image, so depth is the forward
            // X coordinate, not the ROS/Webots Z coordinate.
            if (!std::isfinite(x) || !std::isfinite(y) ||
                !std::isfinite(z) || x <= 0.0f) {
                continue;
            }

            const double depth_m = static_cast<double>(x);
            if (depth_m < min_r || depth_m > max_r) {
                continue;
            }

            organized.depth(static_cast<int>(v), static_cast<int>(u)) = x;
            ++organized.valid_points;

            const double du = static_cast<double>(u);
            const double dv = static_cast<double>(v);
            const double right_over_depth = -static_cast<double>(y) / x;
            const double down_over_depth = -static_cast<double>(z) / x;
            sum_u += du;
            sum_u2 += du * du;
            sum_xz += right_over_depth;
            sum_u_xz += du * right_over_depth;
            sum_v += dv;
            sum_v2 += dv * dv;
            sum_yz += down_over_depth;
            sum_v_yz += dv * down_over_depth;
            ++n_fit;
        }
    }

    if (organized.valid_points < 16) {
        return std::nullopt;
    }

    const double width = static_cast<double>(msg.width);
    const double height = static_cast<double>(msg.height);
    if (fx > 0.0 && fy > 0.0) {
        organized.fx = fx;
        organized.fy = fy;
        organized.cx = (cx >= 0.0) ? cx : (width - 1.0) * 0.5;
        organized.cy = (cy >= 0.0) ? cy : (height - 1.0) * 0.5;
    } else if (!estimate_intrinsics) {
        const double pi = std::acos(-1.0);
        const double fov_rad =
            std::clamp(horizontal_fov_deg, 1.0, 179.0) * pi / 180.0;
        const double focal = width / (2.0 * std::tan(0.5 * fov_rad));
        organized.fx = focal;
        organized.fy = focal;
        organized.cx = (cx >= 0.0) ? cx : (width - 1.0) * 0.5;
        organized.cy = (cy >= 0.0) ? cy : (height - 1.0) * 0.5;
    } else {
        const double denom_u =
            static_cast<double>(n_fit) * sum_u2 - sum_u * sum_u;
        const double denom_v =
            static_cast<double>(n_fit) * sum_v2 - sum_v * sum_v;
        if (std::abs(denom_u) < 1e-9 || std::abs(denom_v) < 1e-9) {
            return std::nullopt;
        }

        const double ax =
            (static_cast<double>(n_fit) * sum_u_xz - sum_u * sum_xz) / denom_u;
        const double bx = (sum_xz - ax * sum_u) / static_cast<double>(n_fit);
        const double ay =
            (static_cast<double>(n_fit) * sum_v_yz - sum_v * sum_yz) / denom_v;
        const double by = (sum_yz - ay * sum_v) / static_cast<double>(n_fit);
        if (std::abs(ax) < 1e-9 || std::abs(ay) < 1e-9) {
            return std::nullopt;
        }

        organized.fx = 1.0 / ax;
        organized.fy = 1.0 / ay;
        organized.cx = -bx / ax;
        organized.cy = -by / ay;
    }

    if (!std::isfinite(organized.fx) || !std::isfinite(organized.fy) ||
        !std::isfinite(organized.cx) || !std::isfinite(organized.cy) ||
        organized.fx <= 0.0 || organized.fy <= 0.0 ||
        organized.cx < -width || organized.cx > 2.0 * width ||
        organized.cy < -height || organized.cy > 2.0 * height) {
        return std::nullopt;
    }

    return organized;
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
