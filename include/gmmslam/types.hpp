#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace gmmslam {

using Matrix4d = Eigen::Matrix4d;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

struct GmmComponent {
    Vector3d mean;
    Matrix3d covariance;
    double weight;
};

struct GmmLocalData {
    Vector3d scales;     // sqrt(eigenvalues)
    Matrix3d rotation;   // eigenvector matrix (proper rotation)
    Vector3d mean_local; // mean in local frame
};

struct GmmModel {
    std::vector<GmmComponent> components;
    int numComponents() const { return static_cast<int>(components.size()); }
};

struct GmmFrame {
    using Ptr = std::shared_ptr<GmmFrame>;
    using ConstPtr = std::shared_ptr<const GmmFrame>;

    GmmModel gmm;
    double timestamp = 0.0;
    int frame_id = -1;
    Matrix4d pose = Matrix4d::Identity();
};

struct LocalGmmEntry {
    double stamp_sec = 0.0;
    GmmModel model;
    Matrix4d map_pose = Matrix4d::Identity();
    double capture_t_sec = 0.0;
    double fit_t_sec = 0.0;
    Matrix4d capture_pose = Matrix4d::Identity();
    bool has_map_pose = false;
    bool has_capture_pose = false;
};

} // namespace gmmslam
