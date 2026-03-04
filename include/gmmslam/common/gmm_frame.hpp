#pragma once

#include <memory>
#include <Eigen/Geometry>
#include <sogmm_open3d/SOGMMGPU.h>
#include <self_organizing_gmm/SOGMMCPU.h>
// GmmFrame holds the SOGMM fitted to one scan, plus its pose in the world.
// The SOGMM is 4-dimensional: (x, y, z, intensity).
struct GmmFrame
{
    using Ptr      = std::shared_ptr<GmmFrame>;
    using ConstPtr = std::shared_ptr<const GmmFrame>;

    sogmm::cpu::SOGMM<float, 4> sogmm;

    // Timestamp [seconds]
    double timestamp = 0.0;

    int frame_id = -1;

    // Pose of this frame's sensor in the world frame (set after optimization)
    Eigen::Isometry3d T_world_sensor = Eigen::Isometry3d::Identity();
};