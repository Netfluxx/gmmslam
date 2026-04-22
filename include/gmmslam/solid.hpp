#pragma once

// SOLiD: Spatially Organized and Lightweight Global Descriptor for
// FOV-constrained LiDAR Place Recognition.
//
// Credit to:
//   @article{kim2024narrowing,
//     title  = {Narrowing your FOV with SOLiD: Spatially Organized and
//               Lightweight Global Descriptor for FOV-constrained LiDAR
//               Place Recognition},
//     author = {Kim, Hogyun and Choi, Jiwon and Sim, Taehu and
//               Kim, Giseop and Cho, Younggun},
//     journal= {IEEE Robotics and Automation Letters},
//     year   = {2024},
//     publisher = {IEEE}
//   }
//
// This implementation is adapted to the gmmslam pipeline: it consumes the
// already range/voxel-filtered Eigen::MatrixXf point cloud in the sensor
// frame, avoids any PCL dependency, and exposes a configurable SolidConfig.
// The angle-vector shift search is FOV-aware: shifts that yield too little
// non-zero overlap are rejected, which is the key fidelity fix for scanners
// with narrow azimuthal coverage (here a 120° depth camera).

#include "gmmslam/config.hpp"

#include <Eigen/Core>
#include <limits>

namespace gmmslam {

struct SolidDescriptor {
    // Concatenated [range_vec (num_range) | angle_vec (num_angle)].
    Eigen::VectorXd vec;
    // Cached L2 norm of the range head. 0 when descriptor is empty/invalid.
    double range_norm = 0.0;
    // Number of points that contributed (0 means "invalid / drop").
    int point_count = 0;

    bool empty() const { return vec.size() == 0 || point_count == 0; }
};

class SOLiDModule {
public:
    struct YawEstimate {
        // Signed yaw in radians, in (-pi, pi]. Sign convention: rotation of
        // the query sensor frame that best aligns it with the candidate
        // sensor frame, subject to SolidConfig::yaw_sign.
        double yaw_rad = 0.0;
        double l1_distance = std::numeric_limits<double>::infinity();
        double overlap = 0.0;  // fraction of bins non-zero in both, after shift
        bool   valid = false;
    };

    explicit SOLiDModule(const SolidConfig& cfg);

    // Build a SOLiD descriptor from a preprocessed Nx3 cloud in the sensor
    // frame. Returns an empty descriptor if the input is insufficient.
    SolidDescriptor makeDescriptor(const Eigen::MatrixXf& pts) const;

    // Cosine similarity on the range head. Returns 0 if either descriptor
    // is empty.
    double rangeCosine(const SolidDescriptor& q,
                       const SolidDescriptor& c) const;

    // FOV-aware yaw estimate via shift search on the angle head.
    YawEstimate yawEstimate(const SolidDescriptor& q,
                            const SolidDescriptor& c) const;

    const SolidConfig& config() const { return cfg_; }
    int numRange() const { return cfg_.num_range; }
    int numAngle() const { return cfg_.num_angle; }
    int numHeight() const { return cfg_.num_height; }

private:
    SolidConfig cfg_;
    double gap_angle_deg_;
    double gap_range_m_;
    double gap_height_deg_;
};

} // namespace gmmslam
