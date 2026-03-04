#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gmmslam/common/gmm_frame.hpp>

// Estimate the relative pose (T_source_in_target) between two GMM frames
// using on-manifold D2D GMM registration from gira3d
//
// Inputs:
//   source        – current frame (moving)
//   target        – previous frame (fixed reference)
//   initial_guess – SE3 prior for the relative pose
//
// Returns: gtsam::Pose3  T_source_in_target
gtsam::Pose3 estimate(const GmmFrame&     source,
                       const GmmFrame&     target,
                       const gtsam::Pose3& initial_guess);
