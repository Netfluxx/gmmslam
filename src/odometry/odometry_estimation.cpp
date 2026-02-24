#include <gslam/odometry/odometry_estimation.hpp>

//input: current GmmFrame, previous GmmFrame, initial guess of the relative pose
//output: relative pose between the two frames
// wrapper for gmm_d2d_registration from gira

gtsam::Pose3 estimate(const GmmFrame& source, const GmmFrame& target, 
                       const gtsam::Pose3& initial_guess);