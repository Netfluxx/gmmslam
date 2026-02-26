//input: current GmmFrame, previous GmmFrame, initial guess of the relative pose
//output: relative pose between the two frames
// wrapper for gmm_d2d_registration from gira


#include <gslam/odometry/odometry_estimation.hpp>
#include <gtsam/geometry/Pose3.h>
//iSAM2 Fixed Lag Smoother : https://gitlab.sintef.no/nix-flakes/gtsam/-/blob/46f3a48a5b1e6b57bc5da5374c74a8f5248b6fec/gtsam_unstable/examples/FixedLagSmootherExample.cpp

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>


// gtsam::Pose3 estimate(const GmmFrame& source, const GmmFrame& target, 
//                        const gtsam::Pose3& initial_guess);

using namespace gtsam;
// using namespace std;

// Configure iSAM2
ISAM2Params isam2_params;
isam2_params.relinearizeThreshold = 0.1;
isam2_params.relinearizeSkip = 1;
isam2_params.cacheLinearizedFactors = false;

double lag = 2.0; // 2 second window
IncrementalFixedLagSmoother smoother(lag, isam2_params);

//every new variable must be assigned a timestamp so the smoother knows when to marginalize it out


NonlinearFactorGraph new_factors;
Values new_values;
FixedLagSmoother::KeyTimestampMap new_timestamps;

double t = current_time_seconds; // e.g. from header stamp
Key pose_key = symbol_shorthand::X(frame_id);


current_pose_estimate = Pose3::Identity(); // TODO: use odometry input
new_values.insert(pose_key, current_pose_estimate);
new_timestamps[pose_key] = t;

// Add a between factor to the previous pose
new_factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
    gtsam::symbol_shorthand::X(frame_id - 1),
    pose_key,
    relative_pose,         // from odometry_estimation
    odometry_noise_model
));

smoother.update(new_factors, new_values, new_timestamps);

gtsam::Values result = smoother.calculateEstimate();
gtsam::Pose3 pose = result.at<gtsam::Pose3>(pose_key);