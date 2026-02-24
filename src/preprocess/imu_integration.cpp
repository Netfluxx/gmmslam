// #include <gtsam/geometry/Rot3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/navigation/GPSFactor.h>
// #include <gtsam/navigation/ImuFactor.h>
// #include <gtsam/navigation/CombinedImuFactor.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/inference/Symbol.h>

// #include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

// using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
// using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
// using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

// class IMUIntegration {

// public:
//     bool systemInitialized = false;

//     gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
//     gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
//     gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
//     gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
//     gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
// };