// Maintain a keyframe GMM submap (merge nearby GMMs)
// Add odometry factors to a GTSAM NonlinearFactorGraph
// Add loop closure by re-running D2D registration on candidate keyframe pairs

//the idea of a submap from glim:
//consumes a stream of LiDAR (and maybe IMU later) odometry frames, builds a small local factor graph 
//over a short window, optimizes it, and when it has enough keyframes it exports one SubMap:
    //  a merged GMM local map
    //  poses for frames inside that window (refined)
    //  “endpoints” transforms that describe how the submap connects to the trajectory before/after the window


#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/cuda/cuda_stream.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

