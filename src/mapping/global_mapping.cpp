

#include <map>
#include <unordered_set>

#include <gtsam/base/serialization.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PoseRotationPrior.h>
#include <gtsam/slam/PoseTranslationPrior.h>

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>

#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/isam2_ext_dummy.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

//idea of global map from glim:
//consumes a stream of SubMaps, inserts each as a node in a global factor graph,
// and keeps optimizing the graph of submaps incrementally with ISAM2.
// It adds constaints between neighbor submaps (betweenFactors)
// it also adds loop closure by checking overlap between submaps (matching cost factors) --> need to find equivalent for GMMs
// Each submap is a GMM, so we can add factors between submaps based on the D2D registration cost between their GMMs.
// later add IMU constraints connecting submaps.


//addOdomFactor() adds a prior for the first pose
//Add a BetweenFactor between last and new pose, or from loop closures

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,roll,pitch,yaw)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::E; // Edge


class GlobalMapOptimization {
public:
    // gtsam related
    NonlinearFactorGraph gtsam_global_graph;
    Values gtsam_initial_estimate;
    ISAM2 *gtsam_isam2;
    Values isam_current_estimate;
    Eigen::MatrixXd pose_covariance;

    bool is_degenerate = false;
    bool loop_closure_added = false;
}


void GlobalMapOptimization::insert_submap(const SubMap::Ptr& submap) {
    // Add the submap as a node in the global factor graph
    // Add BetweenFactors to connect it to neighboring submaps
    // Add loop closure factors if there is significant overlap with existing submaps
    // Optimize the graph incrementally with ISAM2

    const int current = submaps.size();
    const int last = current - 1;
    insert_submap_to_graph(current, submap);

    gtsam::Pose3 current_T_world_submap = gtsam::Pose3::Identity();
    gtsam::Pose3 last_T_world_submap = gtsam::Pose3::Identity();

    if(current != 0){
        if(isam2->valueExists(X(last))){
            last_T_world_submap = isam2->calculateEstimatePose3(X(last));
        }else{
            last_T_world_submap = new_values->at<gtsam::Pose3>(X(last));
        }
    
        const Eigen::Isometry3d T_origin0_endpointR0 = submaps[last]->T_origin_endpoint_R;
        const Eigen::Isometry3d T_origin1_endpointL1 = submaps[current]->T_origin_endpoint_L;
        const Eigen::Isometry3d T_endpointR0_endpointL1 = submaps[last]->odom_frames.back()->T_world_sensor().inverse() * submaps[current]->odom_frames.front()->T_world_sensor();
        const Eigen::Isometry3d T_origin0_origin1 = T_origin0_endpointR0 * T_endpointR0_endpointL1 * T_origin1_endpointL1.inverse();
        
        current_T_world_submap = last_T_world_submap * gtsam::Pose3(T_origin0_origin1.matrix());
    } else{
        // For the first submap, we can set its pose to zero
        current_T_world_submap = gtsam::Pose3(submap->T_world_origin.matrix());
    }




}