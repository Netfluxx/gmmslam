// Maintain a keyframe GMM submap (merge nearby GMMs)
// Add odometry factors to a GTSAM NonlinearFactorGraph
// Add loop closure by re-running D2D registration on candidate keyframe pairs

//the idea of a submap from glim:
//consumes a stream of LiDAR (and maybe IMU later) odometry frames, builds a small local factor graph 
//over a short window, optimizes it, and when it has enough keyframes it exports one SubMap:
    //  a merged GMM local map
    //  poses for frames inside that window (refined)
    //  “endpoints” transforms that describe how the submap connects to the trajectory before/after the window

#include <memory>
#include <deque>
#include <random>
#include <vector>

#include <gmmslam/include/gslam/mapping/sub_mapping.hpp>
#include <gmmslam/common/gmm_frame.hpp>

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;




//on each new odom frame:
// add pose node to factor graph, with initial pose from odometry
// add between factor to previous pose node, with relative pose from odometry
// decide based on displacement if keyframe or not
// if keyframe:
//      add D2D registration factors to all previous keyframes in the submap
// if nbr of keyframes > max, LM optimize submap graph, merge keyframes into single submap, and export it to global mapping,
// then clear the submap graph and start building the next one.


//read params from yaml file in gmmslam/config/params.yaml
SubMapParams params;

SubMapping::SubMapping(const SubMapParams& params) : params_(params) {
    reset_submap();
}

SubMapping::~SubMapping() {}

void SubMapping::insert_keyframe(const GmmFrame::ConstPtr& keyframe) {

    const int current = static_cast<int>(keyframes_.size()); // current index of the new keyframe
    keyframes_.push_back(keyframe);
    values_->insert(X(current), gtsam::Pose3(keyframe->T_world_sensor.matrix()));
    // if first node of the graph
    if (current == 0) {
        graph_->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(keyframe->T_world_sensor.matrix()), gtsam::noiseModel::Isotropic::Sigma(6, 1e-6));
    } else {
        // add between factor to previous pose
        const Eigen::Isometry3d T_prev_curr = keyframes_[current - 1]->T_world_sensor.inverse() * keyframe->T_world_sensor;
        // add between factor to previous pose with relative pose from odometry
        graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(current - 1), X(current), gtsam::Pose3(T_prev_curr.matrix()), gtsam::noiseModel::Isotropic::Precision(6, params_.between_factor_precision));
    }

    // add D2D registration factors to all previous keyframes in the submap
    for (int i = 0; i < current; ++i) {
        const Eigen::Isometry3d T_prev_curr = keyframes_[i]->T_world_sensor.inverse() * keyframe->T_world_sensor;
        // call D2D registration to get relative pose and score
        Eigen::Isometry3d T_prev_curr_d2d;
        float score = register_gmm_frames_cpu(*keyframes_[i], *keyframe, T_prev_curr.cast<float>(), T_prev_curr_d2d.cast<float>());
        // add D2D registration factor to the graph
        graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(i), X(current), gtsam::Pose3(T_prev_curr_d2d.matrix()), gtsam::noiseModel::Isotropic::Precision(6, params_.d2d_factor_precision));
    }

    // keyframe selection based on odometry displacement
    bool is_keyframe = false;
    if (current > 0) {
        // T_prev_curr = T_prev_sensor_world * T_world_sensor : transform from previous sensor frame to current sensor frame, expressed in world frame
        const Eigen::Isometry3d T_prev_curr = keyframes_[current - 1]->T_world_sensor.inverse() * keyframe->T_world_sensor;
        const double translation = (T_prev_curr.translation()).norm();
        const double rotation = (T_prev_curr.rotation().eulerAngles(0,1,2)).norm();
        if (translation > params_.keyframe_threshold_translation || rotation > params_.keyframe_threshold_rotation) {
            is_keyframe = true;
        }
    }

    // create submap if enough keyframes
    if (static_cast<int>(keyframes_.size()) >= params_.max_nbr_keyframes) {
        // optimize submap graph with LM
        try{
            gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *values_);
            gtsam::Values result = optimizer.optimize();
        } catch (const std::exception& e) {
            std::cerr << "Error optimizing submap graph: " << e.what() << std::endl;
            return;
        }
        // gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *values_);
        // gtsam::Values result = optimizer.optimize();

        // merge keyframes into single submap (TODO: implement merge function)
        Submap submap;
        submap.id = 0; // TODO: assign unique ID
        submap.T_world_origin = keyframes_.front()->T_world_sensor; // TODO: compute actual origin
        submap.merged_gmm = merge_keyframes_into_submap(keyframes_); // TODO: implement merge function

        // export submap to global mapping (TODO: implement export function)
        export_submap_to_global_mapping(submap);

        // clear submap graph and start building the next one
        reset_submap();
    }

}

std::vector<GmmFrame::ConstPtr> SubMapping::get_submaps() const {
    std::vector<GmmFrame::ConstPtr> submaps;
    for (const auto& submap : submaps_) {
        submaps.push_back(std::make_shared<GmmFrame>(export_submap_to_global_mapping(submap)));
    }
    return submaps;
}

std::vector<GmmFrame::ConstPtr> SubMapping::export_submap_to_global_mapping(const Submap& submap) {
    
}
    




