#pragma once

#include <memory>
#include <deque>
#include <random>
#include <vector>

#include <gmmslam/common/gmm_frame.hpp>


struct SubMapParams {
    int max_nbr_keyframes;
    double keyframe_selection_score_threshold;
    double keyframe_threshold_translation; // [m]
    double keyframe_threshold_rotation;    // [rad]
};

struct Submap {
    int id = 0;

    // origin and endpoints relative to submap
    Eigen::Isometry3d T_world_origin = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_origin_endpoint_L = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_origin_endpoint_R = Eigen::Isometry3d::Identity();

    // Merged submap GMM (GmmFrame) in ORIGIN frame
    GmmFrame merged_gmm;

    // optimized poses for frames (in world)
    std::vector<Eigen::Isometry3d> T_world_sensor_optimized;
};

class SubMapping {
public:
    SubMapping(const SubMapParams& params);
    virtual ~SubMapping();

    virtual void insert_keyframe(const GmmFrame::ConstPtr& keyframe) override;
    virtual std::vector<GmmFrame::ConstPtr> get_submaps() const override;
    void reset_submap();


private:
    SubMapParams params_;
    std::deque<GmmFrame::ConstPtr> keyframes_;
    gtsam::NonlinearFactorGraph graph_;
    std::vector<Submap> submaps_;

    GmmFrame merge_keyframes_into_submap(const std::vector<GmmFrame::ConstPtr>& keyframes);
    GmmFrame export_submap_to_global_mapping(const Submap& submap);
}