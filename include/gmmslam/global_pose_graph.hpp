#pragma once

#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <nav_msgs/Path.h>

namespace gmmslam {

class GlobalPoseGraph {
public:
    // Callback types for cross-module access
    using GetPoseFn = std::function<std::optional<Matrix4d>(int)>;
    using GetGmmFn = std::function<std::optional<std::pair<GmmModel, Matrix4d>>(int)>;
    using GetPoseUncertaintyFn = std::function<std::optional<double>(int)>;
    using GetSubmapTrajDeltaFn = std::function<std::optional<Matrix4d>(int, int, double, double)>;

    GlobalPoseGraph(const GlobalGraphConfig& gg_cfg,
                    const RegistrationConfig& reg_cfg,
                    const LoopClosureConfig& lc_cfg,
                    const MapConfig& map_cfg,
                    const std::string& odom_frame,
                    const std::string& gmm_dir,
                    ros::Publisher path_pub,
                    ros::Publisher reg_request_pub,
                    GetPoseFn get_pose_fn,
                    GetGmmFn get_gmm_fn = nullptr,
                    GetPoseUncertaintyFn get_pose_uncertainty_fn = nullptr,
                    GetSubmapTrajDeltaFn get_traj_delta_fn = nullptr);

    bool shouldCreateSubmap(int key_idx) const;

    void updateWithKeyframe(int key_idx, const ros::Time& stamp,
                            const Matrix4d& T_curr, double t_sec);

    /// Retry merging GMMs for submaps whose rollover happened before async SOGMM
    /// fits finished (typically invoked from RegistrationManager after each fit).
    void processPendingSubmapFinalizations(const ros::Time& stamp);

    /// Set when submap traj auxiliary factor fails absurd-motion gates; node
    /// consumes to re-anchor the fixed-lag smoother on GT.
    bool consumeSmootherReanchorRequest();

    void handleSubmapRegistrationResult(int sid_prev, int sid_curr,
                                        const Matrix4d& T_rel, double score,
                                        const ros::Time& stamp);

    /// Called for every overlap D2D outcome (clears pending edge and overlap counter).
    void acknowledgeSubmapOverlapAttempt(int sid_prev, int sid_curr,
                                         const ros::Time& stamp);

    /// If submap `sid_new` finished its overlap D2D wave, apply deferred internal prune.
    void maybeApplyDeferredInternalSubmapPrune(int sid_new, const ros::Time& stamp);

    void addLoopFactor(int prev_key_idx, int curr_key_idx,
                       const Matrix4d& T_prev_to_curr,
                       const ros::Time& stamp,
                       const std::map<int, Matrix4d>& pose_by_idx,
                       double score = -1.0);

    void commit(const ros::Time& stamp);

    // Public state (protected by lock). Recursive: commit() is nested from
    // updateWithKeyframe / addLoopFactor / handleSubmapRegistrationResult.
    mutable std::recursive_mutex lock;
    bool enable;
    std::vector<int> submap_ids;
    std::map<int, Matrix4d> submap_pose_by_idx;
    std::map<int, int> submap_anchor_key;
    std::map<int, double> submap_anchor_time_sec;
    std::map<int, int> key_to_submap;
    std::map<int, Matrix4d> keyframe_pose_by_idx;
    std::map<int, std::vector<int>> submap_keyframes;
    std::map<int, GmmModel> submap_gmm;
    std::map<int, std::string> submap_gmm_path;
    std::map<int, std::vector<GmmLocalData>> submap_gmm_components;
    std::map<int, Matrix4d> submap_frozen_pose_by_idx;
    std::set<std::pair<int,int>> loop_edges_added;

    static std::array<double,3> submapColor(int sid);

private:
    static gtsam::Pose3 toPose3(const Matrix4d& T);
    static Matrix4d toMatrix(const gtsam::Pose3& p);
    static double rotAngleDeg(const Matrix3d& R);

    struct NoiseResult {
        gtsam::SharedNoiseModel noise;
        double sigma_t;
        double sigma_r;
    };
    NoiseResult noiseFromScore(double score,
                                double sigma_t_min, double sigma_t_max,
                                double sigma_r_min, double sigma_r_max) const;
    bool passesAuxGate(const std::string& name, const Matrix4d& T_aux,
                       const Matrix4d* T_ref, bool use_traj_aux_limits) const;

    void enqueueSubmapFinalization(int sid, const ros::Time& stamp);
    /// Returns true if submap is finalized (or already had a merged GMM), false if
    /// still waiting for keyframe GMMs from the async fit worker.
    bool tryFinalizeSubmap(int sid, const ros::Time& stamp,
                           const ros::Time& pending_since);
    void requestOverlapRegistrations(int sid_new, const ros::Time& stamp);
    void addTransitionAuxFactors(int prev_sid, int curr_sid,
                                  const Matrix4d& T_ref_rel,
                                  int prev_anchor_key, int curr_anchor_key,
                                  double prev_anchor_t, double curr_anchor_t);

    /// Concatenate two finalized submap GMMs in world frame, run the same
    /// Bhattacharyya pruning as submap finalization, then split components
    /// back into each submap's body frame (by source keyframe → submap).
    void pruneSubmapPairGmmsAfterLoop(int sid_a, int sid_b,
                                      const ros::Time& stamp);

    // Config
    std::string odom_frame_;
    std::string gmm_dir_;
    MapConfig map_cfg_;
    int submap_keyframes_per_submap_;
    double overlap_radius_m_;
    double submap_reg_score_threshold_;
    int min_loop_submap_gap_;
    bool enable_traj_aux_factors_;
    double score_sigma_low_;
    double score_sigma_high_;

    // Sigma bounds
    double loop_sigma_t_min_, loop_sigma_t_max_;
    double loop_sigma_r_min_, loop_sigma_r_max_;
    double submap_loop_sigma_t_min_, submap_loop_sigma_t_max_;
    double submap_loop_sigma_r_min_, submap_loop_sigma_r_max_;
    double keyframe_loop_consistency_trans_m_;
    double keyframe_loop_consistency_rot_deg_;
    double aux_gate_abs_trans_m_, aux_gate_abs_rot_deg_;
    double aux_gate_consistency_trans_m_, aux_gate_consistency_rot_deg_;
    double traj_aux_gate_abs_trans_m_, traj_aux_gate_abs_rot_deg_;
    double traj_aux_gate_consistency_trans_m_, traj_aux_gate_consistency_rot_deg_;

    /// SOGMM workers call processPendingSubmapFinalizations concurrently; serialize
    /// the heavy merge / disk / overlap-request path to avoid races and flaky crashes.
    mutable std::mutex finalize_serialization_mu_;

    // Noise models
    gtsam::SharedNoiseModel between_noise_;
    gtsam::SharedNoiseModel submap_traj_noise_;
    gtsam::SharedNoiseModel prior_noise_;
    gtsam::SharedNoiseModel loop_super_noise_;

    // iSAM2
    std::unique_ptr<gtsam::ISAM2> isam_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;

    // Submap lifecycle state
    int last_submap_idx_ = -1;
    int last_submap_anchor_key_idx_ = -1;
    Matrix4d last_submap_pose_ = Matrix4d::Identity();
    double last_submap_t_sec_ = 0.0;

    // Pending registrations
    std::set<std::pair<int,int>> pending_submap_registrations_;

    std::vector<std::pair<int, ros::Time>> pending_submap_finalize_;

    bool reanchor_on_traj_fail_ = true;
    int submap_finalize_min_ready_keyframes_ = 1;
    double submap_finalize_min_ready_fraction_ = 0.0;
    double submap_finalize_max_wait_s_ = 0.0;
    std::atomic<bool> smoother_reanchor_requested_{false};

    /// Remaining overlap D2D requests issued when submap `sid` was finalized (sid is
    /// always the larger index / "curr" side in prev/curr pairs). When zero, internal
    /// prune may run after overlap wave (see submap_needs_internal_prune_after_overlap_wave_).
    std::map<int, int> submap_overlap_d2d_remaining_;
    std::set<int> submap_needs_internal_prune_after_overlap_wave_;

    // Callbacks
    GetPoseFn get_pose_;
    GetGmmFn get_gmm_;
    GetPoseUncertaintyFn get_pose_uncertainty_;
    GetSubmapTrajDeltaFn get_traj_delta_;

    // Publishers
    ros::Publisher path_pub_;
    ros::Publisher reg_request_pub_;
    nav_msgs::Path path_;
};

} // namespace gmmslam
