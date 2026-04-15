#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/gmm_utils.hpp"
#include "gmmslam/ros_helpers.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <nlohmann/json.hpp>
#include <std_msgs/String.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

using gtsam::symbol_shorthand::X;

namespace gmmslam {

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

gtsam::Pose3 GlobalPoseGraph::toPose3(const Matrix4d& T) {
    return gtsam::Pose3(gtsam::Rot3(T.block<3,3>(0,0)),
                        gtsam::Point3(T(0,3), T(1,3), T(2,3)));
}

Matrix4d GlobalPoseGraph::toMatrix(const gtsam::Pose3& p) {
    return p.matrix();
}

double GlobalPoseGraph::rotAngleDeg(const Matrix3d& R) {
    double c = std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(c) * 180.0 / M_PI;
}

std::array<double,3> GlobalPoseGraph::submapColor(int sid) {
    return gmmslam::submapColor(sid);
}

// ---------------------------------------------------------------
// noiseFromScore
// ---------------------------------------------------------------

GlobalPoseGraph::NoiseResult GlobalPoseGraph::noiseFromScore(
        double score,
        double sigma_t_min, double sigma_t_max,
        double sigma_r_min, double sigma_r_max) const {
    double s_lo = score_sigma_low_;
    double s_hi = score_sigma_high_;
    if (s_hi <= s_lo) s_hi = s_lo + 1e-6;

    double alpha = std::clamp((score - s_lo) / (s_hi - s_lo), 0.0, 1.0);
    double sigma_t = sigma_t_max - alpha * (sigma_t_max - sigma_t_min);
    double sigma_r = sigma_r_max - alpha * (sigma_r_max - sigma_r_min);

    Eigen::Matrix<double,6,1> sigmas;
    sigmas << sigma_r, sigma_r, sigma_r, sigma_t, sigma_t, sigma_t;
    return {gtsam::noiseModel::Diagonal::Sigmas(sigmas), sigma_t, sigma_r};
}

// ---------------------------------------------------------------
// passesAuxGate
// ---------------------------------------------------------------

bool GlobalPoseGraph::passesAuxGate(const std::string& name,
                                     const Matrix4d& T_aux,
                                     const Matrix4d* T_ref) const {
    if (!T_aux.allFinite()) return false;

    const double d_abs = T_aux.block<3,1>(0,3).norm();
    const double r_abs = rotAngleDeg(T_aux.block<3,3>(0,0));

    if (d_abs > aux_gate_abs_trans_m_) {
        ROS_WARN("[global_graph] rejected %s submap factor (abs trans %.3fm > %.3fm)",
                 name.c_str(), d_abs, aux_gate_abs_trans_m_);
        return false;
    }
    if (r_abs > aux_gate_abs_rot_deg_) {
        ROS_WARN("[global_graph] rejected %s submap factor (abs rot %.2fdeg > %.2fdeg)",
                 name.c_str(), r_abs, aux_gate_abs_rot_deg_);
        return false;
    }

    if (T_ref != nullptr && T_ref->allFinite()) {
        const Matrix4d T_err = T_ref->inverse() * T_aux;
        const double d_err = T_err.block<3,1>(0,3).norm();
        const double r_err = rotAngleDeg(T_err.block<3,3>(0,0));

        if (d_err > aux_gate_consistency_trans_m_) {
            ROS_WARN("[global_graph] rejected %s submap factor "
                     "(consistency trans %.3fm > %.3fm)",
                     name.c_str(), d_err, aux_gate_consistency_trans_m_);
            return false;
        }
        if (r_err > aux_gate_consistency_rot_deg_) {
            ROS_WARN("[global_graph] rejected %s submap factor "
                     "(consistency rot %.2fdeg > %.2fdeg)",
                     name.c_str(), r_err, aux_gate_consistency_rot_deg_);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------

GlobalPoseGraph::GlobalPoseGraph(
        const GlobalGraphConfig& gg_cfg,
        const RegistrationConfig& reg_cfg,
        const LoopClosureConfig& lc_cfg,
        const std::string& odom_frame,
        const std::string& gmm_dir,
        ros::Publisher path_pub,
        ros::Publisher reg_request_pub,
        GetPoseFn get_pose_fn,
        GetGmmFn get_gmm_fn,
        GetSubmapTrajDeltaFn get_traj_delta_fn)
    : enable(gg_cfg.enable)
    , odom_frame_(odom_frame)
    , gmm_dir_(gmm_dir)
    , submap_keyframes_per_submap_(gg_cfg.submap_keyframes_per_submap)
    , overlap_radius_m_(gg_cfg.overlap_radius_m)
    , submap_reg_score_threshold_(gg_cfg.reg_score_threshold)
    , score_sigma_low_(reg_cfg.score_sigma_low)
    , score_sigma_high_(reg_cfg.score_sigma_high)
    , loop_sigma_t_min_(reg_cfg.loop_sigma_t_min)
    , loop_sigma_t_max_(reg_cfg.loop_sigma_t_max)
    , loop_sigma_r_min_(reg_cfg.loop_sigma_r_min)
    , loop_sigma_r_max_(reg_cfg.loop_sigma_r_max)
    , submap_loop_sigma_t_min_(gg_cfg.submap_loop_sigma_t_min)
    , submap_loop_sigma_t_max_(gg_cfg.submap_loop_sigma_t_max)
    , submap_loop_sigma_r_min_(gg_cfg.submap_loop_sigma_r_min)
    , submap_loop_sigma_r_max_(gg_cfg.submap_loop_sigma_r_max)
    , aux_gate_abs_trans_m_(gg_cfg.aux_gate_abs_trans_m)
    , aux_gate_abs_rot_deg_(gg_cfg.aux_gate_abs_rot_deg)
    , aux_gate_consistency_trans_m_(gg_cfg.aux_gate_consistency_trans_m)
    , aux_gate_consistency_rot_deg_(gg_cfg.aux_gate_consistency_rot_deg)
    , get_pose_(std::move(get_pose_fn))
    , get_gmm_(std::move(get_gmm_fn))
    , get_traj_delta_(std::move(get_traj_delta_fn))
    , path_pub_(std::move(path_pub))
    , reg_request_pub_(std::move(reg_request_pub))
{
    std::filesystem::create_directories(gmm_dir_);

    // GTSAM noise ordering: 3 rotation + 3 translation
    auto makeSigmas = [](double sigma_r, double sigma_t) {
        Eigen::Matrix<double,6,1> s;
        s << sigma_r, sigma_r, sigma_r, sigma_t, sigma_t, sigma_t;
        return s;
    };

    between_noise_      = gtsam::noiseModel::Diagonal::Sigmas(
                              makeSigmas(gg_cfg.between_sigma_r, gg_cfg.between_sigma_t));
    submap_traj_noise_  = gtsam::noiseModel::Diagonal::Sigmas(
                              makeSigmas(gg_cfg.traj_sigma_r, gg_cfg.traj_sigma_t));
    prior_noise_        = gtsam::noiseModel::Diagonal::Sigmas(
                              makeSigmas(gg_cfg.prior_sigma_r, gg_cfg.prior_sigma_t));
    loop_super_noise_   = gtsam::noiseModel::Diagonal::Sigmas(
                              makeSigmas(lc_cfg.super_sigma_r, lc_cfg.super_sigma_t));

    isam_ = std::make_unique<gtsam::ISAM2>(gtsam::ISAM2Params());

    path_.header.frame_id = odom_frame_;
}

// ---------------------------------------------------------------
// Keyframe / submap lifecycle
// ---------------------------------------------------------------

bool GlobalPoseGraph::shouldCreateSubmap(int key_idx) const {
    if (last_submap_anchor_key_idx_ < 0) return true;
    return (key_idx - last_submap_anchor_key_idx_) >= submap_keyframes_per_submap_;
}

void GlobalPoseGraph::updateWithKeyframe(int key_idx, const ros::Time& stamp,
                                          const Matrix4d& T_curr, double t_sec) {
    if (!enable) return;

    if (last_submap_idx_ >= 0) {
        submap_keyframes[last_submap_idx_].push_back(key_idx);
        key_to_submap[key_idx] = last_submap_idx_;
    }

    if (!shouldCreateSubmap(key_idx)) return;

    if (last_submap_idx_ >= 0) {
        finalizeSubmap(last_submap_idx_, stamp);
    }

    const int sid = static_cast<int>(submap_ids.size());
    const auto key_sid = X(sid);

    {
        std::lock_guard<std::mutex> lk(lock);
        new_values_.insert(key_sid, toPose3(T_curr));

        if (sid == 0) {
            new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                key_sid, toPose3(T_curr), prior_noise_);
        } else {
            const Matrix4d rel = last_submap_pose_.inverse() * T_curr;
            new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(last_submap_idx_), key_sid, toPose3(rel), between_noise_);

            addTransitionAuxFactors(
                last_submap_idx_, sid, rel,
                submap_anchor_key.count(last_submap_idx_)
                    ? submap_anchor_key.at(last_submap_idx_) : -1,
                key_idx,
                submap_anchor_time_sec.count(last_submap_idx_)
                    ? submap_anchor_time_sec.at(last_submap_idx_) : 0.0,
                t_sec);
        }
    }

    submap_ids.push_back(sid);
    submap_pose_by_idx[sid] = T_curr;
    submap_anchor_key[sid] = key_idx;
    submap_anchor_time_sec[sid] = t_sec;
    last_submap_anchor_key_idx_ = key_idx;
    last_submap_idx_ = sid;
    last_submap_pose_ = T_curr;
    last_submap_t_sec_ = t_sec;
    submap_keyframes[sid] = {key_idx};
    key_to_submap[key_idx] = sid;

    const auto color = submapColor(sid);
    ROS_INFO("[global_graph] new submap S(%d) anchored at X(%d) "
             "color=[%.2f, %.2f, %.2f]",
             sid, key_idx, color[0], color[1], color[2]);

    commit(stamp);
}

// ---------------------------------------------------------------
// Transition auxiliary factors
// ---------------------------------------------------------------

void GlobalPoseGraph::addTransitionAuxFactors(
        int prev_sid, int curr_sid,
        const Matrix4d& T_ref_rel,
        int prev_anchor_key, int curr_anchor_key,
        double prev_anchor_t, double curr_anchor_t) {
    if (prev_anchor_key < 0 || curr_anchor_key < 0) return;
    if (!get_traj_delta_) return;

    try {
        auto opt_T_traj = get_traj_delta_(prev_anchor_key, curr_anchor_key,
                                          prev_anchor_t, curr_anchor_t);
        if (!opt_T_traj.has_value()) return;

        const Matrix4d& T_traj = opt_T_traj.value();
        if (passesAuxGate("traj", T_traj, &T_ref_rel)) {
            new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(prev_sid), X(curr_sid),
                toPose3(T_traj), submap_traj_noise_);

            const auto pos = T_traj.block<3,1>(0,3);
            ROS_INFO("[global_graph] ADDED BetweenFactor (submap traj) "
                     "S(%d)->S(%d) t=[%.3f, %.3f, %.3f]",
                     prev_sid, curr_sid, pos(0), pos(1), pos(2));
        }
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(2.0,
            "[global_graph] failed traj submap factor S(%d)->S(%d): %s",
            prev_sid, curr_sid, e.what());
    }
}

// ---------------------------------------------------------------
// Submap finalization
// ---------------------------------------------------------------

void GlobalPoseGraph::finalizeSubmap(int sid, const ros::Time& stamp) {
    if (!get_gmm_) return;

    const auto it_keys = submap_keyframes.find(sid);
    if (it_keys == submap_keyframes.end() || it_keys->second.empty()) return;
    const auto& key_indices = it_keys->second;

    const auto it_pose = submap_pose_by_idx.find(sid);
    if (it_pose == submap_pose_by_idx.end()) return;
    const Matrix4d& T_ref = it_pose->second;

    std::vector<std::pair<GmmModel, Matrix4d>> gmms_with_poses;
    for (int ki : key_indices) {
        auto result = get_gmm_(ki);
        if (!result.has_value()) continue;

        auto& [gmm, capture_pose] = result.value();
        Matrix4d T_kf = capture_pose;
        if (!T_kf.allFinite()) {
            auto opt_pose = get_pose_(ki);
            if (!opt_pose.has_value()) continue;
            T_kf = opt_pose.value();
        }
        if (gmm.components.empty() || !T_kf.allFinite()) continue;
        gmms_with_poses.emplace_back(std::move(gmm), T_kf);
    }

    if (gmms_with_poses.empty()) {
        ROS_WARN("[global_graph] S(%d): no keyframe GMMs available "
                 "(%zu keyframes, 0 GMMs)",
                 sid, key_indices.size());
        return;
    }

    GmmModel merged = mergeGmmsConcatenate(gmms_with_poses, T_ref);
    if (merged.components.empty()) return;

    submap_gmm[sid] = merged;
    submap_gmm_components[sid] = precomputeGmmLocalData(merged);
    submap_frozen_pose_by_idx[sid] = T_ref;

    ROS_INFO("[global_graph] S(%d) finalized: %d components "
             "from %zu/%zu keyframes",
             sid, merged.numComponents(),
             gmms_with_poses.size(), key_indices.size());

    const std::string gmm_path =
        gmm_dir_ + "/submap_" + [&]{
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%04d", sid);
            return std::string(buf);
        }() + ".gmm";

    try {
        saveGmmToFile(merged, gmm_path);
        submap_gmm_path[sid] = gmm_path;
    } catch (const std::exception& e) {
        ROS_WARN("[global_graph] failed to save S(%d) GMM: %s",
                 sid, e.what());
        return;
    }

    requestOverlapRegistrations(sid, stamp);
}

// ---------------------------------------------------------------
// Overlap-based submap registration
// ---------------------------------------------------------------

void GlobalPoseGraph::requestOverlapRegistrations(int sid_new,
                                                   const ros::Time& stamp) {
    if (!reg_request_pub_) return;

    const auto it_new = submap_pose_by_idx.find(sid_new);
    const auto it_path_new = submap_gmm_path.find(sid_new);
    if (it_new == submap_pose_by_idx.end() ||
        it_path_new == submap_gmm_path.end()) return;

    const Vector3d pos_new = it_new->second.block<3,1>(0,3);
    const std::string& new_path = it_path_new->second;

    for (int sid_old : submap_ids) {
        if (sid_old >= sid_new) continue;
        if (sid_old == sid_new - 1) continue;

        const auto edge = std::make_pair(std::min(sid_old, sid_new),
                                         std::max(sid_old, sid_new));
        if (loop_edges_added.count(edge) ||
            pending_submap_registrations_.count(edge)) continue;

        const auto it_old = submap_pose_by_idx.find(sid_old);
        const auto it_path_old = submap_gmm_path.find(sid_old);
        if (it_old == submap_pose_by_idx.end() ||
            it_path_old == submap_gmm_path.end()) continue;

        const double d = (it_old->second.block<3,1>(0,3) - pos_new).norm();
        if (d > overlap_radius_m_) continue;

        nlohmann::json payload;
        payload["prev_idx"]                 = sid_old;
        payload["curr_idx"]                 = sid_new;
        payload["stamp"]                    = stampToSec(stamp);
        payload["source_path"]              = new_path;
        payload["target_path"]              = it_path_old->second;
        payload["is_loop_closure"]          = false;
        payload["is_submap_registration"]   = true;

        std_msgs::String msg;
        msg.data = payload.dump();
        reg_request_pub_.publish(msg);

        pending_submap_registrations_.insert(edge);
        ROS_INFO("[global_graph] requested registration "
                 "S(%d)<->S(%d) (d=%.2fm)", sid_old, sid_new, d);
    }
}

// ---------------------------------------------------------------
// Submap registration result
// ---------------------------------------------------------------

void GlobalPoseGraph::handleSubmapRegistrationResult(
        int sid_prev, int sid_curr,
        const Matrix4d& T_rel, double score,
        const ros::Time& stamp) {
    const auto edge = std::make_pair(std::min(sid_prev, sid_curr),
                                     std::max(sid_prev, sid_curr));
    pending_submap_registrations_.erase(edge);

    if (score < submap_reg_score_threshold_) {
        ROS_INFO("[global_graph] submap reg S(%d)->S(%d) "
                 "rejected (score=%.4f < %.4f)",
                 sid_prev, sid_curr, score, submap_reg_score_threshold_);
        return;
    }

    if (loop_edges_added.count(edge)) {
        ROS_DEBUG("[global_graph] submap edge S(%d)<->S(%d) "
                  "already exists, skipping", sid_prev, sid_curr);
        return;
    }

    const auto pos = T_rel.block<3,1>(0,3);
    const auto [noise, sigma_t, sigma_r] = noiseFromScore(
        score,
        submap_loop_sigma_t_min_, submap_loop_sigma_t_max_,
        submap_loop_sigma_r_min_, submap_loop_sigma_r_max_);

    {
        std::lock_guard<std::mutex> lk(lock);
        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(sid_prev), X(sid_curr), toPose3(T_rel), noise);
    }
    loop_edges_added.insert(edge);

    ROS_INFO("[global_graph] ADDED BetweenFactor "
             "S(%d)->S(%d) score=%.4f t=[%.3f, %.3f, %.3f] "
             "sigma_t=%.4f sigma_r=%.4f",
             sid_prev, sid_curr, score,
             pos(0), pos(1), pos(2), sigma_t, sigma_r);

    commit(stamp);
}

// ---------------------------------------------------------------
// Keyframe-level loop factor (from registration results)
// ---------------------------------------------------------------

void GlobalPoseGraph::addLoopFactor(
        int prev_key_idx, int curr_key_idx,
        const Matrix4d& T_prev_to_curr,
        const ros::Time& stamp,
        const std::map<int, Matrix4d>& pose_by_idx,
        double score) {
    if (!enable) return;

    const auto it_prev = key_to_submap.find(prev_key_idx);
    const auto it_curr = key_to_submap.find(curr_key_idx);
    if (it_prev == key_to_submap.end() || it_curr == key_to_submap.end()) return;

    const int sid_prev = it_prev->second;
    const int sid_curr = it_curr->second;
    if (sid_prev == sid_curr) return;

    const auto edge = std::make_pair(std::min(sid_prev, sid_curr),
                                     std::max(sid_prev, sid_curr));
    if (loop_edges_added.count(edge)) return;

    Matrix4d T_rel_sub = T_prev_to_curr;

    const auto pkp = pose_by_idx.find(prev_key_idx);
    const auto pkc = pose_by_idx.find(curr_key_idx);
    const auto psp = submap_pose_by_idx.find(sid_prev);
    const auto psc = submap_pose_by_idx.find(sid_curr);

    if (pkp != pose_by_idx.end() && pkc != pose_by_idx.end() &&
        psp != submap_pose_by_idx.end() && psc != submap_pose_by_idx.end()) {
        const Matrix4d T_sp_kp = psp->second.inverse() * pkp->second;
        const Matrix4d T_kc_sc = pkc->second.inverse() * psc->second;
        T_rel_sub = T_sp_kp * T_prev_to_curr * T_kc_sc;
    }

    const double use_score = (score < 0.0) ? submap_reg_score_threshold_ : score;
    const auto [noise, sigma_t, sigma_r] = noiseFromScore(
        use_score,
        loop_sigma_t_min_, loop_sigma_t_max_,
        loop_sigma_r_min_, loop_sigma_r_max_);

    {
        std::lock_guard<std::mutex> lk(lock);
        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(sid_prev), X(sid_curr), toPose3(T_rel_sub), noise);
    }
    loop_edges_added.insert(edge);

    const auto pos = T_rel_sub.block<3,1>(0,3);
    ROS_INFO("[global_graph] ADDED BetweenFactor (keyframe loop) "
             "S(%d)->S(%d) via X(%d)->X(%d) "
             "t=[%.3f, %.3f, %.3f] score=%.4f sigma_t=%.4f sigma_r=%.4f",
             sid_prev, sid_curr, prev_key_idx, curr_key_idx,
             pos(0), pos(1), pos(2), use_score, sigma_t, sigma_r);

    commit(stamp);
}

// ---------------------------------------------------------------
// iSAM2 update
// ---------------------------------------------------------------

void GlobalPoseGraph::commit(const ros::Time& stamp) {
    if (!enable) return;

    std::lock_guard<std::mutex> lk(lock);
    try {
        isam_->update(new_factors_, new_values_);
        new_factors_.resize(0);
        new_values_.clear();
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(2.0, "[global_graph] update failed: %s", e.what());
        return;
    }

    gtsam::Values est;
    try {
        est = isam_->calculateEstimate();
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(2.0, "[global_graph] calculateEstimate failed: %s",
                          e.what());
        return;
    }

    nav_msgs::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = odom_frame_;

    for (int sid : submap_ids) {
        try {
            const Matrix4d T_sid = toMatrix(est.at<gtsam::Pose3>(X(sid)));
            submap_pose_by_idx[sid] = T_sid;
            path.poses.push_back(poseToPoseStamped(T_sid, stamp, odom_frame_));
        } catch (const gtsam::ValuesKeyDoesNotExist&) {
            continue;
        }
    }

    path_ = std::move(path);
    path_pub_.publish(path_);
}

} // namespace gmmslam
