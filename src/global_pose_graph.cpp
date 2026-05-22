#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/util/gmm_utils.hpp"
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

namespace {

void normalizeGmmWeights(GmmModel& m) {
    double s = 0.0;
    for (const auto& c : m.components) {
        s += c.weight;
    }
    if (s <= 0.0) {
        return;
    }
    for (auto& c : m.components) {
        c.weight /= s;
    }
}

/// Apply rigid transform T_src_to_dst (maps points from src frame to dst).
GmmModel transformGmmRigid(const GmmModel& in, const Matrix4d& T_src_to_dst) {
    const Matrix3d R = T_src_to_dst.block<3, 3>(0, 0);
    const Vector3d t = T_src_to_dst.block<3, 1>(0, 3);
    GmmModel out;
    out.components.reserve(in.components.size());
    for (const auto& comp : in.components) {
        GmmComponent c;
        c.mean = R * comp.mean + t;
        Matrix3d cov = R * comp.covariance * R.transpose();
        c.covariance = 0.5 * (cov + cov.transpose());
        c.weight = comp.weight;
        c.source_key_idx = comp.source_key_idx;
        c.pose_uncertainty = comp.pose_uncertainty;
        out.components.push_back(std::move(c));
    }
    normalizeGmmWeights(out);
    return out;
}

GmmComponent transformComponentRigid(const GmmComponent& comp,
                                     const Matrix4d& T_src_to_dst) {
    const Matrix3d R = T_src_to_dst.block<3, 3>(0, 0);
    const Vector3d t = T_src_to_dst.block<3, 1>(0, 3);
    GmmComponent c;
    c.mean = R * comp.mean + t;
    Matrix3d cov = R * comp.covariance * R.transpose();
    c.covariance = 0.5 * (cov + cov.transpose());
    c.weight = comp.weight;
    c.source_key_idx = comp.source_key_idx;
    c.pose_uncertainty = comp.pose_uncertainty;
    return c;
}

} // namespace

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
                                     const Matrix4d* T_ref,
                                     bool use_traj_aux_limits) const {
    if (!T_aux.allFinite()) return false;

    const double abs_t = use_traj_aux_limits ? traj_aux_gate_abs_trans_m_
                                             : aux_gate_abs_trans_m_;
    const double abs_r = use_traj_aux_limits ? traj_aux_gate_abs_rot_deg_
                                             : aux_gate_abs_rot_deg_;
    const double con_t = use_traj_aux_limits ? traj_aux_gate_consistency_trans_m_
                                             : aux_gate_consistency_trans_m_;
    const double con_r = use_traj_aux_limits ? traj_aux_gate_consistency_rot_deg_
                                             : aux_gate_consistency_rot_deg_;

    const double d_abs = T_aux.block<3,1>(0,3).norm();
    const double r_abs = rotAngleDeg(T_aux.block<3,3>(0,0));

    if (d_abs > abs_t) {
        ROS_WARN("[global_graph] rejected %s submap factor (abs trans %.3fm > %.3fm)",
                 name.c_str(), d_abs, abs_t);
        return false;
    }
    if (r_abs > abs_r) {
        ROS_WARN("[global_graph] rejected %s submap factor (abs rot %.2fdeg > %.2fdeg)",
                 name.c_str(), r_abs, abs_r);
        return false;
    }

    if (T_ref != nullptr && T_ref->allFinite()) {
        const Matrix4d T_err = T_ref->inverse() * T_aux;
        const double d_err = T_err.block<3,1>(0,3).norm();
        const double r_err = rotAngleDeg(T_err.block<3,3>(0,0));

        if (d_err > con_t) {
            ROS_WARN("[global_graph] rejected %s submap factor "
                     "(consistency trans %.3fm > %.3fm)",
                     name.c_str(), d_err, con_t);
            return false;
        }
        if (r_err > con_r) {
            ROS_WARN("[global_graph] rejected %s submap factor "
                     "(consistency rot %.2fdeg > %.2fdeg)",
                     name.c_str(), r_err, con_r);
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
        const MapConfig& map_cfg,
        const std::string& odom_frame,
        const std::string& gmm_dir,
        ros::Publisher path_pub,
        ros::Publisher reg_request_pub,
        GetPoseFn get_pose_fn,
        GetGmmFn get_gmm_fn,
        GetPoseUncertaintyFn get_pose_uncertainty_fn,
        GetSubmapTrajDeltaFn get_traj_delta_fn)
    : enable(gg_cfg.enable)
    , odom_frame_(odom_frame)
    , gmm_dir_(gmm_dir)
    , map_cfg_(map_cfg)
    , submap_keyframes_per_submap_(gg_cfg.submap_keyframes_per_submap)
    , overlap_radius_m_(gg_cfg.overlap_radius_m)
    , max_overlap_registrations_(std::max(0, gg_cfg.max_overlap_registrations))
    , submap_reg_score_threshold_(gg_cfg.reg_score_threshold)
    , min_loop_submap_gap_(std::max(0, gg_cfg.min_loop_submap_gap))
    , enable_traj_aux_factors_(gg_cfg.enable_traj_aux_factors)
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
    , keyframe_loop_consistency_trans_m_(lc_cfg.consistency_gate_trans_m)
    , keyframe_loop_consistency_rot_deg_(lc_cfg.consistency_gate_rot_deg)
    , aux_gate_abs_trans_m_(gg_cfg.aux_gate_abs_trans_m)
    , aux_gate_abs_rot_deg_(gg_cfg.aux_gate_abs_rot_deg)
    , aux_gate_consistency_trans_m_(gg_cfg.aux_gate_consistency_trans_m)
    , aux_gate_consistency_rot_deg_(gg_cfg.aux_gate_consistency_rot_deg)
    , traj_aux_gate_abs_trans_m_(gg_cfg.traj_aux_gate_abs_trans_m)
    , traj_aux_gate_abs_rot_deg_(gg_cfg.traj_aux_gate_abs_rot_deg)
    , traj_aux_gate_consistency_trans_m_(gg_cfg.traj_aux_gate_consistency_trans_m)
    , traj_aux_gate_consistency_rot_deg_(gg_cfg.traj_aux_gate_consistency_rot_deg)
    , reanchor_on_traj_fail_(gg_cfg.reanchor_smoother_on_traj_gate_fail)
    , submap_finalize_min_ready_keyframes_(std::max(0, gg_cfg.submap_finalize_min_ready_keyframes))
    , submap_finalize_min_ready_fraction_(gg_cfg.submap_finalize_min_ready_fraction)
    , submap_finalize_max_wait_s_(std::max(0.0, gg_cfg.submap_finalize_max_wait_s))
    , get_pose_(std::move(get_pose_fn))
    , get_gmm_(std::move(get_gmm_fn))
    , get_pose_uncertainty_(std::move(get_pose_uncertainty_fn))
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

bool GlobalPoseGraph::consumeSmootherReanchorRequest() {
    return smoother_reanchor_requested_.exchange(false, std::memory_order_acq_rel);
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

    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        keyframe_pose_by_idx[key_idx] = T_curr;
        if (last_submap_idx_ >= 0) {
            submap_keyframes[last_submap_idx_].push_back(key_idx);
            key_to_submap[key_idx] = last_submap_idx_;
        }
        if (!shouldCreateSubmap(key_idx)) {
            return;
        }
    }

    if (last_submap_idx_ >= 0) {
        enqueueSubmapFinalization(last_submap_idx_, stamp);
    }

    std::lock_guard<std::recursive_mutex> lk(lock);
    const int sid = static_cast<int>(submap_ids.size());
    const auto key_sid = X(sid);

    new_values_.insert(key_sid, toPose3(T_curr));

    if (sid == 0) {
        new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            key_sid, toPose3(T_curr), prior_noise_);
    } else {
        const Matrix4d rel = last_submap_pose_.inverse() * T_curr;
        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(last_submap_idx_), key_sid, toPose3(rel), between_noise_);

        if (enable_traj_aux_factors_) {
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
        const bool gate_ok = passesAuxGate("traj", T_traj, &T_ref_rel, true);
        if (!gate_ok && reanchor_on_traj_fail_) {
            const double d_abs = T_traj.block<3,1>(0,3).norm();
            const double r_abs = rotAngleDeg(T_traj.block<3,3>(0,0));
            if (d_abs > traj_aux_gate_abs_trans_m_ ||
                r_abs > traj_aux_gate_abs_rot_deg_) {
                smoother_reanchor_requested_.store(true, std::memory_order_relaxed);
            }
        }
        if (gate_ok) {
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

void GlobalPoseGraph::enqueueSubmapFinalization(int sid, const ros::Time& stamp) {
    if (!enable || !get_gmm_) return;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        if (submap_gmm.count(sid)) return;
        const auto dup = std::find_if(
            pending_submap_finalize_.begin(), pending_submap_finalize_.end(),
            [sid](const std::pair<int, ros::Time>& p) { return p.first == sid; });
        if (dup == pending_submap_finalize_.end()) {
            pending_submap_finalize_.emplace_back(sid, stamp);
        }
    }
    processPendingSubmapFinalizations(stamp);
}

void GlobalPoseGraph::processPendingSubmapFinalizations(const ros::Time& stamp) {
    if (!enable || !get_gmm_) return;

    std::lock_guard<std::mutex> fk(finalize_serialization_mu_);

    // Snapshot pending work under lock, then finalize without holding the graph
    // mutex across get_gmm_/get_pose_ (those take registration / smoother locks).
    // Holding graph+registration in opposite order to publishGmmMarkers causes
    // deadlock or hard-to-debug failure under load.
    std::vector<std::pair<int, ros::Time>> snapshot;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        snapshot = pending_submap_finalize_;
    }

    for (const auto& entry : snapshot) {
        const int sid = entry.first;
        const ros::Time since = entry.second;

        bool already_done = false;
        {
            std::lock_guard<std::recursive_mutex> lk(lock);
            if (submap_gmm.count(sid)) {
                already_done = true;
            }
        }
        if (already_done) {
            std::lock_guard<std::recursive_mutex> lk(lock);
            pending_submap_finalize_.erase(
                std::remove_if(
                    pending_submap_finalize_.begin(),
                    pending_submap_finalize_.end(),
                    [sid](const std::pair<int, ros::Time>& p) {
                        return p.first == sid;
                    }),
                pending_submap_finalize_.end());
            continue;
        }

        if (tryFinalizeSubmap(sid, stamp, since)) {
            std::lock_guard<std::recursive_mutex> lk(lock);
            pending_submap_finalize_.erase(
                std::remove_if(
                    pending_submap_finalize_.begin(),
                    pending_submap_finalize_.end(),
                    [sid](const std::pair<int, ros::Time>& p) {
                        return p.first == sid;
                    }),
                pending_submap_finalize_.end());
        }
    }
}

bool GlobalPoseGraph::tryFinalizeSubmap(int sid, const ros::Time& stamp,
                                        const ros::Time& pending_since) {
    if (!get_gmm_) return true;

    std::vector<int> key_indices;
    Matrix4d T_ref = Matrix4d::Identity();
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        if (submap_gmm.count(sid)) return true;
        const auto it_keys = submap_keyframes.find(sid);
        if (it_keys == submap_keyframes.end() || it_keys->second.empty()) {
            return true;
        }
        key_indices = it_keys->second;

        const auto it_pose = submap_pose_by_idx.find(sid);
        if (it_pose == submap_pose_by_idx.end()) return true;
        T_ref = it_pose->second;
    }

    std::vector<PosedGmmInput> gmms_with_poses;
    gmms_with_poses.reserve(key_indices.size());
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

        PosedGmmInput input;
        input.model = std::move(gmm);
        input.pose = T_kf;
        input.key_idx = ki;
        if (get_pose_uncertainty_) {
            auto opt_uncertainty = get_pose_uncertainty_(ki);
            if (opt_uncertainty.has_value() && std::isfinite(opt_uncertainty.value())) {
                input.pose_uncertainty = opt_uncertainty.value();
            }
        }
        gmms_with_poses.push_back(std::move(input));
    }

    const int n_keys = static_cast<int>(key_indices.size());
    const int n_ready = static_cast<int>(gmms_with_poses.size());

    if (n_ready == 0) {
        ROS_DEBUG_THROTTLE(
            5.0,
            "[global_graph] S(%d): submap finalize waiting for async GMM fits "
            "(%zu keyframes, none ready yet)",
            sid, key_indices.size());
        return false;
    }

    int required = 1;
    if (submap_finalize_min_ready_fraction_ > 0.0 &&
        submap_finalize_min_ready_fraction_ <= 1.0 && n_keys > 0) {
        required = std::max(
            required,
            static_cast<int>(std::ceil(submap_finalize_min_ready_fraction_ *
                                        static_cast<double>(n_keys))));
    }
    required = std::max(required, submap_finalize_min_ready_keyframes_);
    required = std::min(required, std::max(1, n_keys));

    const double wait_s = (stamp - pending_since).toSec();
    const bool wait_timed_out = submap_finalize_max_wait_s_ > 0.0 &&
                               wait_s >= submap_finalize_max_wait_s_;

    if (n_ready < required) {
        if (!wait_timed_out) {
            ROS_DEBUG_THROTTLE(
                5.0,
                "[global_graph] S(%d): submap finalize waiting for GMM fits "
                "(%d/%d keyframes ready, need %d, max_wait_s=%.2f, waited=%.2fs)",
                sid, n_ready, n_keys, required,
                submap_finalize_max_wait_s_, wait_s);
            return false;
        }
        ROS_WARN("[global_graph] S(%d): submap finalize readiness timeout "
                 "(%.2fs >= %.2fs); finalizing with %d/%d keyframe GMMs (required was %d)",
                 sid, wait_s, submap_finalize_max_wait_s_, n_ready, n_keys, required);
    }

    GmmModel raw = mergeGmmsConcatenate(gmms_with_poses, T_ref);
    // Keep full (unpruned) merged GMM for overlap D2D; internal prune runs after the
    // overlap registration wave for this submap (see maybeApplyDeferredInternalSubmapPrune).
    if (raw.components.empty()) return true;

    const std::vector<GmmLocalData> components_cache =
        precomputeGmmLocalData(raw);

    const std::string gmm_path =
        gmm_dir_ + "/submap_" + [&]{
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%04d", sid);
            return std::string(buf);
        }() + ".gmm";

    try {
        saveGmmToFile(raw, gmm_path);
    } catch (const std::exception& e) {
        ROS_WARN("[global_graph] failed to save S(%d) GMM: %s",
                 sid, e.what());
        return true;
    }

    const int raw_n = raw.numComponents();

    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        if (submap_gmm.count(sid)) return true;
        submap_gmm[sid] = std::move(raw);
        submap_gmm_components[sid] = components_cache;
        submap_frozen_pose_by_idx[sid] = T_ref;
        submap_gmm_path[sid] = gmm_path;
    }

    ROS_INFO("[global_graph] S(%d) finalized: %d components (internal prune deferred "
             "until after overlap D2D; D_B gate %.2f) from %zu/%zu keyframes",
             sid, raw_n,
             map_cfg_.prune_bhatt_threshold,
             gmms_with_poses.size(), key_indices.size());

    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        requestOverlapRegistrations(sid, stamp);
        maybeApplyDeferredInternalSubmapPrune(sid, stamp);
    }
    return true;
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

    struct OverlapCandidate {
        double distance;
        int sid_old;
        std::string path_old;
    };
    std::vector<OverlapCandidate> candidates;

    int n_published = 0;
    for (int sid_old : submap_ids) {
        if (sid_old >= sid_new) continue;
        if (sid_old == sid_new - 1) continue;
        if ((sid_new - sid_old) < min_loop_submap_gap_) continue;

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

        candidates.push_back({d, sid_old, it_path_old->second});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const OverlapCandidate& a, const OverlapCandidate& b) {
                  if (a.distance == b.distance) {
                      return a.sid_old > b.sid_old;
                  }
                  return a.distance < b.distance;
              });

    const int n_to_publish = max_overlap_registrations_ > 0
        ? std::min(max_overlap_registrations_,
                   static_cast<int>(candidates.size()))
        : static_cast<int>(candidates.size());

    if (static_cast<int>(candidates.size()) > n_to_publish) {
        ROS_INFO("[global_graph] S(%d): overlap candidates %zu, publishing closest %d",
                 sid_new, candidates.size(), n_to_publish);
    }

    for (int i = 0; i < n_to_publish; ++i) {
        const auto& cand = candidates[static_cast<std::size_t>(i)];
        const int sid_old = cand.sid_old;
        const auto edge = std::make_pair(std::min(sid_old, sid_new),
                                         std::max(sid_old, sid_new));

        nlohmann::json payload;
        payload["prev_idx"]                 = sid_old;
        payload["curr_idx"]                 = sid_new;
        payload["stamp"]                    = stampToSec(stamp);
        payload["source_path"]              = new_path;
        payload["target_path"]              = cand.path_old;
        payload["is_loop_closure"]          = false;
        payload["is_submap_registration"]   = true;

        const auto it_old_pose = submap_pose_by_idx.find(sid_old);
        if (it_old_pose != submap_pose_by_idx.end()) {
            const Matrix4d T_init = it_old_pose->second.inverse() * it_new->second;
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row(T_init);
            std::vector<double> flat(16);
            for (int k = 0; k < 16; ++k) {
                flat[static_cast<std::size_t>(k)] = T_row.data()[k];
            }
            payload["initial_transform"] = std::move(flat);
        }

        std_msgs::String msg;
        msg.data = payload.dump();
        reg_request_pub_.publish(msg);

        pending_submap_registrations_.insert(edge);
        ++n_published;
        ROS_INFO("[global_graph] requested registration "
                 "S(%d)<->S(%d) (d=%.2fm)", sid_old, sid_new, cand.distance);
    }

    submap_overlap_d2d_remaining_[sid_new] = n_published;
    if (n_published == 0 && map_cfg_.prune_enable) {
        submap_needs_internal_prune_after_overlap_wave_.insert(sid_new);
    }
}

void GlobalPoseGraph::acknowledgeSubmapOverlapAttempt(int sid_prev, int sid_curr,
                                                      const ros::Time& /*stamp*/) {
    if (!enable) return;

    const auto edge = std::make_pair(std::min(sid_prev, sid_curr),
                                     std::max(sid_prev, sid_curr));
    const int sid_new = std::max(sid_prev, sid_curr);

    std::lock_guard<std::recursive_mutex> lk(lock);
    pending_submap_registrations_.erase(edge);

    auto it = submap_overlap_d2d_remaining_.find(sid_new);
    if (it == submap_overlap_d2d_remaining_.end()) {
        return;
    }
    if (it->second > 0) {
        --(it->second);
    }
    if (it->second <= 0) {
        submap_overlap_d2d_remaining_.erase(it);
        if (map_cfg_.prune_enable) {
            submap_needs_internal_prune_after_overlap_wave_.insert(sid_new);
        }
    }
}

void GlobalPoseGraph::maybeApplyDeferredInternalSubmapPrune(int sid_new,
                                                            const ros::Time& /*stamp*/) {
    if (!enable) return;

    GmmModel input;
    std::string path;
    int before_n = 0;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        if (!map_cfg_.prune_enable) {
            submap_needs_internal_prune_after_overlap_wave_.erase(sid_new);
            return;
        }

        // Atomically claim the deferred prune marker. The previous implementation
        // read this set before locking while result/finalize callbacks mutated it,
        // which is undefined behavior and can segfault under heavy submap D2D load.
        if (!submap_needs_internal_prune_after_overlap_wave_.erase(sid_new)) {
            return;
        }

        auto git = submap_gmm.find(sid_new);
        if (git == submap_gmm.end() || git->second.components.empty()) {
            return;
        }
        input = git->second;
        before_n = input.numComponents();

        const auto pit = submap_gmm_path.find(sid_new);
        if (pit != submap_gmm_path.end()) {
            path = pit->second;
        }
    }

    GmmModel pruned = pruneNewerFrameComponents(input, map_cfg_);
    if (pruned.components.empty()) {
        ROS_WARN("[global_graph] deferred internal prune S(%d) would remove "
                 "all components; keeping unpruned map", sid_new);
        return;
    }

    GmmModel to_save;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        const auto it = submap_gmm.find(sid_new);
        if (it == submap_gmm.end()) {
            return;
        }
        if (it->second.numComponents() != before_n) {
            ROS_WARN_THROTTLE(
                5.0,
                "[global_graph] deferred internal prune S(%d): submap changed "
                "during prune (%d -> %d comp); skipping apply",
                sid_new, before_n, it->second.numComponents());
            return;
        }
        it->second = std::move(pruned);
        submap_gmm_components[sid_new] = precomputeGmmLocalData(it->second);
        to_save = it->second;

        ROS_INFO("[global_graph] deferred internal prune S(%d): %d -> %d components "
                 "(after overlap D2D wave)",
                 sid_new, before_n, it->second.numComponents());
    }

    if (!path.empty()) {
        try {
            saveGmmToFile(to_save, path);
        } catch (const std::exception& e) {
            ROS_WARN("[global_graph] deferred internal prune: failed to save S(%d): %s",
                     sid_new, e.what());
        }
    }
}

void GlobalPoseGraph::pruneSubmapPairGmmsAfterLoop(
        int sid_a, int sid_b, const ros::Time& stamp) {
    if (!map_cfg_.prune_enable) return;
    if (sid_a == sid_b) return;

    const int s_lo = std::min(sid_a, sid_b);
    const int s_hi = std::max(sid_a, sid_b);

    GmmModel ga_copy, gb_copy;
    Matrix4d Ta = Matrix4d::Identity();
    Matrix4d Tb = Matrix4d::Identity();
    std::map<int, int> key_sid_copy;
    std::string path_a, path_b;
    bool have_a_path = false;
    bool have_b_path = false;

    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        const auto ga_it = submap_gmm.find(sid_a);
        const auto gb_it = submap_gmm.find(sid_b);
        if (ga_it == submap_gmm.end() || gb_it == submap_gmm.end()) {
            return;
        }
        if (ga_it->second.components.empty() ||
            gb_it->second.components.empty()) {
            return;
        }
        ga_copy = ga_it->second;
        gb_copy = gb_it->second;

        const auto pa = submap_pose_by_idx.find(sid_a);
        const auto pb = submap_pose_by_idx.find(sid_b);
        if (pa != submap_pose_by_idx.end() && pa->second.allFinite()) {
            Ta = pa->second;
        }
        if (pb != submap_pose_by_idx.end() && pb->second.allFinite()) {
            Tb = pb->second;
        }

        key_sid_copy = key_to_submap;

        const auto path_it_a = submap_gmm_path.find(sid_a);
        if (path_it_a != submap_gmm_path.end()) {
            path_a = path_it_a->second;
            have_a_path = true;
        }
        const auto path_it_b = submap_gmm_path.find(sid_b);
        if (path_it_b != submap_gmm_path.end()) {
            path_b = path_it_b->second;
            have_b_path = true;
        }
    }

    const int snap_na = ga_copy.numComponents();
    const int snap_nb = gb_copy.numComponents();

    GmmModel world_a = transformGmmRigid(ga_copy, Ta);
    GmmModel world_b = transformGmmRigid(gb_copy, Tb);

    GmmModel combined;
    combined.components.reserve(world_a.components.size() +
                                world_b.components.size());
    combined.components.insert(
        combined.components.end(),
        std::make_move_iterator(world_a.components.begin()),
        std::make_move_iterator(world_a.components.end()));
    combined.components.insert(
        combined.components.end(),
        std::make_move_iterator(world_b.components.begin()),
        std::make_move_iterator(world_b.components.end()));
    normalizeGmmWeights(combined);

    const GmmModel pruned = pruneNewerFrameComponents(combined, map_cfg_);
    if (pruned.components.empty()) {
        ROS_WARN("[global_graph] cross-submap prune S(%d)<->S(%d): "
                 "prune removed all components; keeping originals",
                 s_lo, s_hi);
        return;
    }

    GmmModel new_a, new_b;
    const Matrix4d invTa = Ta.inverse();
    const Matrix4d invTb = Tb.inverse();
    int dropped_keys = 0;
    for (const auto& wc : pruned.components) {
        const int kid = wc.source_key_idx;
        const auto it = key_sid_copy.find(kid);
        if (it == key_sid_copy.end()) {
            ++dropped_keys;
            continue;
        }
        const int owner = it->second;
        if (owner == sid_a) {
            new_a.components.push_back(transformComponentRigid(wc, invTa));
        } else if (owner == sid_b) {
            new_b.components.push_back(transformComponentRigid(wc, invTb));
        } else {
            ++dropped_keys;
        }
    }

    normalizeGmmWeights(new_a);
    normalizeGmmWeights(new_b);

    if (new_a.components.empty() || new_b.components.empty()) {
        ROS_WARN("[global_graph] cross-submap prune S(%d)<->S(%d): "
                 "split would empty a submap (A=%d B=%d dropped_key=%d); "
                 "keeping originals",
                 s_lo, s_hi,
                 static_cast<int>(new_a.components.size()),
                 static_cast<int>(new_b.components.size()),
                 dropped_keys);
        return;
    }

    const int na_before = ga_copy.numComponents();
    const int nb_before = gb_copy.numComponents();
    const int na_after = new_a.numComponents();
    const int nb_after = new_b.numComponents();

    GmmModel to_save_a, to_save_b;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);
        if (!submap_gmm.count(sid_a) || !submap_gmm.count(sid_b)) {
            return;
        }
        if (submap_gmm.at(sid_a).numComponents() != snap_na ||
            submap_gmm.at(sid_b).numComponents() != snap_nb) {
            ROS_WARN_THROTTLE(
                5.0,
                "[global_graph] cross-submap prune S(%d)<->S(%d): "
                "submap GMM changed during prune; skipping apply",
                s_lo, s_hi);
            return;
        }
        submap_gmm[sid_a] = std::move(new_a);
        submap_gmm[sid_b] = std::move(new_b);
        submap_gmm_components[sid_a] = precomputeGmmLocalData(submap_gmm[sid_a]);
        submap_gmm_components[sid_b] = precomputeGmmLocalData(submap_gmm[sid_b]);
        to_save_a = submap_gmm[sid_a];
        to_save_b = submap_gmm[sid_b];
    }

    ROS_INFO("[global_graph] cross-submap prune after loop S(%d)<->S(%d): "
             "S(%d) %d->%d comp, S(%d) %d->%d comp (dropped_key=%d)",
             s_lo, s_hi,
             sid_a, na_before, na_after,
             sid_b, nb_before, nb_after,
             dropped_keys);

    try {
        if (have_a_path) {
            saveGmmToFile(to_save_a, path_a);
        }
    } catch (const std::exception& e) {
        ROS_WARN("[global_graph] cross-submap prune: failed to save S(%d): %s",
                 sid_a, e.what());
    }
    try {
        if (have_b_path) {
            saveGmmToFile(to_save_b, path_b);
        }
    } catch (const std::exception& e) {
        ROS_WARN("[global_graph] cross-submap prune: failed to save S(%d): %s",
                 sid_b, e.what());
    }
}

// ---------------------------------------------------------------
// Submap registration result
// ---------------------------------------------------------------

void GlobalPoseGraph::handleSubmapRegistrationResult(
        int sid_prev, int sid_curr,
        const Matrix4d& T_rel, double score,
        const ros::Time& stamp) {
    if (!enable) return;

    const int sid_new = std::max(sid_prev, sid_curr);
    struct RunDeferredPrune {
        GlobalPoseGraph* graph = nullptr;
        int sid = -1;
        ros::Time st;
        ~RunDeferredPrune() {
            if (graph != nullptr) {
                graph->maybeApplyDeferredInternalSubmapPrune(sid, st);
            }
        }
    } defer{this, sid_new, stamp};

    const auto edge = std::make_pair(std::min(sid_prev, sid_curr),
                                     std::max(sid_prev, sid_curr));

    {
        std::lock_guard<std::recursive_mutex> lk(lock);

        if (score < submap_reg_score_threshold_) {
            ROS_INFO("[global_graph] submap reg S(%d)->S(%d) "
                     "rejected (score=%.4f < %.4f)",
                     sid_prev, sid_curr, score, submap_reg_score_threshold_);
            return;
        }

        const int submap_gap = std::abs(sid_curr - sid_prev);
        if (submap_gap < min_loop_submap_gap_) {
            ROS_INFO("[global_graph] submap reg S(%d)->S(%d) rejected "
                     "(submap gap %d < min_loop_submap_gap %d)",
                     sid_prev, sid_curr, submap_gap, min_loop_submap_gap_);
            return;
        }

        if (loop_edges_added.count(edge)) {
            ROS_INFO("[global_graph] submap reg S(%d)->S(%d) rejected "
                     "(edge already exists)",
                     sid_prev, sid_curr);
            return;
        }

        if (!T_rel.allFinite()) {
            ROS_WARN("[global_graph] submap reg S(%d)->S(%d) rejected (non-finite T)",
                     sid_prev, sid_curr);
            return;
        }

        // Absolute + consistency gates for overlap D2D (tight); reject outliers
        // vs current submap anchor geometry. Trajectory chain uses traj_aux_gate_*.
        {
            const auto psp = submap_pose_by_idx.find(sid_prev);
            const auto psc = submap_pose_by_idx.find(sid_curr);
            if (psp != submap_pose_by_idx.end() && psc != submap_pose_by_idx.end()) {
                const Matrix4d T_ref_rel = psp->second.inverse() * psc->second;
                if (!passesAuxGate("overlap_reg", T_rel, &T_ref_rel, false)) {
                    ROS_INFO("[global_graph] submap reg S(%d)->S(%d) "
                             "rejected (aux gate vs anchor poses)",
                             sid_prev, sid_curr);
                    return;
                }
            } else {
                if (!passesAuxGate("overlap_reg", T_rel, nullptr, false)) {
                    ROS_INFO("[global_graph] submap reg S(%d)->S(%d) "
                             "rejected (aux magnitude gate; missing anchor pose)",
                             sid_prev, sid_curr);
                    return;
                }
            }
        }

        const auto pos = T_rel.block<3,1>(0,3);
        const auto [noise, sigma_t, sigma_r] = noiseFromScore(
            score,
            submap_loop_sigma_t_min_, submap_loop_sigma_t_max_,
            submap_loop_sigma_r_min_, submap_loop_sigma_r_max_);

        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(sid_prev), X(sid_curr), toPose3(T_rel), noise);
        loop_edges_added.insert(edge);

        ROS_INFO("[global_graph] ADDED BetweenFactor "
                 "S(%d)->S(%d) score=%.4f t=[%.3f, %.3f, %.3f] "
                 "sigma_t=%.4f sigma_r=%.4f",
                 sid_prev, sid_curr, score,
                 pos(0), pos(1), pos(2), sigma_t, sigma_r);
    }

    commit(stamp);
    pruneSubmapPairGmmsAfterLoop(sid_prev, sid_curr, stamp);
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

    int sid_prev = -1;
    int sid_curr = -1;
    {
        std::lock_guard<std::recursive_mutex> lk(lock);

        const auto it_prev = key_to_submap.find(prev_key_idx);
        const auto it_curr = key_to_submap.find(curr_key_idx);
        if (it_prev == key_to_submap.end() || it_curr == key_to_submap.end()) {
            ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) rejected "
                     "(missing key_to_submap: prev=%s curr=%s)",
                     prev_key_idx, curr_key_idx,
                     it_prev == key_to_submap.end() ? "missing" : "ok",
                     it_curr == key_to_submap.end() ? "missing" : "ok");
            return;
        }

        sid_prev = it_prev->second;
        sid_curr = it_curr->second;
        if (sid_prev == sid_curr) {
            ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) rejected "
                     "(same submap S(%d))",
                     prev_key_idx, curr_key_idx, sid_prev);
            return;
        }

        const int submap_gap = std::abs(sid_curr - sid_prev);
        if (submap_gap < min_loop_submap_gap_) {
            ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) rejected "
                     "(submap gap S(%d)->S(%d) is %d < min_loop_submap_gap %d)",
                     prev_key_idx, curr_key_idx, sid_prev, sid_curr,
                     submap_gap, min_loop_submap_gap_);
            return;
        }

        const auto edge = std::make_pair(std::min(sid_prev, sid_curr),
                                         std::max(sid_prev, sid_curr));
        if (loop_edges_added.count(edge)) {
            ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) rejected "
                     "(submap edge S(%d)<->S(%d) already exists)",
                     prev_key_idx, curr_key_idx, edge.first, edge.second);
            return;
        }

        Matrix4d T_rel_sub = T_prev_to_curr;

        const auto pkp = pose_by_idx.find(prev_key_idx);
        const auto pkc = pose_by_idx.find(curr_key_idx);
        const auto gkp = keyframe_pose_by_idx.find(prev_key_idx);
        const auto gkc = keyframe_pose_by_idx.find(curr_key_idx);
        const auto psp = submap_pose_by_idx.find(sid_prev);
        const auto psc = submap_pose_by_idx.find(sid_curr);

        const Matrix4d* T_w_prev_key = nullptr;
        const Matrix4d* T_w_curr_key = nullptr;
        if (pkp != pose_by_idx.end()) {
            T_w_prev_key = &pkp->second;
        } else if (gkp != keyframe_pose_by_idx.end()) {
            T_w_prev_key = &gkp->second;
        }
        if (pkc != pose_by_idx.end()) {
            T_w_curr_key = &pkc->second;
        } else if (gkc != keyframe_pose_by_idx.end()) {
            T_w_curr_key = &gkc->second;
        }

        if (T_w_prev_key != nullptr && T_w_curr_key != nullptr &&
            psp != submap_pose_by_idx.end() && psc != submap_pose_by_idx.end()) {
            const Matrix4d T_sp_kp = psp->second.inverse() * *T_w_prev_key;
            const Matrix4d T_kc_sc = T_w_curr_key->inverse() * psc->second;
            T_rel_sub = T_sp_kp * T_prev_to_curr * T_kc_sc;
        } else {
            ROS_WARN("[global_graph] keyframe loop X(%d)->X(%d): missing "
                     "keyframe/submap pose context, using raw relative transform",
                     prev_key_idx, curr_key_idx);
        }

        if (!T_rel_sub.allFinite()) {
            ROS_WARN("[global_graph] keyframe loop X(%d)->X(%d) rejected (non-finite T)",
                     prev_key_idx, curr_key_idx);
            return;
        }

        if (psp != submap_pose_by_idx.end() && psc != submap_pose_by_idx.end()) {
            const Matrix4d T_ref_rel = psp->second.inverse() * psc->second;
            const Matrix4d T_err = T_ref_rel.inverse() * T_rel_sub;
            const double trans_err = T_err.block<3, 1>(0, 3).norm();
            const double rot_err_deg = rotAngleDeg(T_err.block<3, 3>(0, 0));
            if ((keyframe_loop_consistency_trans_m_ > 0.0 &&
                 trans_err > keyframe_loop_consistency_trans_m_) ||
                (keyframe_loop_consistency_rot_deg_ > 0.0 &&
                 rot_err_deg > keyframe_loop_consistency_rot_deg_)) {
                ROS_WARN("[global_graph] rejected keyframe_loop submap factor "
                         "(consistency trans %.3fm rot %.2fdeg > %.3fm %.2fdeg)",
                         trans_err, rot_err_deg,
                         keyframe_loop_consistency_trans_m_,
                         keyframe_loop_consistency_rot_deg_);
                ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) "
                         "rejected (aux gate vs submap poses)",
                         prev_key_idx, curr_key_idx);
                return;
            }
        } else {
            ROS_INFO("[global_graph] keyframe loop X(%d)->X(%d) "
                     "rejected (missing submap pose context)",
                     prev_key_idx, curr_key_idx);
            return;
        }

        const double use_score = (score < 0.0) ? submap_reg_score_threshold_ : score;
        const auto [noise, sigma_t, sigma_r] = noiseFromScore(
            use_score,
            submap_loop_sigma_t_min_, submap_loop_sigma_t_max_,
            submap_loop_sigma_r_min_, submap_loop_sigma_r_max_);

        new_factors_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(sid_prev), X(sid_curr), toPose3(T_rel_sub), noise);
        loop_edges_added.insert(edge);

        const auto pos = T_rel_sub.block<3,1>(0,3);
        ROS_INFO("[global_graph] ADDED BetweenFactor (keyframe loop) "
                 "S(%d)->S(%d) via X(%d)->X(%d) "
                 "t=[%.3f, %.3f, %.3f] score=%.4f sigma_t=%.4f sigma_r=%.4f",
                 sid_prev, sid_curr, prev_key_idx, curr_key_idx,
                 pos(0), pos(1), pos(2), use_score, sigma_t, sigma_r);
    }

    commit(stamp);
    pruneSubmapPairGmmsAfterLoop(sid_prev, sid_curr, stamp);
}

// ---------------------------------------------------------------
// iSAM2 update
// ---------------------------------------------------------------

void GlobalPoseGraph::commit(const ros::Time& stamp) {
    if (!enable) return;

    std::lock_guard<std::recursive_mutex> lk(lock);
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
            // Keep the finalized (render-time) pose tracking iSAM2.
            // Non-finalized submaps never had an entry here and the
            // visualiser falls back to submap_pose_by_idx for them,
            // so we only refresh sids that are already frozen.
            auto fz_it = submap_frozen_pose_by_idx.find(sid);
            if (fz_it != submap_frozen_pose_by_idx.end()) {
                fz_it->second = T_sid;
            }
            path.poses.push_back(poseToPoseStamped(T_sid, stamp, odom_frame_));
        } catch (const gtsam::ValuesKeyDoesNotExist&) {
            continue;
        }
    }

    path_ = std::move(path);
    path_pub_.publish(path_);
}

} // namespace gmmslam
