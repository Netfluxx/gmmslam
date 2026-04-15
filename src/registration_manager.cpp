#include "gmmslam/registration_manager.hpp"
#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/sogmm_fitting.hpp"
#include "gmmslam/gmm_utils.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include <ros/ros.h>
#include <std_msgs/String.h>

using gtsam::symbol_shorthand::X;

namespace gmmslam {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

RegistrationManager::RegistrationManager(
    FixedLagBackend& smoother,
    GlobalPoseGraph* global_graph,
    ros::Publisher reg_request_pub,
    const RegistrationConfig& reg_cfg,
    const LoopClosureConfig& lc_cfg,
    const SogmmConfig& sogmm_cfg,
    const std::string& gmm_dir)
    : smoother_(smoother),
      global_graph_(global_graph),
      reg_pub_(std::move(reg_request_pub)),
      sogmm_cfg_(sogmm_cfg),
      gmm_dir_(gmm_dir),
      registration_score_threshold_(reg_cfg.score_threshold),
      registration_factor_every_n_frames_(reg_cfg.factor_every_n_frames),
      loop_closure_score_threshold_(reg_cfg.strong_factor_score_threshold),
      loop_closure_detect_score_threshold_(lc_cfg.detect_score_threshold),
      enable_loop_closure_(lc_cfg.enable),
      loop_closure_min_keyframe_gap_(lc_cfg.min_keyframe_gap),
      loop_closure_max_candidates_(lc_cfg.max_candidates),
      loop_closure_search_radius_m_(lc_cfg.search_radius_m),
      loop_closure_search_cooldown_keyframes_(lc_cfg.search_cooldown_keyframes),
      loop_closure_request_every_n_keyframes_(lc_cfg.request_every_n_keyframes),
      loop_closure_min_separation_m_(lc_cfg.min_separation_m),
      loop_closure_min_separation_deg_(lc_cfg.min_separation_deg),
      loop_closure_max_age_s_(lc_cfg.max_age_s),
      loop_closure_gmm_keep_keyframes_(lc_cfg.gmm_keep_keyframes),
      score_sigma_low_(reg_cfg.score_sigma_low),
      score_sigma_high_(reg_cfg.score_sigma_high),
      seq_sigma_t_min_(reg_cfg.seq_sigma_t_min),
      seq_sigma_t_max_(reg_cfg.seq_sigma_t_max),
      seq_sigma_r_min_(reg_cfg.seq_sigma_r_min),
      seq_sigma_r_max_(reg_cfg.seq_sigma_r_max),
      loop_sigma_t_min_(reg_cfg.loop_sigma_t_min),
      loop_sigma_t_max_(reg_cfg.loop_sigma_t_max),
      loop_sigma_r_min_(reg_cfg.loop_sigma_r_min),
      loop_sigma_r_max_(reg_cfg.loop_sigma_r_max),
      compensate_fit_latency_(reg_cfg.compensate_fit_latency),
      fit_queue_(static_cast<std::size_t>(reg_cfg.queue_size)),
      result_queue_(static_cast<std::size_t>(reg_cfg.result_queue_size))
{
    std::filesystem::create_directories(gmm_dir_);
}

// ---------------------------------------------------------------------------
// noiseFromScore — linear interpolation between sigma bounds
// ---------------------------------------------------------------------------

RegistrationManager::NoiseResult
RegistrationManager::noiseFromScore(double score,
                                     double sigma_t_min, double sigma_t_max,
                                     double sigma_r_min, double sigma_r_max) const
{
    double s_lo = score_sigma_low_;
    double s_hi = score_sigma_high_;
    if (s_hi <= s_lo) {
        s_hi = s_lo + 1e-6;
    }
    double alpha = (score - s_lo) / (s_hi - s_lo);
    alpha = std::clamp(alpha, 0.0, 1.0);

    const double sigma_t = sigma_t_max - alpha * (sigma_t_max - sigma_t_min);
    const double sigma_r = sigma_r_max - alpha * (sigma_r_max - sigma_r_min);

    gtsam::Vector6 sigmas;
    sigmas << sigma_r, sigma_r, sigma_r, sigma_t, sigma_t, sigma_t;

    return {gtsam::noiseModel::Diagonal::Sigmas(sigmas), sigma_t, sigma_r};
}

// ---------------------------------------------------------------------------
// enqueueFit — backpressure: drop oldest if queue full
// ---------------------------------------------------------------------------

bool RegistrationManager::enqueueFit(int frame_idx, const ros::Time& stamp,
                                      const Eigen::MatrixXf& pts,
                                      double capture_t_sec,
                                      const Matrix4d& capture_pose)
{
    FitJob job{frame_idx, stamp, pts, capture_t_sec, capture_pose};

    if (fit_queue_.tryPush(std::move(job))) {
        return true;
    }

    ++dropped_fit_frames;
    // Drop the oldest entry and try again
    fit_queue_.tryPop();
    FitJob retry{frame_idx, stamp, pts, capture_t_sec, capture_pose};
    return fit_queue_.tryPush(std::move(retry));
}

// ---------------------------------------------------------------------------
// fitWorkerLoop — background thread
// ---------------------------------------------------------------------------

void RegistrationManager::fitWorkerLoop(const std::atomic<bool>& shutdown)
{
    while (!shutdown.load(std::memory_order_relaxed)) {
        auto maybe_job = fit_queue_.popWithTimeout(std::chrono::milliseconds(50));
        if (!maybe_job) {
            continue;
        }

        FitJob& job = *maybe_job;
        try {
            GmmModel model = fitSogmm(job.points, sogmm_cfg_);
            if (model.numComponents() > 0) {
                finishFit(model, job.frame_idx, job.stamp,
                          job.capture_t_sec, job.capture_pose);
            } else {
                ROS_WARN_THROTTLE(5.0,
                    "[registration] SOGMM fit returned empty model for frame %d",
                    job.frame_idx);
            }
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0,
                "[registration] fit error for frame %d: %s",
                job.frame_idx, e.what());
        }
    }
}

// ---------------------------------------------------------------------------
// finishFit — post-fit bookkeeping, save .gmm, emit registration requests
// ---------------------------------------------------------------------------

void RegistrationManager::finishFit(const GmmModel& model, int frame_idx,
                                     const ros::Time& stamp,
                                     double capture_t_sec,
                                     const Matrix4d& capture_pose)
{
    {
        std::lock_guard<std::mutex> lk(lock);
        latest_gmm_idx = frame_idx;
        latest_gmm_model = model;
        has_latest_gmm = true;

        LocalGmmEntry entry;
        entry.stamp_sec = stamp.toSec();
        entry.model = model;
        entry.capture_t_sec = capture_t_sec;
        entry.capture_pose = capture_pose;
        entry.has_capture_pose = true;
        local_gmms_by_idx[frame_idx] = std::move(entry);

        // Purge very old entries (keep last ~400)
        const int stale_thresh = frame_idx - 400;
        for (auto it = local_gmms_by_idx.begin(); it != local_gmms_by_idx.end();) {
            if (it->first < stale_thresh) {
                it = local_gmms_by_idx.erase(it);
            } else {
                break; // map is ordered
            }
        }
    }

    ROS_INFO("[registration] frame %4d | local GMM: %3d components",
             frame_idx, model.numComponents());

    // Save .gmm file
    char buf[64];
    std::snprintf(buf, sizeof(buf), "frame_%06d.gmm", frame_idx);
    const std::string gmm_path = gmm_dir_ + "/" + buf;
    try {
        saveGmmToFile(model, gmm_path);
    } catch (const std::exception& e) {
        ROS_ERROR("[registration] failed to save GMM: %s", e.what());
        return;
    }

    const double fit_t_sec = ros::Time::now().toSec();

    // Retrieve the best available pose from the smoother.
    // When compensate_fit_latency is on, the smoother may have a more
    // accurate estimate of this frame's pose than the prediction at
    // capture time (the robot kept moving during the SOGMM fit).
    Matrix4d map_pose = capture_pose;
    Matrix4d effective_capture_pose = capture_pose;
    if (compensate_fit_latency_) {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        auto it = smoother_.pose_by_idx.find(frame_idx);
        if (it != smoother_.pose_by_idx.end()) {
            map_pose = it->second;
            effective_capture_pose = it->second;
        }
    }

    // Update entry with map pose and fit time
    int prev_idx = -1;
    std::string prev_path;
    {
        std::lock_guard<std::mutex> lk(lock);
        gmm_paths_by_idx[frame_idx] = gmm_path;

        auto entry_it = local_gmms_by_idx.find(frame_idx);
        if (entry_it != local_gmms_by_idx.end()) {
            entry_it->second.map_pose = map_pose;
            entry_it->second.has_map_pose = true;
            entry_it->second.fit_t_sec = fit_t_sec;
            entry_it->second.capture_pose = effective_capture_pose;
        }

        // Find the immediately preceding GMM
        auto curr_it = gmm_paths_by_idx.find(frame_idx);
        if (curr_it != gmm_paths_by_idx.begin()) {
            --curr_it;
            prev_idx = curr_it->first;
            prev_path = curr_it->second;
        }
    }

    // Publish sequential D2D request
    if (prev_idx >= 0 && !prev_path.empty()) {
        nlohmann::json payload;
        payload["prev_idx"]         = prev_idx;
        payload["curr_idx"]         = frame_idx;
        payload["stamp"]            = stamp.toSec();
        payload["source_path"]      = gmm_path;
        payload["target_path"]      = prev_path;
        payload["is_loop_closure"]  = false;

        std_msgs::String msg;
        msg.data = payload.dump();
        reg_pub_.publish(msg);

        enqueueLoopClosureRequests(frame_idx, stamp, gmm_path, prev_idx);
    }

    // Purge old gmm_paths beyond keep window
    {
        std::lock_guard<std::mutex> lk(lock);
        const int min_keep = std::max(0, frame_idx - loop_closure_gmm_keep_keyframes_);
        for (auto it = gmm_paths_by_idx.begin(); it != gmm_paths_by_idx.end();) {
            if (it->first < min_keep) {
                it = gmm_paths_by_idx.erase(it);
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// enqueueLoopClosureRequests
// ---------------------------------------------------------------------------

void RegistrationManager::enqueueLoopClosureRequests(
    int curr_idx, const ros::Time& stamp,
    const std::string& source_path, int sequential_prev_idx)
{
    if (!enable_loop_closure_) return;
    if (curr_idx < loop_closure_min_keyframe_gap_) return;
    if ((curr_idx % loop_closure_request_every_n_keyframes_) != 0) return;
    if ((curr_idx - last_loop_search_idx_) < loop_closure_search_cooldown_keyframes_) return;
    last_loop_search_idx_ = curr_idx;

    Vector3d curr_pos;
    double t_curr = 0.0;
    std::map<int, Vector3d> position_snapshot;
    std::map<int, double> time_snapshot;
    {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        auto it = smoother_.pose_by_idx.find(curr_idx);
        if (it == smoother_.pose_by_idx.end()) {
            ROS_WARN_THROTTLE(5.0,
                "[registration] loop search: no pose for X(%d)", curr_idx);
            return;
        }
        curr_pos = it->second.block<3, 1>(0, 3);

        auto t_it = smoother_.key_t_sec.find(curr_idx);
        if (t_it == smoother_.key_t_sec.end()) return;
        t_curr = t_it->second;

        for (const auto& [idx, T] : smoother_.pose_by_idx) {
            position_snapshot.emplace(idx, T.block<3, 1>(0, 3));
        }
        time_snapshot = smoother_.key_t_sec;
    }

    std::map<int, std::string> gmm_snapshot;
    std::set<std::pair<int,int>> pending_snapshot;
    std::set<std::pair<int,int>> added_snapshot;
    {
        std::lock_guard<std::mutex> lk(lock);
        gmm_snapshot = gmm_paths_by_idx;
        pending_snapshot = pending_loop_requests_;
        added_snapshot = loop_edges_added;
    }

    struct Candidate {
        double distance;
        int idx;
        std::string path;
    };
    std::vector<Candidate> near;

    for (const auto& [idx, target_path] : gmm_snapshot) {
        if (idx >= curr_idx) continue;
        if (curr_idx - idx < loop_closure_min_keyframe_gap_) continue;
        if (sequential_prev_idx >= 0 && idx == sequential_prev_idx) continue;

        auto t_it = time_snapshot.find(idx);
        if (t_it == time_snapshot.end()) continue;
        if ((t_curr - t_it->second) > loop_closure_max_age_s_) continue;

        auto pos_it = position_snapshot.find(idx);
        if (pos_it == position_snapshot.end()) continue;

        const double d = (pos_it->second - curr_pos).norm();
        if (d > loop_closure_search_radius_m_) continue;

        const auto edge = std::make_pair(std::min(idx, curr_idx),
                                         std::max(idx, curr_idx));
        if (pending_snapshot.count(edge) || added_snapshot.count(edge)) continue;

        near.push_back({d, idx, target_path});
    }

    if (near.empty()) {
        ROS_INFO_THROTTLE(10.0,
            "[registration] loop search @key %d: 0 candidates within %.1fm "
            "(GMMs=%zu, gap>=%d)",
            curr_idx, loop_closure_search_radius_m_,
            gmm_snapshot.size(), loop_closure_min_keyframe_gap_);
        return;
    }

    std::sort(near.begin(), near.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.distance < b.distance;
              });

    // Greedy spatial diversity: select candidates that are at least
    // min_separation_m apart from each previously selected candidate.
    std::vector<Candidate> selected;
    selected.reserve(static_cast<std::size_t>(loop_closure_max_candidates_));
    for (const auto& cand : near) {
        if (static_cast<int>(selected.size()) >= loop_closure_max_candidates_) break;

        bool too_close = false;
        auto cand_pos = position_snapshot.find(cand.idx);
        if (cand_pos != position_snapshot.end()) {
            for (const auto& prev : selected) {
                auto prev_pos = position_snapshot.find(prev.idx);
                if (prev_pos != position_snapshot.end() &&
                    (cand_pos->second - prev_pos->second).norm()
                        < loop_closure_min_separation_m_) {
                    too_close = true;
                    break;
                }
            }
        }
        if (!too_close) {
            selected.push_back(cand);
        }
    }

    const int n_selected = static_cast<int>(selected.size());
    ROS_INFO("[registration] loop search @key %d: %zu candidates, dispatching %d",
             curr_idx, near.size(), n_selected);

    for (int i = 0; i < n_selected; ++i) {
        const auto& cand = selected[static_cast<std::size_t>(i)];
        nlohmann::json payload;
        payload["prev_idx"]        = cand.idx;
        payload["curr_idx"]        = curr_idx;
        payload["stamp"]           = stamp.toSec();
        payload["source_path"]     = source_path;
        payload["target_path"]     = cand.path;
        payload["is_loop_closure"] = true;

        std_msgs::String msg;
        msg.data = payload.dump();
        reg_pub_.publish(msg);

        {
            std::lock_guard<std::mutex> lk(lock);
            pending_loop_requests_.emplace(
                std::min(cand.idx, curr_idx),
                std::max(cand.idx, curr_idx));
        }
    }
}

// ---------------------------------------------------------------------------
// resultCallback — ROS subscriber callback
// ---------------------------------------------------------------------------

void RegistrationManager::resultCallback(const std_msgs::String::ConstPtr& msg)
{
    nlohmann::json data;
    try {
        data = nlohmann::json::parse(msg->data);
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(2.0, "[registration] JSON parse error: %s", e.what());
        return;
    }

    double result_stamp_sec = data.value("stamp", ros::Time::now().toSec());
    if (!std::isfinite(result_stamp_sec)) {
        result_stamp_sec = ros::Time::now().toSec();
    }

    // Submap-level registration results
    if (data.value("is_submap_registration", false)) {
        handleSubmapResult(data, result_stamp_sec);
        return;
    }

    const bool is_loop = data.value("is_loop_closure", false);
    const int prev_idx = data.at("prev_idx").get<int>();
    const int curr_idx = data.at("curr_idx").get<int>();
    const auto edge = std::make_pair(std::min(prev_idx, curr_idx),
                                     std::max(prev_idx, curr_idx));
    if (is_loop) {
        std::lock_guard<std::mutex> lk(lock);
        pending_loop_requests_.erase(edge);
    }

    if (!data.value("success", false)) return;

    const double score = data.value("score", -1e30);
    if (is_loop) {
        if ((curr_idx - prev_idx) < loop_closure_min_keyframe_gap_) return;
        if (score < loop_closure_detect_score_threshold_) return;
    } else {
        if (score < registration_score_threshold_) return;
        if ((curr_idx % registration_factor_every_n_frames_) != 0) return;
    }

    if (prev_idx >= curr_idx) return;

    // Parse the 4x4 transform
    const auto& t_arr = data.at("transform");
    Matrix4d T;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            T(r, c) = t_arr[static_cast<std::size_t>(r * 4 + c)].get<double>();
        }
    }

    if (!T.allFinite()) return;

    const bool force_loop = is_loop || (score >= loop_closure_score_threshold_);
    const bool use_super = is_loop && (score >= loop_closure_detect_score_threshold_);

    if (is_loop) {
        ROS_INFO("[registration] loop detected: X(%d)->X(%d) score=%.4f (super_noise=%s)",
                 prev_idx, curr_idx, score, use_super ? "true" : "false");
    }

    ResultItem item{prev_idx, curr_idx, T, force_loop, use_super,
                    is_loop, score, result_stamp_sec};

    if (!result_queue_.tryPush(std::move(item))) {
        ++dropped_result_msgs;
    }
}

// ---------------------------------------------------------------------------
// handleSubmapResult
// ---------------------------------------------------------------------------

void RegistrationManager::handleSubmapResult(const nlohmann::json& data,
                                              double result_stamp_sec)
{
    const int sid_prev = data.at("prev_idx").get<int>();
    const int sid_curr = data.at("curr_idx").get<int>();
    const bool success = data.value("success", false);
    const double score = data.value("score", -1e30);

    ROS_INFO("[registration] submap result S(%d)<->S(%d) success=%s score=%.4f",
             sid_prev, sid_curr, success ? "true" : "false", score);

    if (!success) return;
    if (global_graph_ == nullptr) return;

    const auto& t_arr = data.at("transform");
    Matrix4d T;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            T(r, c) = t_arr[static_cast<std::size_t>(r * 4 + c)].get<double>();
        }
    }

    if (!T.allFinite()) {
        ROS_WARN("[registration] submap result S(%d)<->S(%d) has NaN/Inf transform, "
                 "discarding", sid_prev, sid_curr);
        return;
    }

    const ros::Time stamp_msg = ros::Time(result_stamp_sec);
    global_graph_->handleSubmapRegistrationResult(sid_prev, sid_curr, T, score, stamp_msg);
}

// ---------------------------------------------------------------------------
// drainResults
// ---------------------------------------------------------------------------

void RegistrationManager::drainResults(const ros::Time& stamp)
{
    bool staged_any = false;
    while (true) {
        auto maybe = result_queue_.tryPop();
        if (!maybe) break;

        const ResultItem& r = *maybe;
        ros::Time factor_stamp = (std::isfinite(r.stamp_sec))
            ? ros::Time(r.stamp_sec)
            : stamp;

        try {
            stageRegistrationFactor(r.prev_idx, r.curr_idx, r.transform,
                                    r.score, r.force_loop, r.use_super,
                                    r.is_loop, factor_stamp);
            staged_any = true;
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[registration] drain error: %s", e.what());
            break;
        }
    }

    if (staged_any) {
        smoother_.flushStagedFactors();
    }
}

// ---------------------------------------------------------------------------
// stageRegistrationFactor
// ---------------------------------------------------------------------------

void RegistrationManager::stageRegistrationFactor(
    int prev_idx, int curr_idx, const Matrix4d& T_prev_to_curr,
    double score, bool force_loop, bool use_super,
    bool is_loop_candidate, const ros::Time& stamp)
{
    if (prev_idx < 0 || curr_idx > smoother_.latest_key_idx) return;

    auto t_curr_it = smoother_.key_t_sec.find(curr_idx);
    if (t_curr_it == smoother_.key_t_sec.end()) return;
    const double t_curr = t_curr_it->second;

    const double t_latest = [&]() {
        auto it = smoother_.key_t_sec.find(smoother_.latest_key_idx);
        return (it != smoother_.key_t_sec.end()) ? it->second : 0.0;
    }();

    if ((t_latest - t_curr) > smoother_.fixed_lag_s * 1.1) return;

    const auto edge = std::make_pair(std::min(prev_idx, curr_idx),
                                     std::max(prev_idx, curr_idx));
    {
        std::lock_guard<std::mutex> lk(lock);
        if (is_loop_candidate && loop_edges_added.count(edge)) return;
    }

    if (force_loop) {
        const auto [noise, sigma_t, sigma_r] = noiseFromScore(
            score, loop_sigma_t_min_, loop_sigma_t_max_,
            loop_sigma_r_min_, loop_sigma_r_max_);

        const gtsam::Pose3 rel_pose =
            FixedLagBackend::pose3FromMatrix(T_prev_to_curr);

        const bool prev_in_window = smoother_.isInLagWindow(prev_idx);

        if (prev_in_window) {
            // Both keys alive in the smoother → relative BetweenFactor
            smoother_.stageFactor(
                gtsam::BetweenFactor<gtsam::Pose3>(
                    X(prev_idx), X(curr_idx), rel_pose, noise).clone());

            ROS_INFO("[registration] staged loop between X(%d)->X(%d) "
                     "score=%.4f sigma_t=%.4f sigma_r=%.4f",
                     prev_idx, curr_idx, score, sigma_t, sigma_r);
        } else {
            // prev_idx marginalized — fall back to absolute prior on curr
            Matrix4d T_w_prev;
            {
                std::lock_guard<std::mutex> lk(smoother_.graph_lock);
                auto it = smoother_.pose_by_idx.find(prev_idx);
                if (it == smoother_.pose_by_idx.end()) {
                    ROS_WARN("[registration] no stored pose for X(%d), "
                             "cannot stage loop factor on X(%d)", prev_idx, curr_idx);
                    return;
                }
                T_w_prev = it->second;
            }
            const Matrix4d T_w_curr_from_loop = T_w_prev * T_prev_to_curr;
            smoother_.stageFactor(
                gtsam::PriorFactor<gtsam::Pose3>(
                    X(curr_idx),
                    FixedLagBackend::pose3FromMatrix(T_w_curr_from_loop),
                    noise).clone());

            ROS_INFO("[registration] staged loop prior on X(%d) "
                     "(prev X(%d) outside lag) score=%.4f sigma_t=%.4f sigma_r=%.4f",
                     curr_idx, prev_idx, score, sigma_t, sigma_r);
        }

        if (is_loop_candidate) {
            {
                std::lock_guard<std::mutex> lk(lock);
                loop_edges_added.insert(edge);
            }
            if (use_super && global_graph_ != nullptr) {
                std::map<int, Matrix4d> pose_snap;
                {
                    std::lock_guard<std::mutex> lk(smoother_.graph_lock);
                    pose_snap = smoother_.pose_by_idx;
                }
                global_graph_->addLoopFactor(prev_idx, curr_idx,
                                             T_prev_to_curr, stamp,
                                             pose_snap, score);
            }
        }
    } else {
        // Sequential → BetweenFactor
        auto t_prev_it = smoother_.key_t_sec.find(prev_idx);
        if (t_prev_it == smoother_.key_t_sec.end()) return;
        if ((t_latest - t_prev_it->second) > smoother_.fixed_lag_s * 1.1) return;

        const gtsam::Pose3 rel_pose =
            FixedLagBackend::pose3FromMatrix(T_prev_to_curr);

        const auto [noise, sigma_t, sigma_r] = noiseFromScore(
            score, seq_sigma_t_min_, seq_sigma_t_max_,
            seq_sigma_r_min_, seq_sigma_r_max_);

        smoother_.stageFactor(
            gtsam::BetweenFactor<gtsam::Pose3>(
                X(prev_idx), X(curr_idx), rel_pose, noise).clone());

        ROS_INFO("[registration] staged between factor X(%d)->X(%d) "
                 "score=%.4f sigma_t=%.4f sigma_r=%.4f",
                 prev_idx, curr_idx, score, sigma_t, sigma_r);
    }
}

} // namespace gmmslam
