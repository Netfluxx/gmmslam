#include "gmmslam/registration_manager.hpp"
#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/sogmm_fitting.hpp"
#include "gmmslam/util/gmm_utils.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sstream>

using gtsam::symbol_shorthand::X;

namespace {

// Null-safe JSON accessors. nlohmann::json serializes non-finite doubles
// (NaN, ±Inf) as JSON null, so a raw `.value<T>()` on such a field throws
// type_error.302. These helpers treat missing / null / wrong-type fields
// as "use default".
template <typename T>
T jsonNumber(const nlohmann::json& j, const char* key, T fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null() || !it->is_number()) return fallback;
    return it->get<T>();
}

bool jsonBool(const nlohmann::json& j, const char* key, bool fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null() || !it->is_boolean()) return fallback;
    return it->get<bool>();
}

} // namespace

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
    const SolidConfig& solid_cfg,
    const SogmmConfig& sogmm_cfg,
    const std::string& gmm_dir,
    ros::Publisher loop_closure_markers_pub,
    std::string map_frame)
    : smoother_(smoother),
      global_graph_(global_graph),
      reg_pub_(std::move(reg_request_pub)),
      loop_closure_markers_pub_(std::move(loop_closure_markers_pub)),
      map_frame_(map_frame.empty() ? std::string("map") : std::move(map_frame)),
      sogmm_cfg_(sogmm_cfg),
      solid_cfg_(solid_cfg),
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
      place_index_(solid_cfg),
      fit_queue_(static_cast<std::size_t>(reg_cfg.queue_size)),
      result_queue_(static_cast<std::size_t>(reg_cfg.result_queue_size))
{
    std::filesystem::create_directories(gmm_dir_);
    if (solid_cfg_.enable) {
        ROS_INFO("[registration] SOLiD enabled: dim=%d (range=%d + angle=%d) "
                 "height=%d range=[%.2f, %.2f] fov=[%.1f, %.1f] deg "
                 "cos_gate=%.2f yaw_prior=%s",
                 solid_cfg_.num_range + solid_cfg_.num_angle,
                 solid_cfg_.num_range, solid_cfg_.num_angle,
                 solid_cfg_.num_height,
                 solid_cfg_.min_distance_m, solid_cfg_.max_distance_m,
                 solid_cfg_.fov_down_deg, solid_cfg_.fov_up_deg,
                 solid_cfg_.cos_similarity_threshold,
                 solid_cfg_.provide_yaw_prior ? "on" : "off");
    } else {
        ROS_INFO("[registration] SOLiD disabled — loop search is radius-only");
    }
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
// submitKeyframeDescriptor — synchronous SOLiD index insertion
// ---------------------------------------------------------------------------

void RegistrationManager::submitKeyframeDescriptor(int frame_idx,
                                                    const Eigen::MatrixXf& pts)
{
    if (!solid_cfg_.enable) return;
    ROS_INFO("[registration][DBG] submitKeyframeDescriptor X(%d) pts=%ld",
             frame_idx, static_cast<long>(pts.rows()));
    SolidDescriptor desc = place_index_.compute(pts);
    if (desc.empty()) {
        ROS_WARN("[registration][DBG] SOLiD: descriptor empty for X(%d) "
                 "(points=%d after FOV/range filter)",
                 frame_idx, desc.point_count);
        return;
    }
    place_index_.insert(frame_idx, std::move(desc));
    ROS_INFO("[registration][DBG] SOLiD index now size=%zu",
             place_index_.size());
}

// ---------------------------------------------------------------------------
// enqueueFit — backpressure: drop oldest if queue full
// ---------------------------------------------------------------------------

bool RegistrationManager::enqueueFit(int frame_idx, const ros::Time& stamp,
                                      const Eigen::MatrixXf& pts,
                                      double capture_t_sec,
                                      const Matrix4d& capture_pose)
{
    const std::size_t qsize_before = fit_queue_.size();
    ROS_INFO("[registration][DBG] enqueueFit ENTER X(%d) pts=%ld "
             "queue_depth=%zu",
             frame_idx, static_cast<long>(pts.rows()), qsize_before);
    FitJob job{frame_idx, stamp, pts, capture_t_sec, capture_pose};

    if (fit_queue_.tryPush(std::move(job))) {
        ROS_INFO("[registration][DBG] enqueueFit OK X(%d) queue_depth_after=%zu",
                 frame_idx, fit_queue_.size());
        return true;
    }

    ++dropped_fit_frames;
    ROS_WARN("[registration][DBG] enqueueFit FULL X(%d) queue_depth=%zu -> "
             "dropping oldest and retrying",
             frame_idx, fit_queue_.size());
    fit_queue_.tryPop();
    FitJob retry{frame_idx, stamp, pts, capture_t_sec, capture_pose};
    const bool ok = fit_queue_.tryPush(std::move(retry));
    ROS_WARN("[registration][DBG] enqueueFit retry X(%d) ok=%d "
             "queue_depth_after=%zu",
             frame_idx, ok ? 1 : 0, fit_queue_.size());
    return ok;
}

// ---------------------------------------------------------------------------
// fitWorkerLoop — background thread
// ---------------------------------------------------------------------------

void RegistrationManager::fitWorkerLoop(const std::atomic<bool>& shutdown)
{
    ROS_INFO("[registration][DBG] fitWorkerLoop STARTED tid=%zu",
             static_cast<std::size_t>(
                 std::hash<std::thread::id>{}(std::this_thread::get_id())));
    while (!shutdown.load(std::memory_order_relaxed)) {
        auto maybe_job = fit_queue_.popWithTimeout(std::chrono::milliseconds(50));
        if (!maybe_job) {
            continue;
        }

        FitJob& job = *maybe_job;
        const int fid = job.frame_idx;
        {
            std::lock_guard<std::mutex> lk(lock);
            awaiting_fits_.insert(fid);
        }
        const std::size_t qsize_after_pop = fit_queue_.size();
        ROS_INFO("[registration][DBG] fit START X(%d) pts=%ld "
                 "queue_depth_after_pop=%zu",
                 fid, static_cast<long>(job.points.rows()),
                 qsize_after_pop);
        const auto t0 = std::chrono::steady_clock::now();
        bool finish_ok = false;
        try {
            GmmModel model = fitSogmm(job.points, sogmm_cfg_);
            const auto t1 = std::chrono::steady_clock::now();
            const double dt_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            ROS_INFO("[registration][DBG] fit DONE  X(%d) pts=%ld K=%d "
                     "elapsed_ms=%.1f",
                     fid, static_cast<long>(job.points.rows()),
                     model.numComponents(), dt_ms);
            if (model.numComponents() > 0) {
                finishFit(model, job.frame_idx, job.stamp,
                          job.capture_t_sec, job.capture_pose);
                finish_ok = true;
            } else {
                ROS_WARN(
                    "[registration][DBG] SOGMM fit returned EMPTY model "
                    "for frame %d pts=%ld elapsed_ms=%.1f",
                    fid, static_cast<long>(job.points.rows()),
                    dt_ms);
            }
        } catch (const std::exception& e) {
            const auto t1 = std::chrono::steady_clock::now();
            const double dt_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            ROS_ERROR(
                "[registration][DBG] fit THROW X(%d) pts=%ld elapsed_ms=%.1f "
                "what=%s",
                fid, static_cast<long>(job.points.rows()), dt_ms,
                e.what());
        }
        if (!finish_ok) {
            std::lock_guard<std::mutex> lk(lock);
            awaiting_fits_.erase(fid);
        }
    }
    ROS_WARN("[registration][DBG] fitWorkerLoop EXITING (shutdown)");
}

// ---------------------------------------------------------------------------
// finishFit — post-fit bookkeeping, save .gmm, emit registration requests
// ---------------------------------------------------------------------------

void RegistrationManager::finishFit(const GmmModel& model, int frame_idx,
                                     const ros::Time& stamp,
                                     double capture_t_sec,
                                     const Matrix4d& capture_pose)
{
    ROS_INFO("[registration][DBG] finishFit ENTER X(%d) K=%d",
             frame_idx, model.numComponents());
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
        {
            std::lock_guard<std::mutex> lk(lock);
            awaiting_fits_.erase(frame_idx);
        }
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
    bool skip_sequential_d2d = false;
    {
        std::lock_guard<std::mutex> lk(lock);
        awaiting_fits_.erase(frame_idx);

        gmm_paths_by_idx[frame_idx] = gmm_path;

        auto entry_it = local_gmms_by_idx.find(frame_idx);
        if (entry_it != local_gmms_by_idx.end()) {
            entry_it->second.map_pose = map_pose;
            entry_it->second.has_map_pose = true;
            entry_it->second.fit_t_sec = fit_t_sec;
            entry_it->second.capture_pose = effective_capture_pose;
        }

        // Find the immediately preceding GMM in completion order (map keys).
        auto curr_it = gmm_paths_by_idx.find(frame_idx);
        if (curr_it != gmm_paths_by_idx.begin()) {
            --curr_it;
            prev_idx = curr_it->first;
            prev_path = curr_it->second;
        }

        // Parallel SOGMM workers can finish out of order: `prev_idx` may jump
        // backward over indices whose fits are still running. Sequential D2D
        // would then add a long-hop BetweenFactor that disagrees with the GT
        // chain. Skip until the gap has no in-flight fits.
        if (prev_idx >= 0) {
            for (int k = prev_idx + 1; k < frame_idx; ++k) {
                if (awaiting_fits_.count(k) > 0) {
                    skip_sequential_d2d = true;
                    break;
                }
            }
        }
    }

    if (skip_sequential_d2d && prev_idx >= 0) {
        ROS_WARN_THROTTLE(
            2.0,
            "[registration] skip sequential D2D X(%d)->X(%d): "
            "intermediate odom indices still fitting (async ordering)",
            prev_idx, frame_idx);
    }

    // Publish sequential D2D request
    if (prev_idx >= 0 && !prev_path.empty()) {
        if (!skip_sequential_d2d) {
            const int odom_gap = frame_idx - prev_idx;
            if (odom_gap > 25) {
                ROS_WARN_THROTTLE(
                    5.0,
                    "[registration] sequential D2D long hop X(%d)->X(%d) "
                    "(odom_gap=%d): async backlog or sparse `request_every_n_frames`; "
                    "attaching smoother relative pose as D2D init",
                    prev_idx, frame_idx, odom_gap);
            }

            nlohmann::json payload;
            payload["prev_idx"]         = prev_idx;
            payload["curr_idx"]         = frame_idx;
            payload["stamp"]            = stamp.toSec();
            payload["source_path"]      = gmm_path;
            payload["target_path"]      = prev_path;
            payload["is_loop_closure"]  = false;

            // GMMs are in per-frame sensor coordinates; D2D matches source(curr)
            // into target(prev). inv(T_prev)*T_curr is the same convention as the
            // BetweenFactor relative staged from the D2D result (see stageRegistrationFactor).
            {
                std::lock_guard<std::mutex> lk(smoother_.graph_lock);
                auto itp = smoother_.pose_by_idx.find(prev_idx);
                auto itc = smoother_.pose_by_idx.find(frame_idx);
                if (itp != smoother_.pose_by_idx.end() &&
                    itc != smoother_.pose_by_idx.end()) {
                    const Matrix4d T_init = itp->second.inverse() * itc->second;
                    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row(T_init);
                    std::vector<double> flat(16);
                    for (int k = 0; k < 16; ++k) {
                        flat[static_cast<std::size_t>(k)] = T_row.data()[k];
                    }
                    payload["initial_transform"] = std::move(flat);
                }
            }

            std_msgs::String msg;
            msg.data = payload.dump();
            reg_pub_.publish(msg);
        }

        enqueueLoopClosureRequests(frame_idx, stamp, gmm_path, prev_idx);
    }

    // Purge old gmm_paths beyond keep window
    const int min_keep = std::max(0, frame_idx - loop_closure_gmm_keep_keyframes_);
    {
        std::lock_guard<std::mutex> lk(lock);
        for (auto it = gmm_paths_by_idx.begin(); it != gmm_paths_by_idx.end();) {
            if (it->first < min_keep) {
                it = gmm_paths_by_idx.erase(it);
            } else {
                break;
            }
        }
    }
    // Keep the SOLiD index aligned with the GMM cache so we never rank a
    // candidate whose GMM has already been evicted.
    place_index_.eraseOlderThan(min_keep);

    if (on_fit_complete_) {
        on_fit_complete_(frame_idx, stamp);
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

    // ------------------------------------------------------------------
    // Candidate selection: SOLiD-only (spatial gate already applied above).
    // Among the radius-gated hits, pick the single best SOLiD cosine match
    // and dispatch D2D only if it clears the configured threshold.
    // ------------------------------------------------------------------
    SolidDescriptor q_desc;
    const bool have_q_desc =
        solid_cfg_.enable &&
        place_index_.get(curr_idx, q_desc);

    struct Ranked {
        int    idx;
        double distance;
        double cos_sim;
        std::string path;
    };
    std::vector<Ranked> selected;

    if (!have_q_desc) {
        ROS_INFO_THROTTLE(10.0,
            "[registration] loop search @key %d: no SOLiD query descriptor, "
            "skipping (radius=%zu, gmm_cached=%zu)",
            curr_idx, near.size(), gmm_snapshot.size());
        return;
    }

    int dropped_by_gate = 0;
    int scored = 0;
    Ranked best{-1, 0.0, -1.0, {}};
    for (const auto& cand : near) {
        SolidDescriptor c_desc;
        if (!place_index_.get(cand.idx, c_desc)) continue;
        const double cs = place_index_.rangeCosine(q_desc, c_desc);
        ++scored;
        if (cs < solid_cfg_.cos_similarity_threshold) {
            ++dropped_by_gate;
            continue;
        }
        if (cs > best.cos_sim) {
            best = Ranked{cand.idx, cand.distance, cs, cand.path};
        }
    }

    if (best.idx < 0) {
        ROS_INFO_THROTTLE(10.0,
            "[registration] loop search @key %d: 0 accepted candidates "
            "(radius=%zu, scored=%d, dropped_by_SOLiD=%d)",
            curr_idx, near.size(), scored, dropped_by_gate);
        return;
    }
    selected.push_back(best);

    // ------------------------------------------------------------------
    // Previous implementation: fused α·radius + β·cos ranking, optional
    // full-index rescue path bypassing the radius, and greedy spatial +
    // yaw diversity selection of up to `max_candidates`. Kept here for
    // reference while we evaluate SOLiD-only gating.
    // ------------------------------------------------------------------
    /*
    const double R_inv = loop_closure_search_radius_m_ > 1e-6
                         ? (1.0 / loop_closure_search_radius_m_) : 1.0;
    // ... fused scoring, rescue, greedy diversity ...
    */

    const int n_selected = static_cast<int>(selected.size());
    ROS_INFO("[registration] loop search @key %d: radius=%zu scored=%d "
             "dropped_by_SOLiD=%d, best_cos=%.3f dispatching %d",
             curr_idx, near.size(), scored, dropped_by_gate,
             best.cos_sim, n_selected);

    for (int i = 0; i < n_selected; ++i) {
        const auto& cand = selected[static_cast<std::size_t>(i)];

        nlohmann::json payload;
        payload["prev_idx"]        = cand.idx;
        payload["curr_idx"]        = curr_idx;
        payload["stamp"]           = stamp.toSec();
        payload["source_path"]     = source_path;
        payload["target_path"]     = cand.path;
        payload["is_loop_closure"] = true;
        if (!std::isnan(cand.cos_sim)) {
            payload["solid_cos_sim"] = cand.cos_sim;
        }
        if (cand.distance < 0.0) {
            payload["solid_rescue"] = true;
        }

        // SOLiD-derived yaw prior for D2D initialization.
        if (have_q_desc && solid_cfg_.provide_yaw_prior) {
            SolidDescriptor c_desc;
            if (place_index_.get(cand.idx, c_desc)) {
                const auto est =
                    place_index_.yawEstimate(q_desc, c_desc);
                if (est.valid) {
                    const double c = std::cos(est.yaw_rad);
                    const double s = std::sin(est.yaw_rad);
                    std::vector<double> T_init = {
                        c, -s, 0.0, 0.0,
                        s,  c, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    };
                    payload["initial_transform"] = T_init;
                    payload["solid_yaw_deg"] = est.yaw_rad * 180.0 / M_PI;
                    payload["solid_yaw_overlap"] = est.overlap;
                }
            }
        }

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

    double result_stamp_sec =
        jsonNumber<double>(data, "stamp", ros::Time::now().toSec());
    if (!std::isfinite(result_stamp_sec)) {
        result_stamp_sec = ros::Time::now().toSec();
    }

    // Submap-level registration results
    if (jsonBool(data, "is_submap_registration", false)) {
        handleSubmapResult(data, result_stamp_sec);
        return;
    }

    const bool is_loop = jsonBool(data, "is_loop_closure", false);
    const int prev_idx = jsonNumber<int>(data, "prev_idx", -1);
    const int curr_idx = jsonNumber<int>(data, "curr_idx", -1);
    if (prev_idx < 0 || curr_idx < 0) {
        ROS_WARN_THROTTLE(2.0,
            "[registration] result missing prev_idx/curr_idx, discarding");
        return;
    }
    const auto edge = std::make_pair(std::min(prev_idx, curr_idx),
                                     std::max(prev_idx, curr_idx));
    if (is_loop) {
        std::lock_guard<std::mutex> lk(lock);
        pending_loop_requests_.erase(edge);
    }

    if (!jsonBool(data, "success", false)) return;

    const double score = jsonNumber<double>(data, "score", -1e30);
    if (is_loop) {
        if ((curr_idx - prev_idx) < loop_closure_min_keyframe_gap_) return;
        if (score < loop_closure_detect_score_threshold_) return;
    } else {
        if (score < registration_score_threshold_) return;
        if ((curr_idx % registration_factor_every_n_frames_) != 0) return;
    }

    if (prev_idx >= curr_idx) return;

    // Parse the 4x4 transform (null-safe — a null entry yields NaN, which
    // fails the allFinite() check below).
    auto t_it = data.find("transform");
    if (t_it == data.end() || !t_it->is_array() || t_it->size() != 16) return;
    Matrix4d T;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const auto& v = (*t_it)[static_cast<std::size_t>(r * 4 + c)];
            T(r, c) = v.is_number()
                ? v.get<double>()
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    if (!T.allFinite()) return;

    const bool force_loop = is_loop || (score >= loop_closure_score_threshold_);
    const bool use_super = is_loop && (score >= loop_closure_detect_score_threshold_);

    if (is_loop) {
        ROS_INFO("[registration] loop detected: X(%d)->X(%d) score=%.4f (super_noise=%s)",
                 prev_idx, curr_idx, score, use_super ? "true" : "false");
        if (curr_idx > last_loop_accepted_idx_) {
            last_loop_accepted_idx_ = curr_idx;
        }
    }

    ResultItem item;
    item.prev_idx = prev_idx;
    item.curr_idx = curr_idx;
    item.transform = T;
    item.force_loop = force_loop;
    item.use_super = use_super;
    item.is_loop = is_loop;
    item.score = score;
    item.stamp_sec = result_stamp_sec;
    if (is_loop) {
        item.solid_rescue = jsonBool(data, "solid_rescue", false);
        auto cos_it = data.find("solid_cos_sim");
        if (cos_it != data.end() && cos_it->is_number()) {
            item.has_solid_cos_sim = true;
            item.solid_cos_sim = cos_it->get<double>();
        }
    }

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
    const int sid_prev = jsonNumber<int>(data, "prev_idx", -1);
    const int sid_curr = jsonNumber<int>(data, "curr_idx", -1);
    const bool success = jsonBool(data, "success", false);
    const double score = jsonNumber<double>(data, "score", -1e30);

    if (sid_prev < 0 || sid_curr < 0) {
        ROS_WARN_THROTTLE(2.0,
            "[registration] submap result missing prev_idx/curr_idx, discarding");
        return;
    }

    ROS_INFO("[registration] submap result S(%d)<->S(%d) success=%s score=%.4f",
             sid_prev, sid_curr, success ? "true" : "false", score);

    const ros::Time stamp_msg = ros::Time(result_stamp_sec);
    if (global_graph_ != nullptr) {
        global_graph_->acknowledgeSubmapOverlapAttempt(sid_prev, sid_curr, stamp_msg);
    }

    if (!success) {
        if (global_graph_ != nullptr) {
            const int sid_new = std::max(sid_prev, sid_curr);
            global_graph_->maybeApplyDeferredInternalSubmapPrune(sid_new, stamp_msg);
        }
        return;
    }

    auto t_it = data.find("transform");
    if (t_it == data.end() || !t_it->is_array() || t_it->size() != 16) {
        ROS_WARN("[registration] submap result S(%d)<->S(%d) missing/invalid transform",
                 sid_prev, sid_curr);
        if (global_graph_ != nullptr) {
            const int sid_new = std::max(sid_prev, sid_curr);
            global_graph_->maybeApplyDeferredInternalSubmapPrune(sid_new, stamp_msg);
        }
        return;
    }
    Matrix4d T;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const auto& v = (*t_it)[static_cast<std::size_t>(r * 4 + c)];
            T(r, c) = v.is_number()
                ? v.get<double>()
                : std::numeric_limits<double>::quiet_NaN();
        }
    }

    if (!T.allFinite()) {
        ROS_WARN("[registration] submap result S(%d)<->S(%d) has NaN/Inf transform, "
                 "discarding", sid_prev, sid_curr);
        if (global_graph_ != nullptr) {
            const int sid_new = std::max(sid_prev, sid_curr);
            global_graph_->maybeApplyDeferredInternalSubmapPrune(sid_new, stamp_msg);
        }
        return;
    }

    if (global_graph_ != nullptr) {
        global_graph_->handleSubmapRegistrationResult(sid_prev, sid_curr, T, score, stamp_msg);
    }
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
                                    r.is_loop, factor_stamp,
                                    r.has_solid_cos_sim, r.solid_cos_sim,
                                    r.solid_rescue);
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
    bool is_loop_candidate, const ros::Time& stamp,
    bool has_solid_cos_sim, double solid_cos_sim, bool solid_rescue)
{
    int latest_key_snapshot = 0;
    double t_curr_snapshot = 0.0;
    double t_latest_snapshot = 0.0;
    bool have_prev_t = false;
    double t_prev_snapshot = 0.0;
    bool prev_in_window = false;
    {
        std::lock_guard<std::mutex> gk(smoother_.graph_lock);
        latest_key_snapshot = smoother_.latest_key_idx;
        if (prev_idx < 0 || curr_idx > latest_key_snapshot) {
            return;
        }
        const auto t_curr_it = smoother_.key_t_sec.find(curr_idx);
        if (t_curr_it == smoother_.key_t_sec.end()) {
            return;
        }
        t_curr_snapshot = t_curr_it->second;
        const auto lit = smoother_.key_t_sec.find(latest_key_snapshot);
        t_latest_snapshot =
            (lit != smoother_.key_t_sec.end()) ? lit->second : 0.0;

        const auto t_prev_it = smoother_.key_t_sec.find(prev_idx);
        if (t_prev_it != smoother_.key_t_sec.end()) {
            have_prev_t = true;
            t_prev_snapshot = t_prev_it->second;
        }
        if (have_prev_t) {
            prev_in_window =
                (t_latest_snapshot - t_prev_snapshot) <=
                smoother_.fixed_lag_s * 1.1;
        }
    }

    if ((t_latest_snapshot - t_curr_snapshot) > smoother_.fixed_lag_s * 1.1) {
        return;
    }

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
            publishLoopClosureMarkers(prev_idx, curr_idx, score, stamp,
                                       has_solid_cos_sim, solid_cos_sim,
                                       solid_rescue);
        }
    } else {
        // Sequential → BetweenFactor
        if (!have_prev_t) {
            return;
        }
        if ((t_latest_snapshot - t_prev_snapshot) > smoother_.fixed_lag_s * 1.1) {
            return;
        }

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

void RegistrationManager::publishLoopClosureMarkers(
    int prev_idx, int curr_idx, double score, const ros::Time& stamp,
    bool has_solid_cos, double solid_cos_sim, bool solid_rescue)
{
    if (loop_closure_markers_pub_.getTopic().empty()) {
        return;
    }

    Vector3d p_prev, p_curr;
    {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        auto it1 = smoother_.pose_by_idx.find(prev_idx);
        auto it2 = smoother_.pose_by_idx.find(curr_idx);
        if (it1 == smoother_.pose_by_idx.end() ||
            it2 == smoother_.pose_by_idx.end()) {
            ROS_DEBUG("[registration] loop_closure viz: missing pose "
                      "X(%d) or X(%d)", prev_idx, curr_idx);
            return;
        }
        p_prev = it1->second.block<3, 1>(0, 3);
        p_curr = it2->second.block<3, 1>(0, 3);
    }

    const int uid =
        loop_viz_uid_.fetch_add(2, std::memory_order_relaxed);

    visualization_msgs::MarkerArray arr;
    arr.markers.reserve(2);

    visualization_msgs::Marker line;
    line.header.frame_id = map_frame_;
    line.header.stamp = stamp;
    line.ns = "gmmslam_loop_edges";
    line.id = uid;
    line.type = visualization_msgs::Marker::LINE_LIST;
    line.action = visualization_msgs::Marker::ADD;
    line.pose.orientation.w = 1.0;
    line.scale.x = 0.08;
    line.lifetime = ros::Duration(0.0);
    if (solid_rescue) {
        line.color.r = 1.0f;
        line.color.g = 0.55f;
        line.color.b = 0.05f;
    } else {
        line.color.r = 0.15f;
        line.color.g = 0.85f;
        line.color.b = 0.25f;
    }
    line.color.a = 1.0f;
    geometry_msgs::Point a, b;
    a.x = p_prev.x();
    a.y = p_prev.y();
    a.z = p_prev.z();
    b.x = p_curr.x();
    b.y = p_curr.y();
    b.z = p_curr.z();
    line.points.push_back(a);
    line.points.push_back(b);
    arr.markers.push_back(std::move(line));

    visualization_msgs::Marker txt;
    txt.header.frame_id = map_frame_;
    txt.header.stamp = stamp;
    txt.ns = "gmmslam_loop_labels";
    txt.id = uid + 1;
    txt.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    txt.action = visualization_msgs::Marker::ADD;
    txt.pose.orientation.w = 1.0;
    txt.pose.position.x = 0.5 * (p_prev.x() + p_curr.x());
    txt.pose.position.y = 0.5 * (p_prev.y() + p_curr.y());
    txt.pose.position.z = 0.5 * (p_prev.z() + p_curr.z()) + 0.35;
    txt.scale.z = 0.35;
    txt.lifetime = ros::Duration(0.0);
    txt.color.r = 1.0f;
    txt.color.g = 1.0f;
    txt.color.b = 1.0f;
    txt.color.a = 1.0f;

    std::ostringstream oss;
    oss << "loop X(" << prev_idx << ")-X(" << curr_idx << ") D2D=" << score;
    if (has_solid_cos) {
        oss << " SOLiD_cos=" << solid_cos_sim;
    }
    if (solid_rescue) {
        oss << " [rescue]";
    }
    txt.text = oss.str();
    arr.markers.push_back(std::move(txt));

    loop_closure_markers_pub_.publish(arr);
}

} // namespace gmmslam
