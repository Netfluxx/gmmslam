#pragma once

#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"
#include "gmmslam/thread_safe_queue.hpp"

#include <Eigen/Core>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/String.h>

namespace gmmslam {

// Forward declarations
class FixedLagBackend;
class GlobalPoseGraph;

class RegistrationManager {
public:
    RegistrationManager(FixedLagBackend& smoother,
                        GlobalPoseGraph* global_graph,
                        ros::Publisher reg_request_pub,
                        const RegistrationConfig& reg_cfg,
                        const LoopClosureConfig& lc_cfg,
                        const SogmmConfig& sogmm_cfg,
                        const std::string& gmm_dir);

    // Enqueue a GMM fit. Returns true if enqueued, false if dropped.
    bool enqueueFit(int frame_idx, const ros::Time& stamp,
                    const Eigen::MatrixXf& pts,
                    double capture_t_sec, const Matrix4d& capture_pose);

    // Background fit worker thread entry point
    void fitWorkerLoop(const std::atomic<bool>& shutdown);

    // ROS result callback
    void resultCallback(const std_msgs::String::ConstPtr& msg);

    // Drain result queue and stage factors in the smoother
    void drainResults(const ros::Time& stamp);

    // --- Thread-safe shared state ---
    mutable std::mutex lock;
    std::map<int, std::string> gmm_paths_by_idx;
    std::map<int, LocalGmmEntry> local_gmms_by_idx;
    int latest_gmm_idx = -1;
    GmmModel latest_gmm_model;
    bool has_latest_gmm = false;
    std::set<std::pair<int,int>> loop_edges_added;

    // Backpressure counters
    std::atomic<int> dropped_fit_frames{0};
    std::atomic<int> dropped_result_msgs{0};

private:
    struct FitJob {
        int frame_idx;
        ros::Time stamp;
        Eigen::MatrixXf points;
        double capture_t_sec;
        Matrix4d capture_pose;
    };

    struct ResultItem {
        int prev_idx;
        int curr_idx;
        Matrix4d transform;
        bool force_loop;
        bool use_super;
        bool is_loop;
        double score;
        double stamp_sec;
    };

    struct NoiseResult {
        gtsam::SharedNoiseModel noise;
        double sigma_t;
        double sigma_r;
    };

    NoiseResult noiseFromScore(double score,
                                double sigma_t_min, double sigma_t_max,
                                double sigma_r_min, double sigma_r_max) const;

    void finishFit(const GmmModel& model, int frame_idx, const ros::Time& stamp,
                   double capture_t_sec, const Matrix4d& capture_pose);

    void enqueueLoopClosureRequests(int curr_idx, const ros::Time& stamp,
                                     const std::string& source_path,
                                     int sequential_prev_idx = -1);

    void stageRegistrationFactor(int prev_idx, int curr_idx,
                                  const Matrix4d& T_prev_to_curr,
                                  double score,
                                  bool force_loop, bool use_super,
                                  bool is_loop_candidate,
                                  const ros::Time& stamp);

    void handleSubmapResult(const nlohmann::json& data, double result_stamp_sec);

    // References to other subsystems
    FixedLagBackend& smoother_;
    GlobalPoseGraph* global_graph_;
    ros::Publisher reg_pub_;

    // Config
    SogmmConfig sogmm_cfg_;
    std::string gmm_dir_;
    double registration_score_threshold_;
    int registration_factor_every_n_frames_;
    double loop_closure_score_threshold_;
    double loop_closure_detect_score_threshold_;
    bool enable_loop_closure_;
    int loop_closure_min_keyframe_gap_;
    int loop_closure_max_candidates_;
    double loop_closure_search_radius_m_;
    int loop_closure_search_cooldown_keyframes_;
    int loop_closure_request_every_n_keyframes_;
    double loop_closure_min_separation_m_;
    double loop_closure_min_separation_deg_;
    double loop_closure_max_age_s_;
    int loop_closure_gmm_keep_keyframes_;
    double score_sigma_low_;
    double score_sigma_high_;
    double seq_sigma_t_min_, seq_sigma_t_max_;
    double seq_sigma_r_min_, seq_sigma_r_max_;
    double loop_sigma_t_min_, loop_sigma_t_max_;
    double loop_sigma_r_min_, loop_sigma_r_max_;
    bool compensate_fit_latency_;

    // State
    std::set<std::pair<int,int>> pending_loop_requests_;
    int last_loop_search_idx_ = -1000000;

    // Queues
    ThreadSafeQueue<FitJob> fit_queue_;
    ThreadSafeQueue<ResultItem> result_queue_;
};

} // namespace gmmslam
