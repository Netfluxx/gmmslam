#pragma once

#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"
#include "gmmslam/thread_safe_queue.hpp"

#include <ros/ros.h>
#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <atomic>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace gmmslam {

class FixedLagBackend {
public:
    FixedLagBackend(const SmootherConfig& smoother_cfg,
                    const GtNoiseConfig& gt_cfg,
                    const LoopClosureConfig& loop_cfg,
                    const ImuConfig& imu_cfg,
                    const std::string& benchmark_log_dir = {});

    bool initialize(const Matrix4d& pose, const ros::Time& stamp);
    bool initialized() const { return initialized_; }

    bool addFrame(int prev_idx, int curr_idx, const ros::Time& stamp,
                  const Matrix4d& predicted_pose,
                  const Matrix4d* gt_rel_mat = nullptr,
                  const std::vector<std::tuple<double, Vector3d, Vector3d>>* imu_measurements = nullptr,
                  bool add_external_odometry_factor = true);

    void stageFactor(const gtsam::NonlinearFactor::shared_ptr& factor);
    bool flushStagedFactors();

    bool isInLagWindow(int idx) const;

    void backendLoop(const std::atomic<bool>& shutdown);

    /// Request rebuilding the fixed-lag smoother around `anchor_idx` with pose
    /// `T_odom` (odom frame). Applied on the backend thread before the next solve.
    void scheduleReanchorToGt(int anchor_idx, const Matrix4d& T_odom, double t_sec);

    // --- Thread-safe shared state ---
    mutable std::mutex graph_lock;
    Matrix4d pose = Matrix4d::Identity();
    std::map<int, Matrix4d> pose_by_idx;
    std::map<int, double> pose_uncertainty_by_idx;
    std::map<int, Vector3d> velocity_by_idx;
    std::map<int, Eigen::Matrix<double,6,1>> bias_by_idx;
    std::map<int, double> key_t_sec;
    int latest_key_idx = 0;
    std::atomic<int> deferred_batches{0};

    double fixed_lag_s;

    static gtsam::Pose3 pose3FromMatrix(const Matrix4d& T);
    static Matrix4d matrixFromPose3(const gtsam::Pose3& p);

private:
    struct SolveBatch {
        int curr_idx;
        gtsam::NonlinearFactorGraph factors;
        gtsam::Values values;
        gtsam::FixedLagSmootherKeyTimestampMap timestamps;
    };

    void resetNewData();
    void rebuildSmoother(int anchor_idx);
    void rebuildSmootherCore(int anchor_idx, const Matrix4d& anchor_pose,
                             double t_sec, const char* reason,
                             bool zero_imu_at_anchor);
    void applyScheduledReanchorIfNeeded();
    gtsam::NonlinearFactorGraph filterStaleFactors(
        const gtsam::NonlinearFactorGraph& factors,
        const gtsam::Values& new_values,
        double t_latest,
        bool require_active_keys) const;
    void initBenchmarkLogs(const std::string& log_dir);
    void logOptimizationTiming(double stamp_sec, int curr_idx,
                               std::size_t factors, std::size_t values,
                               double update_ms, double estimate_ms,
                               double total_ms, bool success,
                               const std::string& note);

    int pose_history_keep_;
    bool enable_imu_;

    // Noise models
    gtsam::SharedNoiseModel odom_noise_;
    gtsam::SharedNoiseModel odom_noise_lost_;
    gtsam::SharedNoiseModel prior_noise_;
    gtsam::SharedNoiseModel gt_factor_noise_;
    gtsam::SharedNoiseModel loop_closure_noise_;
    gtsam::SharedNoiseModel loop_closure_super_noise_;
    gtsam::SharedNoiseModel vel_prior_noise_;
    gtsam::SharedNoiseModel bias_prior_noise_;
    gtsam::SharedNoiseModel bias_rw_noise_;

    // Cached IMU preintegration params (built once in constructor)
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_preint_params_;

    // GTSAM state
    std::unique_ptr<gtsam::IncrementalFixedLagSmoother> fixed_lag_;
    std::mutex new_data_lock_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    gtsam::FixedLagSmootherKeyTimestampMap new_timestamps_;
    bool initialized_ = false;
    std::set<int> inserted_pose_keys_;
    std::set<gtsam::Key> active_gtsam_keys_;
    std::ofstream benchmark_smoother_csv_;
    bool benchmark_logs_enabled_ = false;

    // Solve queue
    ThreadSafeQueue<SolveBatch> solve_queue_{16};

    // Failure tracking
    int consecutive_failures_ = 0;
    static constexpr int kMaxConsecutiveFailures = 5;

    std::mutex reanchor_scheduled_mu_;
    bool reanchor_scheduled_ = false;
    int reanchor_anchor_idx_ = 0;
    Matrix4d reanchor_pose_ = Matrix4d::Identity();
    double reanchor_t_sec_ = 0.0;
};

} // namespace gmmslam
