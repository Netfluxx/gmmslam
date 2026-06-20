#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/rclcpp_logging.hpp"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/inference/Symbol.h>

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>

namespace gmmslam {

using gtsam::symbol_shorthand::X;
using gtsam::Symbol;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Point3;
using gtsam::noiseModel::Diagonal;
using gtsam::noiseModel::Isotropic;
using gtsam::imuBias::ConstantBias;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static gtsam::Key keyV(int idx) { return Symbol('v', idx); }
static gtsam::Key keyB(int idx) { return Symbol('b', idx); }

static double stampToSec(const rclcpp::Time& t) { return t.seconds(); }

static ConstantBias zeroBias() {
    return ConstantBias(Vector3d::Zero(), Vector3d::Zero());
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

FixedLagBackend::FixedLagBackend(const SmootherConfig& smoother_cfg,
                                 const ExtOdomConfig& gt_cfg,
                                 const LoopClosureConfig& loop_cfg,
                                 const ImuConfig& imu_cfg,
                                 const std::string& benchmark_log_dir)
    : fixed_lag_s(smoother_cfg.fixed_lag_s),
      pose_history_keep_(smoother_cfg.pose_history_keep),
      enable_imu_(imu_cfg.enable_preintegration)
{
    // Odometry noise (GTSAM ordering: 3 rotation, 3 translation)
    const double sr = smoother_cfg.odom_noise_sigma_r;
    const double st = smoother_cfg.odom_noise_sigma_t;
    Eigen::Matrix<double,6,1> odom_sigmas;
    odom_sigmas << sr, sr, sr, st, st, st;

    odom_noise_      = Diagonal::Sigmas(odom_sigmas);
    odom_noise_lost_ = Diagonal::Sigmas(odom_sigmas * smoother_cfg.lost_scale);
    prior_noise_     = Diagonal::Sigmas(odom_sigmas * smoother_cfg.prior_scale);

    // GT factor noise
    Eigen::Matrix<double,6,1> gt_sigmas;
    gt_sigmas << gt_cfg.factor_sigma_r, gt_cfg.factor_sigma_r, gt_cfg.factor_sigma_r,
                 gt_cfg.factor_sigma_t, gt_cfg.factor_sigma_t, gt_cfg.factor_sigma_t;
    gt_factor_noise_ = Diagonal::Sigmas(gt_sigmas);

    // Loop-closure noise
    Eigen::Matrix<double,6,1> lc_sigmas;
    lc_sigmas << loop_cfg.super_sigma_r, loop_cfg.super_sigma_r, loop_cfg.super_sigma_r,
                 loop_cfg.super_sigma_t, loop_cfg.super_sigma_t, loop_cfg.super_sigma_t;

    // "strong" loop-closure noise reuses the same super_sigma_{t,r} fields
    // (the Python caller passes strong_factor_sigma_{t,r} as loop_sigma_{t,r}).
    // In the C++ config these live as super_sigma_{t,r} on LoopClosureConfig;
    // the registration-level adaptive sigmas are resolved externally.
    loop_closure_noise_ = Diagonal::Sigmas(lc_sigmas);

    Eigen::Matrix<double,6,1> lc_super_sigmas;
    lc_super_sigmas << loop_cfg.super_sigma_r, loop_cfg.super_sigma_r, loop_cfg.super_sigma_r,
                       loop_cfg.super_sigma_t, loop_cfg.super_sigma_t, loop_cfg.super_sigma_t;
    loop_closure_super_noise_ = Diagonal::Sigmas(lc_super_sigmas);

    // IMU noise models
    vel_prior_noise_  = Isotropic::Sigma(3, imu_cfg.velocity_prior_sigma);
    bias_prior_noise_ = Isotropic::Sigma(6, imu_cfg.bias_prior_sigma);
    bias_rw_noise_    = Isotropic::Sigma(6, std::max(1e-9, imu_cfg.bias_prior_sigma));

    // Fixed-lag smoother
    fixed_lag_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(fixed_lag_s);

    // Build IMU preintegration params once (reused per frame)
    if (enable_imu_) {
        imu_preint_params_ = gtsam::PreintegrationCombinedParams::MakeSharedU(imu_cfg.gravity_mps2);
        imu_preint_params_->setAccelerometerCovariance(
            imu_cfg.acc_noise_sigma * imu_cfg.acc_noise_sigma * Eigen::Matrix3d::Identity());
        imu_preint_params_->setGyroscopeCovariance(
            imu_cfg.gyro_noise_sigma * imu_cfg.gyro_noise_sigma * Eigen::Matrix3d::Identity());
        imu_preint_params_->setIntegrationCovariance(
            imu_cfg.integration_sigma * imu_cfg.integration_sigma * Eigen::Matrix3d::Identity());
        imu_preint_params_->setBiasAccCovariance(
            imu_cfg.bias_acc_rw_sigma * imu_cfg.bias_acc_rw_sigma * Eigen::Matrix3d::Identity());
        imu_preint_params_->setBiasOmegaCovariance(
            imu_cfg.bias_gyro_rw_sigma * imu_cfg.bias_gyro_rw_sigma * Eigen::Matrix3d::Identity());
        imu_preint_params_->setBiasAccOmegaInit(
            1e-10 * Eigen::Matrix<double, 6, 6>::Identity());
    }

    pose_by_idx[0] = Matrix4d::Identity();
    pose_uncertainty_by_idx[0] = 0.0;
    initBenchmarkLogs(benchmark_log_dir);

    GMS_INFO("[smoother] initialized: IncrementalFixedLagSmoother (lag=%.2f s, imu=%s)",
             fixed_lag_s, enable_imu_ ? "on" : "off");
}

void FixedLagBackend::initBenchmarkLogs(const std::string& log_dir)
{
    if (log_dir.empty()) {
        return;
    }

    try {
        std::filesystem::create_directories(log_dir);
        benchmark_smoother_csv_.open(log_dir + "/smoother_optimization.csv",
                                     std::ios::out);
        if (!benchmark_smoother_csv_) {
            GMS_WARN("[benchmark] failed to open smoother_optimization.csv in %s",
                     log_dir.c_str());
            return;
        }

        benchmark_smoother_csv_ << std::fixed << std::setprecision(9);
        benchmark_smoother_csv_
            << "stamp,curr_idx,factors,values,update_ms,estimate_ms,total_ms,"
            << "success,note\n";
        benchmark_logs_enabled_ = true;
    } catch (const std::exception& e) {
        GMS_WARN("[benchmark] failed to initialize smoother timing log in %s: %s",
                 log_dir.c_str(), e.what());
    }
}

void FixedLagBackend::logOptimizationTiming(
    double stamp_sec, int curr_idx,
    std::size_t factors, std::size_t values,
    double update_ms, double estimate_ms, double total_ms,
    bool success, const std::string& note)
{
    if (!benchmark_logs_enabled_) {
        return;
    }

    std::string safe_note = note;
    std::replace(safe_note.begin(), safe_note.end(), ',', ';');
    std::replace(safe_note.begin(), safe_note.end(), '\n', ' ');
    benchmark_smoother_csv_
        << stamp_sec << ','
        << curr_idx << ','
        << factors << ','
        << values << ','
        << update_ms << ','
        << estimate_ms << ','
        << total_ms << ','
        << (success ? 1 : 0) << ','
        << safe_note << '\n';
}

// -----------------------------------------------------------------------
// Static conversion helpers
// -----------------------------------------------------------------------

gtsam::Pose3 FixedLagBackend::pose3FromMatrix(const Matrix4d& T) {
    return Pose3(Rot3(T.block<3,3>(0,0)),
                 Point3(T(0,3), T(1,3), T(2,3)));
}

Matrix4d FixedLagBackend::matrixFromPose3(const gtsam::Pose3& p) {
    Matrix4d T = Matrix4d::Identity();
    T.block<3,3>(0,0) = p.rotation().matrix();
    T.block<3,1>(0,3) = p.translation();
    return T;
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

void FixedLagBackend::resetNewData() {
    new_factors_.resize(0);
    new_values_.clear();
    new_timestamps_.clear();
}

gtsam::NonlinearFactorGraph FixedLagBackend::filterStaleFactors(
        const gtsam::NonlinearFactorGraph& factors,
        const gtsam::Values& new_vals,
        double t_latest,
        bool require_active_keys) const {

    std::lock_guard<std::mutex> lk(graph_lock);
    const double t_cutoff = t_latest - fixed_lag_s * 0.85;
    gtsam::NonlinearFactorGraph clean;
    int n_dropped = 0;

    for (std::size_t i = 0; i < factors.size(); ++i) {
        const auto& f = factors.at(i);
        if (!f) { continue; }
        bool stale = false;
        for (const gtsam::Key key : f->keys()) {
            if (new_vals.exists(key)) { continue; }
            if (require_active_keys && active_gtsam_keys_.count(key) == 0) {
                stale = true;
                break;
            }
            const auto idx = static_cast<int>(Symbol(key).index());
            auto it = key_t_sec.find(idx);
            if (it == key_t_sec.end() || it->second < t_cutoff) {
                stale = true;
                break;
            }
        }
        if (stale) {
            ++n_dropped;
        } else {
            clean.push_back(f);
        }
    }

    if (n_dropped > 0) {
        GMS_WARN("[smoother] dropped %d stale factor(s) referencing keys outside the lag window",
                 n_dropped);
    }
    return clean;
}

void FixedLagBackend::rebuildSmootherCore(int anchor_idx, const Matrix4d& anchor_pose,
                                          double t_sec, const char* reason,
                                          bool zero_imu_at_anchor) {
    Vector3d v_anchor = Vector3d::Zero();
    Eigen::Matrix<double,6,1> bvec_anchor = Eigen::Matrix<double,6,1>::Zero();
    if (enable_imu_ && !zero_imu_at_anchor) {
        {
            auto it = velocity_by_idx.find(anchor_idx);
            if (it != velocity_by_idx.end()) v_anchor = it->second;
        }
        {
            auto it = bias_by_idx.find(anchor_idx);
            if (it != bias_by_idx.end()) bvec_anchor = it->second;
        }
    }

    fixed_lag_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(fixed_lag_s);

    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;
    gtsam::FixedLagSmootherKeyTimestampMap timestamps;

    const gtsam::Key key0 = X(anchor_idx);
    const Pose3 p3 = pose3FromMatrix(anchor_pose);
    factors.addPrior(key0, p3, prior_noise_);
    values.insert(key0, p3);
    timestamps[key0] = t_sec;

    if (enable_imu_) {
        const gtsam::Key kv = keyV(anchor_idx);
        const gtsam::Key kb = keyB(anchor_idx);
        const ConstantBias bias(bvec_anchor.head<3>(), bvec_anchor.tail<3>());

        values.insert(kv, v_anchor);
        values.insert(kb, bias);
        timestamps[kv] = t_sec;
        timestamps[kb] = t_sec;

        factors.addPrior(kv, v_anchor, vel_prior_noise_);
        factors.addPrior(kb, bias, bias_prior_noise_);
    }

    fixed_lag_->update(factors, values, timestamps);

    key_t_sec.clear();
    key_t_sec[anchor_idx] = t_sec;
    pose_by_idx.clear();
    pose_by_idx[anchor_idx] = anchor_pose;
    pose_uncertainty_by_idx.clear();
    pose_uncertainty_by_idx[anchor_idx] = 0.0;
    velocity_by_idx.clear();
    bias_by_idx.clear();
    if (enable_imu_) {
        velocity_by_idx[anchor_idx] = v_anchor;
        bias_by_idx[anchor_idx] = bvec_anchor;
    }

    pose = anchor_pose;
    latest_key_idx = anchor_idx;
    inserted_pose_keys_ = {anchor_idx};
    active_gtsam_keys_.clear();
    active_gtsam_keys_.insert(key0);
    if (enable_imu_) {
        active_gtsam_keys_.insert(keyV(anchor_idx));
        active_gtsam_keys_.insert(keyB(anchor_idx));
    }

    solve_queue_.clear();
    resetNewData();
    consecutive_failures_ = 0;

    GMS_WARN("[smoother] RESET fixed-lag smoother (%s); re-anchored at X(%d)",
             reason, anchor_idx);
}

void FixedLagBackend::rebuildSmoother(int anchor_idx) {
    Matrix4d anchor_pose = Matrix4d::Identity();
    double t_sec = 0.0;
    {
        std::lock_guard<std::mutex> lk(graph_lock);
        auto it = pose_by_idx.find(anchor_idx);
        if (it != pose_by_idx.end()) {
            anchor_pose = it->second;
        } else {
            anchor_pose = pose;
        }
        auto tit = key_t_sec.find(anchor_idx);
        t_sec = (tit != key_t_sec.end()) ? tit->second : 0.0;
    }

    std::scoped_lock lk(new_data_lock_, graph_lock);
    rebuildSmootherCore(anchor_idx, anchor_pose, t_sec,
                        "consecutive GTSAM failures", false);
}

void FixedLagBackend::scheduleReanchorToGt(int anchor_idx, const Matrix4d& T_odom,
                                           double t_sec) {
    std::lock_guard<std::mutex> lk(reanchor_scheduled_mu_);
    reanchor_scheduled_ = true;
    reanchor_anchor_idx_ = anchor_idx;
    reanchor_pose_ = T_odom;
    reanchor_t_sec_ = t_sec;
}

void FixedLagBackend::applyScheduledReanchorIfNeeded() {
    std::unique_lock<std::mutex> rk(reanchor_scheduled_mu_);
    if (!reanchor_scheduled_) {
        return;
    }
    reanchor_scheduled_ = false;
    const int aid = reanchor_anchor_idx_;
    const Matrix4d T = reanchor_pose_;
    const double ts = reanchor_t_sec_;
    rk.unlock();

    std::scoped_lock gk(new_data_lock_, graph_lock);
    rebuildSmootherCore(aid, T, ts, "GT re-anchor after submap traj divergence",
                        true);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

bool FixedLagBackend::initialize(const Matrix4d& init_pose, const rclcpp::Time& stamp) {
    if (initialized_) return true;

    try {
        std::scoped_lock lk(new_data_lock_, graph_lock);
        pose = init_pose;

        const gtsam::Key key0 = X(0);
        const double t0 = stampToSec(stamp);
        const Pose3 p0 = pose3FromMatrix(init_pose);

        new_factors_.addPrior(key0, p0, prior_noise_);
        new_values_.insert(key0, p0);
        new_timestamps_[key0] = t0;

        if (enable_imu_) {
            const gtsam::Key kv0 = keyV(0);
            const gtsam::Key kb0 = keyB(0);
            const Vector3d v0 = Vector3d::Zero();
            const ConstantBias b0 = zeroBias();

            new_values_.insert(kv0, v0);
            new_values_.insert(kb0, b0);
            new_timestamps_[kv0] = t0;
            new_timestamps_[kb0] = t0;

            new_factors_.addPrior(kv0, v0, vel_prior_noise_);
            new_factors_.addPrior(kb0, b0, bias_prior_noise_);
        }

        fixed_lag_->update(new_factors_, new_values_, new_timestamps_);
        resetNewData();

        pose_by_idx[0] = init_pose;
        key_t_sec[0] = t0;
        inserted_pose_keys_ = {0};
        active_gtsam_keys_.clear();
        active_gtsam_keys_.insert(key0);

        if (enable_imu_) {
            velocity_by_idx[0] = Vector3d::Zero();
            bias_by_idx[0] = Eigen::Matrix<double,6,1>::Zero();
            active_gtsam_keys_.insert(keyV(0));
            active_gtsam_keys_.insert(keyB(0));
        }

        initialized_ = true;
        GMS_INFO("[smoother] fixed-lag smoother initialized with prior X(0)");
        return true;
    } catch (const std::exception& e) {
        GMS_WARN("[smoother] init failed: %s", e.what());
        return false;
    }
}

std::optional<FixedLagBackend::ImuPrediction>
FixedLagBackend::predictWithImu(
    int prev_idx,
    const std::vector<std::tuple<double, Vector3d, Vector3d>>& imu_measurements) const
{
    if (!enable_imu_ || imu_measurements.empty() || !imu_preint_params_) {
        return std::nullopt;
    }

    Matrix4d T_prev = Matrix4d::Identity();
    Vector3d v_prev = Vector3d::Zero();
    Eigen::Matrix<double,6,1> b_prev = Eigen::Matrix<double,6,1>::Zero();
    {
        std::lock_guard<std::mutex> lk(graph_lock);
        const auto pose_it = pose_by_idx.find(prev_idx);
        if (pose_it == pose_by_idx.end()) {
            return std::nullopt;
        }
        T_prev = pose_it->second;

        const auto vel_it = velocity_by_idx.find(prev_idx);
        if (vel_it != velocity_by_idx.end()) {
            v_prev = vel_it->second;
        }
        const auto bias_it = bias_by_idx.find(prev_idx);
        if (bias_it != bias_by_idx.end()) {
            b_prev = bias_it->second;
        }
    }

    const ConstantBias prev_bias(b_prev.head<3>(), b_prev.tail<3>());
    gtsam::PreintegratedCombinedMeasurements preint(imu_preint_params_, prev_bias);

    int n_imu = 0;
    for (const auto& [dt, acc, gyro] : imu_measurements) {
        if (dt <= 0.0) {
            continue;
        }
        preint.integrateMeasurement(acc, gyro, dt);
        ++n_imu;
    }
    if (n_imu == 0) {
        return std::nullopt;
    }

    const gtsam::NavState prev_state(pose3FromMatrix(T_prev), v_prev);
    const gtsam::NavState predicted = preint.predict(prev_state, prev_bias);

    ImuPrediction out;
    out.pose = matrixFromPose3(predicted.pose());
    out.velocity = predicted.velocity();
    out.bias = b_prev;
    out.sample_count = n_imu;
    return out;
}

bool FixedLagBackend::addFrame(int prev_idx, int curr_idx,
                               const rclcpp::Time& stamp,
                               const Matrix4d& predicted_pose,
                               const Matrix4d* gt_rel_mat,
                               const std::vector<std::tuple<double, Vector3d, Vector3d>>* imu_measurements,
                               bool add_external_odometry_factor,
                               const Vector3d* initial_velocity,
                               const Eigen::Matrix<double,6,1>* initial_bias) {
    if (!initialize(predicted_pose, stamp)) return false;

    if (prev_idx < 0) {
        std::lock_guard<std::mutex> lk(graph_lock);
        key_t_sec[curr_idx] = stampToSec(stamp);
        return true;
    }

    const double t_sec = stampToSec(stamp);
    int factor_prev_idx = prev_idx;
    bool prev_exists = false;
    Matrix4d fallback_rel = Matrix4d::Identity();
    {
        std::lock_guard<std::mutex> lk(graph_lock);
        prev_exists = pose_by_idx.count(factor_prev_idx) > 0;
        if (!prev_exists && latest_key_idx >= 0 &&
            pose_by_idx.count(latest_key_idx) > 0 &&
            latest_key_idx != curr_idx) {
            factor_prev_idx = latest_key_idx;
            prev_exists = true;
            fallback_rel = pose_by_idx.at(factor_prev_idx).inverse() * predicted_pose;
        }
    }

    const gtsam::Key key_prev = X(factor_prev_idx);
    const gtsam::Key key_curr = X(curr_idx);

    std::unique_lock<std::mutex> new_data_lk(new_data_lock_);

    // ---- Pose factor ----
    if (add_external_odometry_factor &&
        prev_exists && factor_prev_idx == prev_idx && gt_rel_mat != nullptr) {
        new_factors_.push_back(
            gtsam::BetweenFactor<Pose3>(key_prev, key_curr,
                                        pose3FromMatrix(*gt_rel_mat),
                                        gt_factor_noise_));
    } else if (add_external_odometry_factor && prev_exists) {
        new_factors_.push_back(
            gtsam::BetweenFactor<Pose3>(key_prev, key_curr,
                                        pose3FromMatrix(fallback_rel),
                                        factor_prev_idx == prev_idx
                                            ? odom_noise_lost_
                                            : odom_noise_lost_));
        if (factor_prev_idx != prev_idx) {
            GMS_WARN_THROTTLE(2.0,
                "[smoother] missing predecessor X(%d); linking X(%d)->X(%d) instead",
                prev_idx, factor_prev_idx, curr_idx);
        }
    } else if (prev_exists) {
        GMS_INFO_THROTTLE(5.0,
            "[smoother] external odometry factor disabled; X(%d) waits for D2D/other factors",
            curr_idx);
    } else {
        new_factors_.addPrior(key_curr, pose3FromMatrix(predicted_pose), prior_noise_);
        GMS_WARN_THROTTLE(2.0,
            "[smoother] missing predecessor X(%d); re-anchoring X(%d) with prior",
            prev_idx, curr_idx);
    }

    new_values_.insert(key_curr, pose3FromMatrix(predicted_pose));
    new_timestamps_[key_curr] = t_sec;

    Vector3d vel_wb = Vector3d::Zero();
    Eigen::Matrix<double,6,1> bias_vec_wb =
        Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> curr_bias_guess_wb =
        Eigen::Matrix<double,6,1>::Zero();

    // ---- Optional IMU preintegration ----
    if (enable_imu_) {
        const gtsam::Key kv_prev = keyV(factor_prev_idx);
        const gtsam::Key kv_curr = keyV(curr_idx);
        const gtsam::Key kb_prev = keyB(factor_prev_idx);
        const gtsam::Key kb_curr = keyB(curr_idx);

        {
            std::lock_guard<std::mutex> lk(graph_lock);
            auto it = bias_by_idx.find(factor_prev_idx);
            if (it != bias_by_idx.end()) bias_vec_wb = it->second;
        }
        curr_bias_guess_wb = initial_bias != nullptr
            ? *initial_bias
            : bias_vec_wb;
        const ConstantBias prev_bias(bias_vec_wb.head<3>(), bias_vec_wb.tail<3>());

        auto preint = boost::make_shared<gtsam::PreintegratedCombinedMeasurements>(
            imu_preint_params_, prev_bias);
        int n_imu = 0;
        if (prev_exists && imu_measurements != nullptr) {
            for (const auto& [dt, acc, gyro] : *imu_measurements) {
                if (dt <= 0.0) continue;
                preint->integrateMeasurement(acc, gyro, dt);
                ++n_imu;
            }
        }

        const bool has_imu_link =
            prev_exists && (factor_prev_idx == prev_idx) && (n_imu > 0);
        if (has_imu_link) {
            new_factors_.push_back(
                gtsam::CombinedImuFactor(key_prev, kv_prev,
                                         key_curr, kv_curr,
                                         kb_prev, kb_curr,
                                         *preint));
        }

        // Velocity initial guess
        const double t_prev = [&]{
            std::lock_guard<std::mutex> lk(graph_lock);
            auto it = key_t_sec.find(factor_prev_idx);
            return (it != key_t_sec.end()) ? it->second : (t_sec - 1e-3);
        }();
        const double dt_pose = std::max(1e-3, t_sec - t_prev);

        {
            std::lock_guard<std::mutex> lk(graph_lock);
            auto it = pose_by_idx.find(factor_prev_idx);
            if (it != pose_by_idx.end()) {
                vel_wb = (predicted_pose.block<3,1>(0,3) - it->second.block<3,1>(0,3)) / dt_pose;
            } else {
                auto vit = velocity_by_idx.find(factor_prev_idx);
                if (vit != velocity_by_idx.end()) vel_wb = vit->second;
            }
        }
        if (initial_velocity != nullptr) {
            vel_wb = *initial_velocity;
        }

        new_values_.insert(kv_curr, vel_wb);
        new_values_.insert(
            kb_curr,
            ConstantBias(curr_bias_guess_wb.head<3>(), curr_bias_guess_wb.tail<3>()));
        new_timestamps_[kv_curr] = t_sec;
        new_timestamps_[kb_curr] = t_sec;

        if (!has_imu_link) {
            new_factors_.addPrior(kv_curr, vel_wb, vel_prior_noise_);
            new_factors_.addPrior(kb_curr, prev_bias, bias_prior_noise_);
            GMS_WARN_THROTTLE(2.0,
                "[smoother] weakly-anchored V/B at X(%d) (imu_link=%s, samples=%d)",
                curr_idx, has_imu_link ? "yes" : "no", n_imu);
        }

        GMS_DEBUG("[smoother] IMU preintegration at X(%d): samples=%d", curr_idx, n_imu);
    }

    // Drop stale factors
    new_factors_ = filterStaleFactors(new_factors_, new_values_, t_sec, false);

    // Snapshot and enqueue
    SolveBatch batch;
    batch.curr_idx  = curr_idx;
    batch.factors    = std::move(new_factors_);
    batch.values     = std::move(new_values_);
    batch.timestamps = std::move(new_timestamps_);

    new_factors_ = gtsam::NonlinearFactorGraph();
    new_values_  = gtsam::Values();
    new_timestamps_.clear();

    if (!solve_queue_.tryPush(std::move(batch))) {
        // Queue full: keep staged so it batches with the next frame
        new_factors_    = std::move(batch.factors);
        new_values_     = std::move(batch.values);
        new_timestamps_ = std::move(batch.timestamps);
        deferred_batches.fetch_add(1, std::memory_order_relaxed);
    }
    new_data_lk.unlock();

    {
        std::lock_guard<std::mutex> lk(graph_lock);
        pose = predicted_pose;
        latest_key_idx = curr_idx;
        pose_by_idx[curr_idx] = predicted_pose;
        key_t_sec[curr_idx] = t_sec;
        if (enable_imu_) {
            velocity_by_idx[curr_idx] = vel_wb;
            bias_by_idx[curr_idx] = curr_bias_guess_wb;
        }
    }
    return true;
}

void FixedLagBackend::stageFactor(const gtsam::NonlinearFactor::shared_ptr& factor) {
    std::lock_guard<std::mutex> lk(new_data_lock_);
    new_factors_.push_back(factor);
}

bool FixedLagBackend::flushStagedFactors() {
    std::unique_lock<std::mutex> new_data_lk(new_data_lock_);
    if (new_factors_.empty()) return false;

    SolveBatch batch;
    {
        std::lock_guard<std::mutex> lk(graph_lock);
        batch.curr_idx = latest_key_idx;
    }
    batch.factors   = std::move(new_factors_);
    batch.values    = std::move(new_values_);
    batch.timestamps = std::move(new_timestamps_);

    new_factors_ = gtsam::NonlinearFactorGraph();
    new_values_  = gtsam::Values();
    new_timestamps_.clear();

    if (!solve_queue_.tryPush(std::move(batch))) {
        new_factors_    = std::move(batch.factors);
        new_values_     = std::move(batch.values);
        new_timestamps_ = std::move(batch.timestamps);
        return false;
    }
    return true;
}

bool FixedLagBackend::isInLagWindow(int idx) const {
    std::lock_guard<std::mutex> lk(graph_lock);
    auto it = key_t_sec.find(idx);
    if (it == key_t_sec.end()) return false;
    auto lit = key_t_sec.find(latest_key_idx);
    const double t_latest = (lit != key_t_sec.end()) ? lit->second : 0.0;
    return (t_latest - it->second) <= fixed_lag_s * 1.1;
}

// -----------------------------------------------------------------------
// Backend solve loop (runs on dedicated thread)
// -----------------------------------------------------------------------

void FixedLagBackend::backendLoop(const std::atomic<bool>& shutdown) {
    while (!shutdown.load(std::memory_order_acquire)) {
        applyScheduledReanchorIfNeeded();

        auto maybe_batch = solve_queue_.popWithTimeout(std::chrono::milliseconds(100));
        if (!maybe_batch.has_value()) continue;

        SolveBatch& batch = maybe_batch.value();
        const int curr_idx = batch.curr_idx;

        try {
            double latest_t = -1.0;
            double batch_t = -1.0;
            {
                std::lock_guard<std::mutex> lk(graph_lock);
                auto lit = key_t_sec.find(latest_key_idx);
                if (lit != key_t_sec.end()) {
                    latest_t = lit->second;
                }
                auto bit = key_t_sec.find(curr_idx);
                if (bit != key_t_sec.end()) {
                    batch_t = bit->second;
                }
            }
            if (!batch.timestamps.empty()) {
                for (const auto& [key, t] : batch.timestamps) {
                    (void)key;
                    batch_t = std::max(batch_t, t);
                }
            }
            if (latest_t >= 0.0 && batch_t >= 0.0 &&
                (latest_t - batch_t) > fixed_lag_s * 0.75) {
                // The callback thread has already propagated the live pose.
                // Feeding old variables into IncrementalFixedLagSmoother after
                // they are near/immediately beyond the lag boundary can trigger
                // fragile leaf marginalization paths in GTSAM.
                GMS_WARN_THROTTLE(
                    1.0,
                    "[smoother] dropping stale backend batch X(%d): "
                    "latest_t - batch_t = %.2fs exceeds %.2fs guard",
                    curr_idx, latest_t - batch_t, fixed_lag_s * 0.75);
                continue;
            }
            if (latest_t >= 0.0) {
                // Registration results are produced asynchronously and can sit
                // behind frame updates. Re-check their keys at solve time so
                // GTSAM never receives factors for variables it may already
                // have marginalized from the fixed-lag smoother.
                batch.factors = filterStaleFactors(batch.factors,
                                                   batch.values,
                                                   latest_t,
                                                   true);
            }

            if (batch.factors.empty() && batch.values.empty()) {
                continue;
            }

            const std::size_t n_factors = batch.factors.size();
            const std::size_t n_values = batch.values.size();
            const double batch_stamp = [&] {
                const gtsam::Key key_curr_for_time = X(curr_idx);
                const auto it = batch.timestamps.find(key_curr_for_time);
                if (it != batch.timestamps.end()) {
                    return it->second;
                }
                std::lock_guard<std::mutex> lk(graph_lock);
                const auto kt = key_t_sec.find(curr_idx);
                return (kt != key_t_sec.end()) ? kt->second : -1.0;
            }();
            const auto solve_t0 = std::chrono::steady_clock::now();
            gtsam::Values estimate;
            double update_ms = 0.0;
            double estimate_ms = 0.0;

            const auto update_t0 = std::chrono::steady_clock::now();
            fixed_lag_->update(batch.factors, batch.values, batch.timestamps);
            const auto update_t1 = std::chrono::steady_clock::now();
            update_ms = std::chrono::duration<double, std::milli>(
                update_t1 - update_t0).count();

            const auto estimate_t0 = std::chrono::steady_clock::now();
            estimate = fixed_lag_->calculateEstimate();
            const auto estimate_t1 = std::chrono::steady_clock::now();
            estimate_ms = std::chrono::duration<double, std::milli>(
                estimate_t1 - estimate_t0).count();
            const double total_ms = std::chrono::duration<double, std::milli>(
                estimate_t1 - solve_t0).count();
            GMS_INFO("[timing][smoother] X(%d) factors=%zu values=%zu "
                     "update_ms=%.3f estimate_ms=%.3f total_ms=%.3f",
                     curr_idx, n_factors, n_values,
                     update_ms, estimate_ms, total_ms);
            logOptimizationTiming(batch_stamp, curr_idx, n_factors, n_values,
                                  update_ms, estimate_ms, total_ms, true,
                                  "fixed_lag_update");

            const gtsam::Key key_curr = X(curr_idx);
            bool has_curr = estimate.exists(key_curr);

            Matrix4d new_pose;
            if (has_curr) {
                new_pose = matrixFromPose3(estimate.at<Pose3>(key_curr));
            } else {
                std::lock_guard<std::mutex> lk(graph_lock);
                auto it = pose_by_idx.find(curr_idx);
                new_pose = (it != pose_by_idx.end()) ? it->second : pose;
                GMS_WARN_THROTTLE(2.0,
                    "[smoother] estimate missing X(%d) after update; keeping previous pose",
                    curr_idx);
            }

            {
                std::lock_guard<std::mutex> lk(graph_lock);
                const double latest_t = [&]{
                    auto it = key_t_sec.find(curr_idx);
                    return (it != key_t_sec.end()) ? it->second : -1.0;
                }();

                active_gtsam_keys_.clear();
                for (const gtsam::Key key : estimate.keys()) {
                    active_gtsam_keys_.insert(key);
                }

                inserted_pose_keys_.insert(curr_idx);

                for (const auto& [k_idx, k_t] : key_t_sec) {
                    if (latest_t >= 0.0 && (latest_t - k_t) > fixed_lag_s * 1.2) continue;

                    const gtsam::Key kx = X(k_idx);
                    if (estimate.exists(kx)) {
                        try {
                            pose_by_idx[k_idx] = matrixFromPose3(estimate.at<Pose3>(kx));
                        } catch (...) {}
                    }

                    if (enable_imu_) {
                        const gtsam::Key kv = keyV(k_idx);
                        if (estimate.exists(kv)) {
                            try {
                                velocity_by_idx[k_idx] = estimate.at<Vector3d>(kv);
                            } catch (...) {}
                        }
                        const gtsam::Key kb = keyB(k_idx);
                        if (estimate.exists(kb)) {
                            try {
                                const auto b = estimate.at<ConstantBias>(kb);
                                Eigen::Matrix<double,6,1> bvec;
                                bvec.head<3>() = b.accelerometer();
                                bvec.tail<3>() = b.gyroscope();
                                bias_by_idx[k_idx] = bvec;
                            } catch (...) {}
                        }
                    }
                }

                pose = new_pose;
                latest_key_idx = curr_idx;
                pose_by_idx[curr_idx] = new_pose;
                // Marginal only for X(curr_idx): full-window marginals spammed
                // GTSAM on ill-conditioned keys.
                if (has_curr) {
                    try {
                        const auto cov = fixed_lag_->marginalCovariance(key_curr);
                        if (cov.rows() >= 6 && cov.cols() >= 6 && cov.allFinite()) {
                            pose_uncertainty_by_idx[curr_idx] = cov.diagonal().sum();
                        } else {
                            pose_uncertainty_by_idx[curr_idx] =
                                std::numeric_limits<double>::infinity();
                        }
                    } catch (...) {
                        pose_uncertainty_by_idx[curr_idx] =
                            std::numeric_limits<double>::infinity();
                    }
                }

                // Trim old entries
                const int cutoff = curr_idx - pose_history_keep_;
                for (auto it = pose_by_idx.begin(); it != pose_by_idx.end(); ) {
                    if (it->first < cutoff) {
                        pose_uncertainty_by_idx.erase(it->first);
                        velocity_by_idx.erase(it->first);
                        bias_by_idx.erase(it->first);
                        it = pose_by_idx.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            consecutive_failures_ = 0;
            GMS_DEBUG("[smoother] GTSAM backend solved at X(%d)", curr_idx);

        } catch (const std::exception& e) {
            ++consecutive_failures_;
            GMS_WARN("[smoother] GTSAM backend failed at X(%d): %s", curr_idx, e.what());

            if (consecutive_failures_ >= kMaxConsecutiveFailures) {
                try {
                    rebuildSmoother(curr_idx);
                    consecutive_failures_ = 0;
                } catch (const std::exception& reset_err) {
                    GMS_ERROR("[smoother] reset also failed: %s", reset_err.what());
                }
            }
        }
    }
}

} // namespace gmmslam
