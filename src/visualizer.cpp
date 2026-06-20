#include "gmmslam/visualizer.hpp"
#include "gmmslam/rclcpp_logging.hpp"
#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/registration_manager.hpp"
#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/ros_helpers.hpp"
#include "gmmslam/util/gmm_utils.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <set>

namespace gmmslam {

namespace {
constexpr double kTwoPi = 6.28318530717958647692;
} // namespace

// =====================================================================
// Constructor
// =====================================================================

Visualizer::Visualizer(FixedLagBackend& smoother,
                       RegistrationManager& registration,
                       GlobalPoseGraph* global_graph,
                       const std::string& odom_frame,
                       const std::string& base_frame,
                       const std::string& map_frame,
                       const VisualizationConfig& vis_cfg,
                       Publishers publishers)
    : smoother_(smoother)
    , registration_(registration)
    , global_graph_(global_graph)
    , odom_frame_(odom_frame)
    , base_frame_(base_frame)
    , map_frame_(map_frame)
    , gmm_marker_sigma_(vis_cfg.gmm_marker_sigma)
    , global_gmm_markers_enable_(vis_cfg.global_gmm_markers_enable)
    , global_gmm_publish_period_s_(vis_cfg.global_gmm_publish_period_s)
    , output_pose_lpf_cutoff_hz_(vis_cfg.output_pose_lpf_cutoff_hz)
    , map_cloud_publish_period_s_(vis_cfg.map_cloud_publish_hz > 1e-6
                                      ? 1.0 / vis_cfg.map_cloud_publish_hz
                                      : 9999.0)
    , map_cloud_max_chunks_(std::max(1, vis_cfg.map_cloud_max_chunks))
    , global_map_cloud_enable_(vis_cfg.global_map_cloud_enable)
    , global_map_cloud_publish_period_s_(vis_cfg.global_map_cloud_publish_hz > 1e-6
                                             ? 1.0 / vis_cfg.global_map_cloud_publish_hz
                                             : 9999.0)
    , global_map_cloud_voxel_size_m_(std::max(0.0, vis_cfg.global_map_cloud_voxel_size_m))
    , prune_debug_markers_enable_(vis_cfg.prune_debug_markers_enable)
    , prune_debug_max_markers_(std::max(0, vis_cfg.prune_debug_max_markers))
    , pubs_(std::move(publishers))
{
    path_.header.frame_id = odom_frame_;
}

// =====================================================================
// Output pose low-pass filter
// =====================================================================

Matrix4d Visualizer::filterOutputPose(const Matrix4d& T,
                                      const rclcpp::Time& stamp) {
    // Apply map->odom correction so the LPF (and its output) lives in map frame.
    Matrix4d T_in = T;
    if (global_graph_ && global_graph_->enable && !map_frame_.empty()) {
        const auto corr = global_graph_->getMapOdomCorrection();
        if (corr.has_value()) {
            T_in = corr.value() * T;
        }
    }

    if (output_pose_lpf_cutoff_hz_ <= 0.0 || !T_in.allFinite()) {
        output_pose_filter_initialized_ = false;
        return T_in;
    }

    if (!output_pose_filter_initialized_ ||
        output_pose_filter_stamp_.nanoseconds() == 0) {
        output_pose_filtered_ = T_in;
        output_pose_filter_stamp_ = stamp;
        output_pose_filter_initialized_ = true;
        return output_pose_filtered_;
    }

    const double dt = (stamp - output_pose_filter_stamp_).seconds();
    if (dt <= 0.0 || !std::isfinite(dt)) {
        output_pose_filtered_ = T_in;
        output_pose_filter_stamp_ = stamp;
        return output_pose_filtered_;
    }

    const double tau = 1.0 / (kTwoPi * output_pose_lpf_cutoff_hz_);
    const double alpha = std::clamp(dt / (tau + dt), 0.0, 1.0);

    const Vector3d p_prev = output_pose_filtered_.block<3, 1>(0, 3);
    const Vector3d p_curr = T_in.block<3, 1>(0, 3);

    Eigen::Quaterniond q_prev(output_pose_filtered_.block<3, 3>(0, 0));
    Eigen::Quaterniond q_curr(T_in.block<3, 3>(0, 0));
    q_prev.normalize();
    q_curr.normalize();

    Matrix4d filtered = Matrix4d::Identity();
    filtered.block<3, 3>(0, 0) =
        q_prev.slerp(alpha, q_curr).normalized().toRotationMatrix();
    filtered.block<3, 1>(0, 3) = p_prev + alpha * (p_curr - p_prev);

    output_pose_filtered_ = filtered;
    output_pose_filter_stamp_ = stamp;
    return output_pose_filtered_;
}

// =====================================================================
// Pose-only publish (called from main thread on every frame)
// =====================================================================

void Visualizer::publishPoseOnly(const Matrix4d& T, const rclcpp::Time& stamp) {
    const auto ts = poseToTransformStamped(T, stamp, odom_frame_, base_frame_);
    if (pubs_.send_transform) {
        pubs_.send_transform(ts);
    }

    // Broadcast map->odom whenever a global graph correction is available.
    if (global_graph_ && global_graph_->enable && !map_frame_.empty()) {
        const auto corr = global_graph_->getMapOdomCorrection();
        if (corr.has_value()) {
            const auto map_odom_ts = poseToTransformStamped(
                corr.value(), stamp, map_frame_, odom_frame_);
            if (pubs_.send_transform) {
                pubs_.send_transform(map_odom_ts);
            }
        }
    }

    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id  = base_frame_;
    odom.pose.pose       = poseToPoseStamped(T, stamp, odom_frame_).pose;
    if (pubs_.odom) {
        pubs_.odom->publish(odom);
    }
}

void Visualizer::publishPoseLpf(const Matrix4d& T, const rclcpp::Time& stamp) {
    if (output_pose_lpf_cutoff_hz_ <= 0.0 || !pubs_.odom_lpf) {
        return;
    }

    // T is in map frame when a global correction has been applied by filterOutputPose.
    bool has_correction = false;
    if (global_graph_ && global_graph_->enable && !map_frame_.empty()) {
        has_correction = global_graph_->getMapOdomCorrection().has_value();
    }
    const std::string& frame = has_correction ? map_frame_ : odom_frame_;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = frame;
    odom.child_frame_id  = base_frame_ + "_lpf";
    odom.pose.pose       = poseToPoseStamped(T, stamp, frame).pose;
    pubs_.odom_lpf->publish(odom);
}

// =====================================================================
// Enqueue a frame for the vis thread (drop if full)
// =====================================================================

void Visualizer::enqueueFrame(const rclcpp::Time& stamp,
                              std::shared_ptr<const Eigen::MatrixXf> pts,
                              int frame_count,
                              const Matrix4d& capture_pose,
                              int smoother_pose_key) {
    if (!pts) {
        return;
    }
    VisFrame frame;
    frame.stamp             = stamp;
    frame.points            = pts;
    frame.frame_count       = frame_count;
    frame.capture_pose      = capture_pose;
    frame.smoother_pose_key = smoother_pose_key;
    vis_queue_.tryPush(std::move(frame));
}

// =====================================================================
// Background vis thread entry point
// =====================================================================

void Visualizer::visLoop(const std::atomic<bool>& shutdown) {
    while (!shutdown.load(std::memory_order_relaxed)) {
        const rclcpp::Time header_stamp =
            last_vis_stamp_.nanoseconds() == 0 ? gmmslam::now() : last_vis_stamp_;
        maybePublishMapCloud(header_stamp);
        maybePublishGlobalMapCloud(header_stamp);
        publishPruneDebugMarkers(header_stamp);

        auto opt = vis_queue_.popWithTimeout(std::chrono::milliseconds(100));
        if (!opt.has_value()) {
            continue;
        }
        try {
            auto& f = opt.value();
            if (!f.points) {
                continue;
            }
            last_vis_stamp_ = f.stamp;
            publishScanProducts(f.stamp, f.points, f.frame_count, f.capture_pose,
                                f.smoother_pose_key);
        } catch (const std::exception& e) {
            GMS_WARN_THROTTLE(2.0, "[vis] thread error: %s", e.what());
        }
    }
}

// =====================================================================
// publishScanProducts
// =====================================================================

void Visualizer::publishScanProducts(const rclcpp::Time& stamp,
                                     std::shared_ptr<const Eigen::MatrixXf> pts,
                                     int frame_count,
                                     const Matrix4d& capture_pose,
                                     int smoother_pose_key) {
    if (!pts) {
        return;
    }
    const Matrix4d& T = capture_pose;

    // --- Path ---
    const auto ps = poseToPoseStamped(T, stamp, odom_frame_);
    path_.header = ps.header;
    path_.poses.push_back(ps);
    if (pubs_.path) {
        pubs_.path->publish(path_);
    }

    const bool latest_cloud_has_subs =
        pubs_.latest_frame_cloud &&
        pubs_.latest_frame_cloud->get_subscription_count() > 0;
    const bool map_cloud_has_subs =
        pubs_.map_cloud && pubs_.map_cloud->get_subscription_count() > 0;

    if (latest_cloud_has_subs) {
        // pts is Nx3 float; stay in float to avoid double↔float round-trip.
        const Eigen::Matrix3f R = T.block<3, 3>(0, 0).cast<float>();
        const Eigen::RowVector3f t =
            T.block<3, 1>(0, 3).cast<float>().transpose();
        const Eigen::MatrixXf pts_w = ((*pts) * R.transpose()).rowwise() + t;
        pubs_.latest_frame_cloud->publish(
            eigenToPc2Rgb(pts_w, stamp, odom_frame_, 255, 0, 0));
    }

    // --- Accumulated map: store lidar-frame points keyed by smoother pose X(k);
    //     map_cloud is rebuilt from current pose_by_idx at map_cloud_publish_hz.
    if (map_cloud_has_subs && smoother_pose_key >= 0 && pts->rows() > 0) {
        MapCloudChunk chunk;
        chunk.pose_key       = smoother_pose_key;
        chunk.points_lidar   = pts;
        std::lock_guard<std::mutex> lk(map_cloud_mutex_);
        while (static_cast<int>(map_cloud_chunks_.size()) >= map_cloud_max_chunks_) {
            map_cloud_chunks_.pop_front();
        }
        map_cloud_chunks_.push_back(std::move(chunk));
    }

    const double now_t = stampToSec(stamp);

    if (pubs_.graph_nodes && pubs_.graph_nodes->get_subscription_count() > 0) {
        publishGraphNodeMarkers(stamp, now_t);
    }

    if (pubs_.global_graph_markers &&
        pubs_.global_graph_markers->get_subscription_count() > 0) {
        publishGlobalGraphMarkers(stamp, now_t);
    }

    if ((pubs_.gmm_markers &&
         pubs_.gmm_markers->get_subscription_count() > 0) ||
        (pubs_.gmm_global_markers &&
         pubs_.gmm_global_markers->get_subscription_count() > 0)) {
        publishGmmMarkers(stamp, T);
    }

    maybePublishMapCloud(stamp);
}

// =====================================================================
// maybePublishMapCloud
// =====================================================================

void Visualizer::maybePublishMapCloud(const rclcpp::Time& header_stamp) {
    if (!pubs_.map_cloud ||
        pubs_.map_cloud->get_subscription_count() == 0) {
        return;
    }

    const double now_t = gmmslam::now().seconds();
    if ((now_t - map_cloud_last_pub_t_) < map_cloud_publish_period_s_) {
        return;
    }

    std::map<int, Matrix4d> pose_snap;
    {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        pose_snap = smoother_.pose_by_idx;
    }

    Eigen::MatrixXf world_pts;
    {
        std::lock_guard<std::mutex> lk(map_cloud_mutex_);
        std::deque<MapCloudChunk> kept;
        std::size_t total_rows = 0;
        for (std::size_t i = 0; i < map_cloud_chunks_.size(); ++i) {
            MapCloudChunk& ch = map_cloud_chunks_[i];
            auto it = pose_snap.find(ch.pose_key);
            if (it != pose_snap.end() && it->second.allFinite() &&
                ch.points_lidar) {
                total_rows +=
                    static_cast<std::size_t>(ch.points_lidar->rows());
                kept.push_back(std::move(ch));
            }
        }
        map_cloud_chunks_ = std::move(kept);

        if (total_rows == 0) {
            map_cloud_last_pub_t_ = now_t;
            return;
        }

        world_pts.resize(static_cast<Eigen::Index>(total_rows), 3);
        Eigen::Index row_out = 0;
        for (const auto& ch : map_cloud_chunks_) {
            auto it = pose_snap.find(ch.pose_key);
            if (it == pose_snap.end() || !it->second.allFinite()) {
                GMS_WARN_THROTTLE(
                    5.0,
                    "[vis] map_cloud chunk pose_key=%d missing from pose snapshot; skip",
                    ch.pose_key);
                continue;
            }
            const Matrix4d& Tw = it->second;
            const Eigen::Matrix3f Rw = Tw.block<3, 3>(0, 0).cast<float>();
            const Eigen::RowVector3f tw =
                Tw.block<3, 1>(0, 3).cast<float>().transpose();
            if (!ch.points_lidar) {
                continue;
            }
            const Eigen::MatrixXf& pl = *ch.points_lidar;
            const Eigen::Index n = pl.rows();
            if (row_out + n > world_pts.rows()) {
                GMS_WARN_THROTTLE(
                    2.0,
                    "[vis] map_cloud row overflow (row_out=%ld n=%ld cap=%ld); truncating",
                    static_cast<long>(row_out), static_cast<long>(n),
                    static_cast<long>(world_pts.rows()));
                break;
            }
            world_pts.block(row_out, 0, n, 3) =
                (pl * Rw.transpose()).rowwise() + tw;
            row_out += n;
        }
        if (row_out <= 0) {
            map_cloud_last_pub_t_ = now_t;
            return;
        }
        world_pts.conservativeResize(row_out, 3);
    }

    pubs_.map_cloud->publish(
        eigenToPc2Rgb(world_pts, header_stamp, odom_frame_, 140, 140, 140));
    map_cloud_last_pub_t_ = now_t;

    GMS_DEBUG("[vis] map_cloud %ld points, period=%.2fs",
              static_cast<long>(world_pts.rows()),
              map_cloud_publish_period_s_);
}

Eigen::MatrixXf Visualizer::makeGlobalMapCloudPoints(
        const GmmModel& model,
        const Matrix4d& T_world) const {
    if (model.components.empty()) {
        return Eigen::MatrixXf(0, 3);
    }
    Eigen::MatrixXf pts(static_cast<Eigen::Index>(model.components.size() * 7), 3);
    const Matrix3d Rw = T_world.block<3, 3>(0, 0);
    const Vector3d tw = T_world.block<3, 1>(0, 3);
    Eigen::Index row = 0;
    for (const auto& comp : model.components) {
        const Vector3d mu = Rw * comp.mean + tw;
        pts.row(row++) = mu.cast<float>().transpose();

        Matrix3d cov = 0.5 * (comp.covariance + comp.covariance.transpose());
        Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
        if (es.info() != Eigen::Success) {
            for (int k = 0; k < 6; ++k) {
                pts.row(row++) = mu.cast<float>().transpose();
            }
            continue;
        }
        const Vector3d scales =
            es.eigenvalues().cwiseMax(1e-9).cwiseSqrt() * gmm_marker_sigma_;
        const Matrix3d axes = Rw * es.eigenvectors();
        for (int axis = 0; axis < 3; ++axis) {
            const Vector3d offset = axes.col(axis) * scales(axis);
            pts.row(row++) = (mu + offset).cast<float>().transpose();
            pts.row(row++) = (mu - offset).cast<float>().transpose();
        }
    }
    pts.conservativeResize(row, 3);
    return pts;
}

Eigen::MatrixXf Visualizer::voxelDownsamplePoints(
        const Eigen::MatrixXf& pts,
        double voxel_size) const {
    if (voxel_size <= 1e-6 || pts.rows() <= 1) {
        return pts;
    }
    struct Accum {
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        int count = 0;
    };
    std::map<std::array<int, 3>, Accum> voxels;
    const float inv = static_cast<float>(1.0 / voxel_size);
    for (Eigen::Index r = 0; r < pts.rows(); ++r) {
        std::array<int, 3> key = {
            static_cast<int>(std::floor(pts(r, 0) * inv)),
            static_cast<int>(std::floor(pts(r, 1) * inv)),
            static_cast<int>(std::floor(pts(r, 2) * inv))};
        auto& acc = voxels[key];
        acc.sum += pts.row(r).transpose();
        ++acc.count;
    }
    Eigen::MatrixXf out(static_cast<Eigen::Index>(voxels.size()), 3);
    Eigen::Index row = 0;
    for (const auto& [key, acc] : voxels) {
        (void)key;
        out.row(row++) = (acc.sum / static_cast<float>(acc.count)).transpose();
    }
    return out;
}

void Visualizer::maybePublishGlobalMapCloud(const rclcpp::Time& header_stamp) {
    if (!global_map_cloud_enable_ || !pubs_.global_map_cloud ||
        pubs_.global_map_cloud->get_subscription_count() == 0 ||
        global_graph_ == nullptr || !global_graph_->enable) {
        return;
    }
    const double now_t = gmmslam::now().seconds();
    if ((now_t - global_map_cloud_last_pub_t_) <
        global_map_cloud_publish_period_s_) {
        return;
    }

    struct Snapshot {
        int sid = -1;
        GmmModel gmm;
        Matrix4d pose = Matrix4d::Identity();
        int prune_generation = 0;
    };
    std::vector<Snapshot> snapshots;
    Matrix4d map_correction = Matrix4d::Identity();
    const auto map_correction_opt = global_graph_->getMapOdomCorrection();
    const bool publish_in_map_frame =
        !map_frame_.empty() && map_correction_opt.has_value();
    if (publish_in_map_frame) {
        map_correction = map_correction_opt.value();
    }
    {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        snapshots.reserve(global_graph_->submap_ids.size());
        for (int sid : global_graph_->submap_ids) {
            const auto gmm_it = global_graph_->submap_gmm.find(sid);
            if (gmm_it == global_graph_->submap_gmm.end() ||
                gmm_it->second.components.empty()) {
                continue;
            }
            Matrix4d pose = Matrix4d::Identity();
            const auto pose_it = global_graph_->submap_pose_by_idx.find(sid);
            const auto frozen_it = global_graph_->submap_frozen_pose_by_idx.find(sid);
            if (pose_it != global_graph_->submap_pose_by_idx.end() &&
                pose_it->second.allFinite()) {
                pose = pose_it->second;
            } else if (frozen_it != global_graph_->submap_frozen_pose_by_idx.end() &&
                       frozen_it->second.allFinite()) {
                pose = frozen_it->second;
            }
            if (publish_in_map_frame) {
                pose = map_correction * pose;
            }
            int generation = 0;
            const auto gen_it = global_graph_->submap_prune_generation.find(sid);
            if (gen_it != global_graph_->submap_prune_generation.end()) {
                generation = gen_it->second;
            }
            snapshots.push_back({sid, gmm_it->second, pose, generation});
        }
    }

    std::size_t total_rows = 0;
    std::set<int> alive_sids;
    for (const auto& snap : snapshots) {
        alive_sids.insert(snap.sid);
        auto& cache = global_map_cloud_cache_[snap.sid];
        const bool needs_recompute =
            cache.points.rows() == 0 ||
            cache.component_count != snap.gmm.numComponents() ||
            cache.prune_generation != snap.prune_generation ||
            (cache.pose - snap.pose).norm() > 1e-6;
        if (needs_recompute) {
            cache.component_count = snap.gmm.numComponents();
            cache.prune_generation = snap.prune_generation;
            cache.pose = snap.pose;
            cache.points = makeGlobalMapCloudPoints(snap.gmm, snap.pose);
        }
        total_rows += static_cast<std::size_t>(cache.points.rows());
    }
    for (auto it = global_map_cloud_cache_.begin();
         it != global_map_cloud_cache_.end();) {
        if (!alive_sids.count(it->first)) {
            it = global_map_cloud_cache_.erase(it);
        } else {
            ++it;
        }
    }

    if (total_rows == 0) {
        global_map_cloud_last_pub_t_ = now_t;
        return;
    }
    Eigen::MatrixXf pts(static_cast<Eigen::Index>(total_rows), 3);
    Eigen::Index row = 0;
    for (const auto& [sid, cache] : global_map_cloud_cache_) {
        (void)sid;
        const Eigen::Index n = cache.points.rows();
        if (n <= 0) {
            continue;
        }
        pts.block(row, 0, n, 3) = cache.points;
        row += n;
    }
    pts.conservativeResize(row, 3);
    pts = voxelDownsamplePoints(pts, global_map_cloud_voxel_size_m_);

    const std::string frame = publish_in_map_frame ? map_frame_ : odom_frame_;
    pubs_.global_map_cloud->publish(
        eigenToPc2Rgb(pts, header_stamp, frame, 80, 190, 255));
    global_map_cloud_last_pub_t_ = now_t;
}

void Visualizer::publishPruneDebugMarkers(const rclcpp::Time& stamp) {
    if (!prune_debug_markers_enable_ || !pubs_.prune_debug_markers ||
        pubs_.prune_debug_markers->get_subscription_count() == 0 ||
        global_graph_ == nullptr || !global_graph_->enable) {
        return;
    }

    std::vector<PruneDebugRecord> records;
    int generation = 0;
    Matrix4d map_correction = Matrix4d::Identity();
    const auto map_correction_opt = global_graph_->getMapOdomCorrection();
    const bool publish_in_map_frame =
        !map_frame_.empty() && map_correction_opt.has_value();
    if (publish_in_map_frame) {
        map_correction = map_correction_opt.value();
    }
    {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        generation = global_graph_->prune_debug_generation;
        if (generation == prune_debug_last_generation_) {
            return;
        }
        records = global_graph_->prune_debug_records;
    }
    prune_debug_last_generation_ = generation;

    visualization_msgs::msg::MarkerArray ma;
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = publish_in_map_frame ? map_frame_ : odom_frame_;
    clear.ns = "prune_debug";
    clear.id = 0;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    if (records.empty() || prune_debug_max_markers_ <= 0) {
        pubs_.prune_debug_markers->publish(ma);
        return;
    }

    const int max_records =
        std::min(prune_debug_max_markers_, static_cast<int>(records.size()));
    const int start = static_cast<int>(records.size()) - max_records;

    visualization_msgs::msg::Marker lines;
    lines.header = clear.header;
    lines.ns = "prune_debug_links";
    lines.id = 1;
    lines.type = visualization_msgs::msg::Marker::LINE_LIST;
    lines.action = visualization_msgs::msg::Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.scale.x = 0.025;
    lines.color.r = 1.0f;
    lines.color.g = 0.15f;
    lines.color.b = 0.05f;
    lines.color.a = 0.9f;

    visualization_msgs::msg::Marker kept;
    kept.header = clear.header;
    kept.ns = "prune_debug_kept";
    kept.id = 2;
    kept.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    kept.action = visualization_msgs::msg::Marker::ADD;
    kept.pose.orientation.w = 1.0;
    kept.scale.x = kept.scale.y = kept.scale.z = 0.10;
    kept.color.g = 1.0f;
    kept.color.a = 0.95f;

    visualization_msgs::msg::Marker merged;
    merged.header = clear.header;
    merged.ns = "prune_debug_merged";
    merged.id = 3;
    merged.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    merged.action = visualization_msgs::msg::Marker::ADD;
    merged.pose.orientation.w = 1.0;
    merged.scale.x = merged.scale.y = merged.scale.z = 0.08;
    merged.color.r = 1.0f;
    merged.color.a = 0.95f;

    auto toPoint = [&](const Vector3d& p_in) {
        const Vector3d p = publish_in_map_frame
            ? (map_correction.block<3, 3>(0, 0) * p_in +
               map_correction.block<3, 1>(0, 3))
            : p_in;
        geometry_msgs::msg::Point out;
        out.x = p.x();
        out.y = p.y();
        out.z = p.z();
        return out;
    };

    for (int i = start; i < static_cast<int>(records.size()); ++i) {
        const auto& rec = records[static_cast<std::size_t>(i)];
        const auto p_keep = toPoint(rec.kept_mean);
        const auto p_merge = toPoint(rec.merged_mean);
        lines.points.push_back(p_keep);
        lines.points.push_back(p_merge);
        kept.points.push_back(p_keep);
        merged.points.push_back(p_merge);
    }

    ma.markers.push_back(std::move(lines));
    ma.markers.push_back(std::move(kept));
    ma.markers.push_back(std::move(merged));
    pubs_.prune_debug_markers->publish(ma);
}

// =====================================================================
// publishGraphNodeMarkers
// =====================================================================

void Visualizer::publishGraphNodeMarkers(const rclcpp::Time& stamp,
                                         double now_t) {
    if ((now_t - graph_nodes_last_pub_t_) < 0.5) {
        return;
    }

    std::map<int, Matrix4d> smoother_poses;
    {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        smoother_poses = smoother_.pose_by_idx;
    }

    std::map<int, Matrix4d> submap_poses;
    std::map<int, int> submap_anchor_key;
    std::map<int, std::vector<int>> submap_keyframes;
    if (global_graph_ != nullptr && global_graph_->enable) {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        submap_poses = global_graph_->submap_pose_by_idx;
        submap_anchor_key = global_graph_->submap_anchor_key;
        submap_keyframes = global_graph_->submap_keyframes;
    }

    std::vector<std::pair<int, Matrix4d>> nodes;
    if (!submap_poses.empty() && !submap_keyframes.empty()) {
        for (const auto& [sid, keys] : submap_keyframes) {
            const auto sid_pose_it = submap_poses.find(sid);
            const auto anchor_it = submap_anchor_key.find(sid);
            if (sid_pose_it == submap_poses.end() ||
                anchor_it == submap_anchor_key.end()) {
                continue;
            }

            const auto anchor_pose_it = smoother_poses.find(anchor_it->second);
            if (anchor_pose_it == smoother_poses.end() ||
                !anchor_pose_it->second.allFinite() ||
                !sid_pose_it->second.allFinite()) {
                continue;
            }

            const Matrix4d T_anchor_inv = anchor_pose_it->second.inverse();
            for (int key_idx : keys) {
                const auto key_pose_it = smoother_poses.find(key_idx);
                if (key_pose_it == smoother_poses.end() ||
                    !key_pose_it->second.allFinite()) {
                    continue;
                }
                const Matrix4d T_key_in_submap =
                    T_anchor_inv * key_pose_it->second;
                nodes.emplace_back(key_idx, sid_pose_it->second * T_key_in_submap);
            }
        }
    } else {
        nodes.assign(smoother_poses.begin(), smoother_poses.end());
    }

    if (nodes.size() > kMaxGraphNodeMarkers) {
        nodes.erase(nodes.begin(),
                    nodes.end() - static_cast<long>(kMaxGraphNodeMarkers));
    }

    visualization_msgs::msg::MarkerArray ma;

    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = odom_frame_;
    clear.ns = "graph_nodes";
    clear.id = 0;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    int marker_id = 0;
    for (const auto& [key_idx, T_node] : nodes) {
        (void)key_idx;
        visualization_msgs::msg::Marker m;
        m.header.stamp = stamp;
        m.header.frame_id = odom_frame_;
        m.ns = "graph_nodes";
        m.id = marker_id++;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = T_node(0, 3);
        m.pose.position.y = T_node(1, 3);
        m.pose.position.z = T_node(2, 3);
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.08;
        m.color.r = 1.0;
        m.color.a = 1.0;
        ma.markers.push_back(m);
    }

    if (pubs_.graph_nodes) {
        pubs_.graph_nodes->publish(ma);
    }
    graph_nodes_last_pub_t_ = now_t;
}

// =====================================================================
// publishGlobalGraphMarkers
// =====================================================================

void Visualizer::publishGlobalGraphMarkers(const rclcpp::Time& stamp,
                                           double now_t) {
    if (global_graph_ == nullptr || !global_graph_->enable) {
        return;
    }
    if ((now_t - global_graph_markers_last_pub_t_) < 0.5) {
        return;
    }

    std::vector<int> sids;
    std::map<int, Matrix4d> submap_poses;
    std::set<std::pair<int,int>> loop_edges;
    {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        sids = global_graph_->submap_ids;
        for (int sid : sids) {
            auto it = global_graph_->submap_pose_by_idx.find(sid);
            if (it != global_graph_->submap_pose_by_idx.end()) {
                submap_poses[sid] = it->second;
            }
        }
        loop_edges = global_graph_->loop_edges_added;
    }

    visualization_msgs::msg::MarkerArray ma;
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = odom_frame_;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    // Green sphere for each submap
    for (int sid : sids) {
        auto it = submap_poses.find(sid);
        if (it == submap_poses.end()) {
            continue;
        }
        const Matrix4d& T_sid = it->second;

        visualization_msgs::msg::Marker m;
        m.header.stamp    = stamp;
        m.header.frame_id = odom_frame_;
        m.ns              = "global_submaps";
        m.id              = sid;
        m.type            = visualization_msgs::msg::Marker::SPHERE;
        m.action          = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = T_sid(0, 3);
        m.pose.position.y = T_sid(1, 3);
        m.pose.position.z = T_sid(2, 3);
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.08;
        m.color.g = 1.0;
        m.color.a = 0.95;
        ma.markers.push_back(m);
    }

    // Magenta LINE_LIST for loop edges
    visualization_msgs::msg::Marker loop_marker;
    loop_marker.header.stamp    = stamp;
    loop_marker.header.frame_id = odom_frame_;
    loop_marker.ns              = "global_loops";
    loop_marker.id              = 0;
    loop_marker.type            = visualization_msgs::msg::Marker::LINE_LIST;
    loop_marker.action          = visualization_msgs::msg::Marker::ADD;
    loop_marker.pose.orientation.w = 1.0;
    loop_marker.scale.x        = 0.05;
    loop_marker.color.r         = 1.0;
    loop_marker.color.b         = 1.0;
    loop_marker.color.a         = 0.95;

    int loop_label_id = 0;
    for (const auto& [sid_a, sid_b] : loop_edges) {
        auto ia = submap_poses.find(sid_a);
        auto ib = submap_poses.find(sid_b);
        if (ia == submap_poses.end() || ib == submap_poses.end()) {
            continue;
        }
        geometry_msgs::msg::Point p0, p1;
        p0.x = ia->second(0, 3);
        p0.y = ia->second(1, 3);
        p0.z = ia->second(2, 3);
        p1.x = ib->second(0, 3);
        p1.y = ib->second(1, 3);
        p1.z = ib->second(2, 3);
        loop_marker.points.push_back(p0);
        loop_marker.points.push_back(p1);

        visualization_msgs::msg::Marker label;
        label.header.stamp = stamp;
        label.header.frame_id = odom_frame_;
        label.ns = "global_loop_labels";
        label.id = loop_label_id++;
        label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::msg::Marker::ADD;
        label.pose.position.x = 0.5 * (p0.x + p1.x);
        label.pose.position.y = 0.5 * (p0.y + p1.y);
        label.pose.position.z = 0.5 * (p0.z + p1.z) + 0.35;
        label.pose.orientation.w = 1.0;
        label.scale.z = 0.28;
        label.color.r = 1.0;
        label.color.g = 0.8;
        label.color.b = 1.0;
        label.color.a = 1.0;
        label.text = "S(" + std::to_string(sid_a) + ")->S(" +
                     std::to_string(sid_b) + ")";
        ma.markers.push_back(label);
    }
    if (!loop_marker.points.empty()) {
        ma.markers.push_back(loop_marker);
    }

    if (pubs_.global_graph_markers) {
        pubs_.global_graph_markers->publish(ma);
    }
    global_graph_markers_last_pub_t_ = now_t;
}

// =====================================================================
// publishGmmMarkers
// =====================================================================

void Visualizer::publishGmmMarkers(const rclcpp::Time& stamp,
                                   const Matrix4d& T) {
    const bool latest_has_subs =
        pubs_.gmm_markers && pubs_.gmm_markers->get_subscription_count() > 0;
    const bool global_has_subs =
        pubs_.gmm_global_markers &&
        pubs_.gmm_global_markers->get_subscription_count() > 0;

    // --- Latest-frame ellipsoids (white) ---
    int latest_idx = -1;
    GmmModel latest_gmm;
    bool has_latest = false;
    if (latest_has_subs) {
        std::lock_guard<std::mutex> lk(registration_.lock);
        latest_idx = registration_.latest_gmm_idx;
        if (registration_.has_latest_gmm) {
            latest_gmm = registration_.latest_gmm_model;
            has_latest  = true;
        }
    }

    if (latest_has_subs && has_latest) {
        Matrix4d T_latest;
        {
            std::lock_guard<std::mutex> lk(smoother_.graph_lock);
            auto it = smoother_.pose_by_idx.find(latest_idx);
            T_latest = (it != smoother_.pose_by_idx.end()) ? it->second : T;
        }
        const auto local_data = precomputeGmmLocalData(latest_gmm);
        const auto latest_ma = makeMarkersFromCache(
            local_data, T_latest, stamp,
            "gmm_latest", {1.0, 1.0, 1.0},
            0.9, 0, 0.6);
        pubs_.gmm_markers->publish(latest_ma);
    }

    // --- Global GMM map: per-submap with distinct colors ---
    if (!global_has_subs) {
        return;
    }
    const double now_t = stampToSec(stamp);
    if (!global_gmm_markers_enable_) {
        if (!global_gmm_markers_cleared_) {
            visualization_msgs::msg::MarkerArray clear_ma;
            visualization_msgs::msg::Marker clear;
            clear.header.stamp = stamp;
            clear.header.frame_id = odom_frame_;
            clear.ns = "gmm_global";
            clear.id = 0;
            clear.action = visualization_msgs::msg::Marker::DELETEALL;
            clear_ma.markers.push_back(clear);
            pubs_.gmm_global_markers->publish(clear_ma);
            global_gmm_markers_cleared_ = true;
        }
        return;
    }
    if ((now_t - global_gmm_markers_last_pub_t_) < global_gmm_publish_period_s_) {
        return;
    }

    visualization_msgs::msg::MarkerArray global_ma;
    // Component counts can shrink after deferred pruning. Clear the previous
    // latched marker set first, otherwise RViz keeps stale high-ID ellipsoids.
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = odom_frame_;
    clear.ns = "gmm_global";
    clear.id = 0;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    global_ma.markers.push_back(clear);
    int id_counter = 0;
    int n_submaps_rendered = 0;

    std::vector<int> submap_ids;
    std::map<int, Matrix4d> submap_poses;
    std::map<int, std::vector<GmmLocalData>> submap_components;
    std::map<int, Matrix4d> frozen_submap_poses;
    std::map<int, int> submap_prune_generation;

    if (global_graph_ != nullptr && global_graph_->enable) {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        submap_ids = global_graph_->submap_ids;
        for (int sid : submap_ids) {
            auto it = global_graph_->submap_pose_by_idx.find(sid);
            if (it != global_graph_->submap_pose_by_idx.end()) {
                submap_poses[sid] = it->second;
            }
        }
        submap_components = global_graph_->submap_gmm_components;
        frozen_submap_poses = global_graph_->submap_frozen_pose_by_idx;
        submap_prune_generation = global_graph_->submap_prune_generation;
    }

    std::set<int> alive_finalized_sids;
    for (int sid : submap_ids) {
        auto comp_it = submap_components.find(sid);
        if (comp_it == submap_components.end()) {
            continue;
        }
        // Finalized submaps prefer the optimized global pose so the marker map
        // follows loop-closure corrections; frozen poses remain a fallback.
        auto frozen_it = frozen_submap_poses.find(sid);
        auto pose_it   = submap_poses.find(sid);
        const Matrix4d* T_sid = nullptr;
        if (pose_it != submap_poses.end()) {
            T_sid = &pose_it->second;
        } else if (frozen_it != frozen_submap_poses.end()) {
            T_sid = &frozen_it->second;
        }
        if (T_sid == nullptr) {
            continue;
        }

        const auto color = submapColor(sid);
        const int generation = submap_prune_generation.count(sid)
            ? submap_prune_generation.at(sid)
            : 0;
        auto& cache = global_gmm_marker_cache_[sid];
        const bool needs_recompute =
            cache.markers.empty() ||
            cache.component_count != static_cast<int>(comp_it->second.size()) ||
            cache.prune_generation != generation ||
            (cache.pose - *T_sid).norm() > 1e-6;
        if (needs_recompute) {
            const auto ma_i = makeMarkersFromCache(
                comp_it->second, *T_sid, stamp,
                "gmm_global", color, 0.45, 0);
            cache.component_count = static_cast<int>(comp_it->second.size());
            cache.prune_generation = generation;
            cache.pose = *T_sid;
            cache.markers = ma_i.markers;
        }
        for (auto marker : cache.markers) {
            marker.header.stamp = stamp;
            marker.id = id_counter++;
            global_ma.markers.push_back(std::move(marker));
        }
        alive_finalized_sids.insert(sid);
        ++n_submaps_rendered;
    }
    for (auto it = global_gmm_marker_cache_.begin();
         it != global_gmm_marker_cache_.end();) {
        if (!alive_finalized_sids.count(it->first)) {
            it = global_gmm_marker_cache_.erase(it);
        } else {
            ++it;
        }
    }

    // --- Keyframes not yet assigned to a finalized submap (gray / open-submap color) ---
    {
        std::lock_guard<std::mutex> lk(registration_.lock);
        std::vector<int> new_indices;
        for (const auto& [k, entry] : registration_.local_gmms_by_idx) {
            if (k > last_global_gmm_processed_idx_) {
                new_indices.push_back(k);
            }
        }
        std::sort(new_indices.begin(), new_indices.end());

        for (int idx : new_indices) {
            const auto& entry = registration_.local_gmms_by_idx.at(idx);
            auto components = precomputeGmmLocalData(entry.model);
            global_gmm_cache_.emplace_back(idx, std::move(components));
            last_global_gmm_processed_idx_ =
                std::max(last_global_gmm_processed_idx_, idx);
        }
    }

    // Determine which keyframe indices belong to finalized submaps
    std::set<int> finalized_keys;
    if (global_graph_ != nullptr && global_graph_->enable) {
        std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
        for (int sid : global_graph_->submap_ids) {
            if (global_graph_->submap_gmm_components.count(sid)) {
                auto kf_it = global_graph_->submap_keyframes.find(sid);
                if (kf_it != global_graph_->submap_keyframes.end()) {
                    for (int ki : kf_it->second) {
                        finalized_keys.insert(ki);
                    }
                }
            }
        }
    }

    // Snapshot poses
    std::map<int, Matrix4d> pose_snap;
    {
        std::lock_guard<std::mutex> lk(smoother_.graph_lock);
        pose_snap = smoother_.pose_by_idx;
    }
    std::map<int, Matrix4d> map_pose_snap;
    {
        std::lock_guard<std::mutex> lk(registration_.lock);
        for (const auto& [k, entry] : registration_.local_gmms_by_idx) {
            if (entry.has_map_pose) {
                map_pose_snap[k] = entry.map_pose;
            }
        }
    }

    for (const auto& [idx, components] : global_gmm_cache_) {
        if (finalized_keys.count(idx)) {
            continue;
        }
        const Matrix4d* T_i = nullptr;
        auto ps_it = pose_snap.find(idx);
        if (ps_it != pose_snap.end()) {
            T_i = &ps_it->second;
        } else {
            auto mp_it = map_pose_snap.find(idx);
            if (mp_it != map_pose_snap.end()) {
                T_i = &mp_it->second;
            }
        }
        if (T_i == nullptr) {
            continue;
        }

        std::array<double,3> color = {0.5, 0.5, 0.5};
        if (global_graph_ != nullptr) {
            std::lock_guard<std::recursive_mutex> lk(global_graph_->lock);
            auto ks_it = global_graph_->key_to_submap.find(idx);
            if (ks_it != global_graph_->key_to_submap.end()) {
                color = submapColor(ks_it->second);
            }
        }

        const auto ma_i = makeMarkersFromCache(
            components, *T_i, stamp,
            "gmm_global", color, 0.35, id_counter);
        global_ma.markers.insert(global_ma.markers.end(),
                                 ma_i.markers.begin(), ma_i.markers.end());
        id_counter += static_cast<int>(ma_i.markers.size());
    }

    GMS_INFO_THROTTLE(10.0, "[vis] global GMM map: %d markers, %d finalized submaps",
                      static_cast<int>(global_ma.markers.size()), n_submaps_rendered);
    pubs_.gmm_global_markers->publish(global_ma);
    global_gmm_markers_last_pub_t_ = now_t;
}

// =====================================================================
// makeMarkersFromCache
// =====================================================================

visualization_msgs::msg::MarkerArray Visualizer::makeMarkersFromCache(
    const std::vector<GmmLocalData>& components,
    const Matrix4d& T_world,
    const rclcpp::Time& stamp,
    const std::string& ns,
    const std::array<double,3>& color_rgb,
    double alpha,
    int id_start,
    double lifetime_s) const {

    visualization_msgs::msg::MarkerArray ma;
    const Matrix3d R_world = T_world.block<3,3>(0,0);
    const double s_factor = 2.0 * gmm_marker_sigma_;
    const rclcpp::Duration dur = rclcpp::Duration::from_seconds(lifetime_s);

    for (int k = 0; k < static_cast<int>(components.size()); ++k) {
        const auto& comp = components[k];

        // World-frame rotation for this ellipsoid
        const Matrix3d R_combined = R_world * comp.rotation;
        const Eigen::Quaterniond q(R_combined);

        // Transform mean to world frame
        const Eigen::Vector4d mu_h(comp.mean_local.x(),
                                   comp.mean_local.y(),
                                   comp.mean_local.z(),
                                   1.0);
        const Vector3d mu_w = (T_world * mu_h).head<3>();

        visualization_msgs::msg::Marker m;
        m.header.stamp    = stamp;
        m.header.frame_id = odom_frame_;
        m.ns              = ns;
        m.id              = id_start + k;
        m.type            = visualization_msgs::msg::Marker::SPHERE;
        m.action          = visualization_msgs::msg::Marker::ADD;

        m.pose.position.x    = mu_w.x();
        m.pose.position.y    = mu_w.y();
        m.pose.position.z    = mu_w.z();
        // Eigen stores quaternion as (w,x,y,z) internally; ROS expects (x,y,z,w) in message fields
        m.pose.orientation.x = q.x();
        m.pose.orientation.y = q.y();
        m.pose.orientation.z = q.z();
        m.pose.orientation.w = q.w();

        m.scale.x = std::max(s_factor * comp.scales.x(), 0.02);
        m.scale.y = std::max(s_factor * comp.scales.y(), 0.02);
        m.scale.z = std::max(s_factor * comp.scales.z(), 0.02);

        m.color.r = static_cast<float>(color_rgb[0]);
        m.color.g = static_cast<float>(color_rgb[1]);
        m.color.b = static_cast<float>(color_rgb[2]);
        m.color.a = static_cast<float>(alpha);
        m.lifetime = dur;

        ma.markers.push_back(m);
    }
    return ma;
}

} // namespace gmmslam
