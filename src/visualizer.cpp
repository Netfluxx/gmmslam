#include "gmmslam/visualizer.hpp"
#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/registration_manager.hpp"
#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/ros_helpers.hpp"
#include "gmmslam/util/gmm_utils.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace gmmslam {

// =====================================================================
// Constructor
// =====================================================================

Visualizer::Visualizer(FixedLagBackend& smoother,
                       RegistrationManager& registration,
                       GlobalPoseGraph* global_graph,
                       const std::string& odom_frame,
                       const std::string& base_frame,
                       const VisualizationConfig& vis_cfg,
                       Publishers publishers)
    : smoother_(smoother)
    , registration_(registration)
    , global_graph_(global_graph)
    , odom_frame_(odom_frame)
    , base_frame_(base_frame)
    , gmm_marker_sigma_(vis_cfg.gmm_marker_sigma)
    , global_gmm_publish_period_s_(vis_cfg.global_gmm_publish_period_s)
    , map_cloud_publish_period_s_(vis_cfg.map_cloud_publish_hz > 1e-6
                                      ? 1.0 / vis_cfg.map_cloud_publish_hz
                                      : 9999.0)
    , map_cloud_max_chunks_(std::max(1, vis_cfg.map_cloud_max_chunks))
    , pubs_(std::move(publishers))
{
    path_.header.frame_id = odom_frame_;
}

// =====================================================================
// Pose-only publish (called from main thread on every frame)
// =====================================================================

void Visualizer::publishPoseOnly(const Matrix4d& T, const ros::Time& stamp) {
    const auto ts = poseToTransformStamped(T, stamp, odom_frame_, base_frame_);
    pubs_.tf_broadcaster.sendTransform(ts);

    nav_msgs::Odometry odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id  = base_frame_;
    odom.pose.pose       = poseToPoseStamped(T, stamp, odom_frame_).pose;
    pubs_.odom.publish(odom);
}

// =====================================================================
// Enqueue a frame for the vis thread (drop if full)
// =====================================================================

void Visualizer::enqueueFrame(const ros::Time& stamp,
                              const Eigen::MatrixXf& pts,
                              int frame_count,
                              const Matrix4d& capture_pose,
                              int smoother_pose_key) {
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
        const ros::Time header_stamp =
            last_vis_stamp_.isZero() ? ros::Time::now() : last_vis_stamp_;
        maybePublishMapCloud(header_stamp);

        auto opt = vis_queue_.popWithTimeout(std::chrono::milliseconds(100));
        if (!opt.has_value()) {
            continue;
        }
        try {
            auto& f = opt.value();
            last_vis_stamp_ = f.stamp;
            publishScanProducts(f.stamp, f.points, f.frame_count, f.capture_pose,
                                f.smoother_pose_key);
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[vis] thread error: %s", e.what());
        }
    }
}

// =====================================================================
// publishScanProducts
// =====================================================================

void Visualizer::publishScanProducts(const ros::Time& stamp,
                                     const Eigen::MatrixXf& pts,
                                     int frame_count,
                                     const Matrix4d& capture_pose,
                                     int smoother_pose_key) {
    const Matrix4d& T = capture_pose;

    // --- Path ---
    const auto ps = poseToPoseStamped(T, stamp, odom_frame_);
    path_.header = ps.header;
    path_.poses.push_back(ps);
    pubs_.path.publish(path_);

    // --- Transform current scan into world frame ---
    // pts is Nx3 float; stay in float to avoid double↔float round-trip.
    const Eigen::Matrix3f R = T.block<3, 3>(0, 0).cast<float>();
    const Eigen::RowVector3f t = T.block<3, 1>(0, 3).cast<float>().transpose();
    const Eigen::MatrixXf pts_w = (pts * R.transpose()).rowwise() + t;

    // --- Latest frame cloud (red) ---
    pubs_.latest_frame_cloud.publish(
        eigenToPc2Rgb(pts_w, stamp, odom_frame_, 255, 0, 0));

    // --- Accumulated map: store lidar-frame points keyed by smoother pose X(k);
    //     map_cloud is rebuilt from current pose_by_idx at map_cloud_publish_hz.
    if (smoother_pose_key >= 0 && pts.rows() > 0) {
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

    // --- Graph node marker (red sphere at keyframe position) ---
    {
        visualization_msgs::Marker m;
        m.header.stamp    = stamp;
        m.header.frame_id = odom_frame_;
        m.ns              = "graph_nodes";
        m.id              = frame_count;
        m.type            = visualization_msgs::Marker::SPHERE;
        m.action          = visualization_msgs::Marker::ADD;
        m.pose.position.x = T(0, 3);
        m.pose.position.y = T(1, 3);
        m.pose.position.z = T(2, 3);
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.08;
        m.color.r = 1.0;
        m.color.a = 1.0;
        if (graph_node_markers_.markers.size() >= kMaxGraphNodeMarkers) {
            auto& mk = graph_node_markers_.markers;
            mk.erase(mk.begin(), mk.begin() + static_cast<long>(mk.size() / 4));
        }
        graph_node_markers_.markers.push_back(m);
    }
    if ((now_t - graph_nodes_last_pub_t_) >= 0.5) {
        pubs_.graph_nodes.publish(graph_node_markers_);
        graph_nodes_last_pub_t_ = now_t;
    }

    // --- Global submap graph markers ---
    publishGlobalGraphMarkers(stamp, now_t);

    // --- GMM component markers ---
    publishGmmMarkers(stamp, T);

    maybePublishMapCloud(stamp);
}

// =====================================================================
// maybePublishMapCloud
// =====================================================================

void Visualizer::maybePublishMapCloud(const ros::Time& header_stamp) {
    if (pubs_.map_cloud.getTopic().empty()) {
        return;
    }

    const double now_t = ros::Time::now().toSec();
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
            if (it != pose_snap.end() && it->second.allFinite()) {
                total_rows += static_cast<std::size_t>(ch.points_lidar.rows());
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
                ROS_WARN_THROTTLE(
                    5.0,
                    "[vis] map_cloud chunk pose_key=%d missing from pose snapshot; skip",
                    ch.pose_key);
                continue;
            }
            const Matrix4d& Tw = it->second;
            const Eigen::Matrix3f Rw = Tw.block<3, 3>(0, 0).cast<float>();
            const Eigen::RowVector3f tw =
                Tw.block<3, 1>(0, 3).cast<float>().transpose();
            const Eigen::MatrixXf& pl = ch.points_lidar;
            const Eigen::Index n = pl.rows();
            if (row_out + n > world_pts.rows()) {
                ROS_WARN_THROTTLE(
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

    pubs_.map_cloud.publish(
        eigenToPc2Rgb(world_pts, header_stamp, odom_frame_, 140, 140, 140));
    map_cloud_last_pub_t_ = now_t;

    ROS_DEBUG("[vis] map_cloud %ld points, period=%.2fs",
              static_cast<long>(world_pts.rows()),
              map_cloud_publish_period_s_);
}

// =====================================================================
// publishGlobalGraphMarkers
// =====================================================================

void Visualizer::publishGlobalGraphMarkers(const ros::Time& stamp,
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

    visualization_msgs::MarkerArray ma;

    // Green sphere for each submap
    for (int sid : sids) {
        auto it = submap_poses.find(sid);
        if (it == submap_poses.end()) {
            continue;
        }
        const Matrix4d& T_sid = it->second;

        visualization_msgs::Marker m;
        m.header.stamp    = stamp;
        m.header.frame_id = odom_frame_;
        m.ns              = "global_submaps";
        m.id              = sid;
        m.type            = visualization_msgs::Marker::SPHERE;
        m.action          = visualization_msgs::Marker::ADD;
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
    visualization_msgs::Marker loop_marker;
    loop_marker.header.stamp    = stamp;
    loop_marker.header.frame_id = odom_frame_;
    loop_marker.ns              = "global_loops";
    loop_marker.id              = 0;
    loop_marker.type            = visualization_msgs::Marker::LINE_LIST;
    loop_marker.action          = visualization_msgs::Marker::ADD;
    loop_marker.pose.orientation.w = 1.0;
    loop_marker.scale.x        = 0.05;
    loop_marker.color.r         = 1.0;
    loop_marker.color.b         = 1.0;
    loop_marker.color.a         = 0.95;

    for (const auto& [sid_a, sid_b] : loop_edges) {
        auto ia = submap_poses.find(sid_a);
        auto ib = submap_poses.find(sid_b);
        if (ia == submap_poses.end() || ib == submap_poses.end()) {
            continue;
        }
        geometry_msgs::Point p0, p1;
        p0.x = ia->second(0, 3);
        p0.y = ia->second(1, 3);
        p0.z = ia->second(2, 3);
        p1.x = ib->second(0, 3);
        p1.y = ib->second(1, 3);
        p1.z = ib->second(2, 3);
        loop_marker.points.push_back(p0);
        loop_marker.points.push_back(p1);
    }
    if (!loop_marker.points.empty()) {
        ma.markers.push_back(loop_marker);
    }

    pubs_.global_graph_markers.publish(ma);
    global_graph_markers_last_pub_t_ = now_t;
}

// =====================================================================
// publishGmmMarkers
// =====================================================================

void Visualizer::publishGmmMarkers(const ros::Time& stamp,
                                   const Matrix4d& T) {
    // --- Latest-frame ellipsoids (white) ---
    int latest_idx = -1;
    GmmModel latest_gmm;
    bool has_latest = false;
    {
        std::lock_guard<std::mutex> lk(registration_.lock);
        latest_idx = registration_.latest_gmm_idx;
        if (registration_.has_latest_gmm) {
            latest_gmm = registration_.latest_gmm_model;
            has_latest  = true;
        }
    }

    if (has_latest) {
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
        pubs_.gmm_markers.publish(latest_ma);
    }

    // --- Global GMM map: per-submap with distinct colors ---
    const double now_t = stampToSec(stamp);
    if ((now_t - global_gmm_markers_last_pub_t_) < global_gmm_publish_period_s_) {
        return;
    }

    visualization_msgs::MarkerArray global_ma;
    int id_counter = 0;
    int n_submaps_rendered = 0;

    std::vector<int> submap_ids;
    std::map<int, Matrix4d> submap_poses;
    std::map<int, std::vector<GmmLocalData>> submap_components;
    std::map<int, Matrix4d> frozen_submap_poses;

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
    }

    for (int sid : submap_ids) {
        auto comp_it = submap_components.find(sid);
        if (comp_it == submap_components.end()) {
            continue;
        }
        // Finalized submaps use a frozen pose to avoid post-hoc skew.
        auto frozen_it = frozen_submap_poses.find(sid);
        auto pose_it   = submap_poses.find(sid);
        const Matrix4d* T_sid = nullptr;
        if (frozen_it != frozen_submap_poses.end()) {
            T_sid = &frozen_it->second;
        } else if (pose_it != submap_poses.end()) {
            T_sid = &pose_it->second;
        }
        if (T_sid == nullptr) {
            continue;
        }

        const auto color = submapColor(sid);
        const auto ma_i = makeMarkersFromCache(
            comp_it->second, *T_sid, stamp,
            "gmm_global", color, 0.45, id_counter);
        global_ma.markers.insert(global_ma.markers.end(),
                                 ma_i.markers.begin(), ma_i.markers.end());
        id_counter += static_cast<int>(ma_i.markers.size());
        ++n_submaps_rendered;
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

    ROS_INFO_THROTTLE(10.0, "[vis] global GMM map: %d markers, %d finalized submaps",
                      static_cast<int>(global_ma.markers.size()), n_submaps_rendered);
    pubs_.gmm_global_markers.publish(global_ma);
    global_gmm_markers_last_pub_t_ = now_t;
}

// =====================================================================
// makeMarkersFromCache
// =====================================================================

visualization_msgs::MarkerArray Visualizer::makeMarkersFromCache(
    const std::vector<GmmLocalData>& components,
    const Matrix4d& T_world,
    const ros::Time& stamp,
    const std::string& ns,
    const std::array<double,3>& color_rgb,
    double alpha,
    int id_start,
    double lifetime_s) const {

    visualization_msgs::MarkerArray ma;
    const Matrix3d R_world = T_world.block<3,3>(0,0);
    const double s_factor = 2.0 * gmm_marker_sigma_;
    const ros::Duration dur(lifetime_s);

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

        visualization_msgs::Marker m;
        m.header.stamp    = stamp;
        m.header.frame_id = odom_frame_;
        m.ns              = ns;
        m.id              = id_start + k;
        m.type            = visualization_msgs::Marker::SPHERE;
        m.action          = visualization_msgs::Marker::ADD;

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
