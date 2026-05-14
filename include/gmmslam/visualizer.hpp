#pragma once

#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"
#include "gmmslam/thread_safe_queue.hpp"

#include <Eigen/Core>
#include <atomic>
#include <algorithm>
#include <map>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>

namespace gmmslam {

class FixedLagBackend;
class RegistrationManager;
class GlobalPoseGraph;

class Visualizer {
public:
    struct Publishers {
        ros::Publisher path;
        ros::Publisher odom;
        ros::Publisher latest_frame_cloud;
        ros::Publisher map_cloud;
        ros::Publisher gmm_markers;
        ros::Publisher gmm_global_markers;
        ros::Publisher global_graph_markers;
        ros::Publisher graph_nodes;
        tf2_ros::TransformBroadcaster tf_broadcaster;
    };

    Visualizer(FixedLagBackend& smoother,
               RegistrationManager& registration,
               GlobalPoseGraph* global_graph,
               const std::string& odom_frame,
               const std::string& base_frame,
               const VisualizationConfig& vis_cfg,
               Publishers publishers);

    void publishPoseOnly(const Matrix4d& T, const ros::Time& stamp);

    /// @param smoother_pose_key Pose graph index `X(k)` for this scan when
    ///        `pts` was captured (smoother frames only); `-1` skips map buffer.
    void enqueueFrame(const ros::Time& stamp, const Eigen::MatrixXf& pts,
                      int frame_count, const Matrix4d& capture_pose,
                      int smoother_pose_key);

    void visLoop(const std::atomic<bool>& shutdown);

private:
    struct VisFrame {
        ros::Time stamp;
        Eigen::MatrixXf points;
        int frame_count;
        Matrix4d capture_pose;
        int smoother_pose_key = -1;
    };

    struct MapCloudChunk {
        int pose_key;
        Eigen::MatrixXf points_lidar;
    };

    void publishScanProducts(const ros::Time& stamp, const Eigen::MatrixXf& pts,
                             int frame_count, const Matrix4d& capture_pose,
                             int smoother_pose_key);
    void maybePublishMapCloud(const ros::Time& header_stamp);
    void publishGlobalGraphMarkers(const ros::Time& stamp, double now_t);
    void publishGmmMarkers(const ros::Time& stamp, const Matrix4d& T);

    visualization_msgs::MarkerArray makeMarkersFromCache(
        const std::vector<GmmLocalData>& components,
        const Matrix4d& T_world,
        const ros::Time& stamp,
        const std::string& ns,
        const std::array<double,3>& color_rgb,
        double alpha,
        int id_start = 0,
        double lifetime_s = 0.0) const;

    FixedLagBackend& smoother_;
    RegistrationManager& registration_;
    GlobalPoseGraph* global_graph_;

    std::string odom_frame_;
    std::string base_frame_;
    double gmm_marker_sigma_;
    double global_gmm_publish_period_s_;
    double map_cloud_publish_period_s_;
    int map_cloud_max_chunks_;

    Publishers pubs_;

    static constexpr std::size_t kMaxGraphNodeMarkers = 2000;

    nav_msgs::Path path_;
    double global_gmm_markers_last_pub_t_ = 0.0;
    int last_global_gmm_processed_idx_ = -1;
    std::vector<std::pair<int, std::vector<GmmLocalData>>> global_gmm_cache_;
    visualization_msgs::MarkerArray graph_node_markers_;
    double graph_nodes_last_pub_t_ = 0.0;
    double global_graph_markers_last_pub_t_ = 0.0;

    std::mutex map_cloud_mutex_;
    std::deque<MapCloudChunk> map_cloud_chunks_;
    double map_cloud_last_pub_t_ = -1e30;
    ros::Time last_vis_stamp_;

    ThreadSafeQueue<VisFrame> vis_queue_{1};
};

} // namespace gmmslam
