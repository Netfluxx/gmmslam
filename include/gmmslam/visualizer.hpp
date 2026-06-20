#pragma once

#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"
#include "gmmslam/thread_safe_queue.hpp"

#include <Eigen/Core>
#include <atomic>
#include <algorithm>
#include <map>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace gmmslam {

class FixedLagBackend;
class RegistrationManager;
class GlobalPoseGraph;

class Visualizer {
public:
    struct Publishers {
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_lpf;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr latest_frame_cloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_cloud;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_cloud;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gmm_markers;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gmm_global_markers;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr global_graph_markers;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_nodes;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr prune_debug_markers;
        std::function<void(const geometry_msgs::msg::TransformStamped&)>
            send_transform;
    };

    Visualizer(FixedLagBackend& smoother,
               RegistrationManager& registration,
               GlobalPoseGraph* global_graph,
               const std::string& odom_frame,
               const std::string& base_frame,
               const std::string& map_frame,
               const VisualizationConfig& vis_cfg,
               Publishers publishers);

    Matrix4d filterOutputPose(const Matrix4d& T, const rclcpp::Time& stamp);
    void publishPoseOnly(const Matrix4d& T, const rclcpp::Time& stamp);
    void publishPoseLpf(const Matrix4d& T, const rclcpp::Time& stamp);

    /// @param smoother_pose_key Pose graph index `X(k)` for this scan when
    ///        `pts` was captured (smoother frames only); `-1` skips map buffer.
    void enqueueFrame(const rclcpp::Time& stamp,
                      std::shared_ptr<const Eigen::MatrixXf> pts,
                      int frame_count, const Matrix4d& capture_pose,
                      int smoother_pose_key);

    void visLoop(const std::atomic<bool>& shutdown);

private:
    struct VisFrame {
        rclcpp::Time stamp;
        std::shared_ptr<const Eigen::MatrixXf> points;
        int frame_count;
        Matrix4d capture_pose;
        int smoother_pose_key = -1;
    };

    struct MapCloudChunk {
        int pose_key;
        std::shared_ptr<const Eigen::MatrixXf> points_lidar;
    };

    struct GlobalMapCloudCache {
        int component_count = 0;
        int prune_generation = 0;
        Matrix4d pose = Matrix4d::Identity();
        Eigen::MatrixXf points;
    };

    struct GlobalGmmMarkerCache {
        int component_count = 0;
        int prune_generation = 0;
        Matrix4d pose = Matrix4d::Identity();
        std::vector<visualization_msgs::msg::Marker> markers;
    };

    void publishScanProducts(const rclcpp::Time& stamp,
                             std::shared_ptr<const Eigen::MatrixXf> pts,
                             int frame_count, const Matrix4d& capture_pose,
                             int smoother_pose_key);
    void maybePublishMapCloud(const rclcpp::Time& header_stamp);
    void maybePublishGlobalMapCloud(const rclcpp::Time& header_stamp);
    void publishPruneDebugMarkers(const rclcpp::Time& stamp);
    void publishGraphNodeMarkers(const rclcpp::Time& stamp, double now_t);
    void publishGlobalGraphMarkers(const rclcpp::Time& stamp, double now_t);
    void publishGmmMarkers(const rclcpp::Time& stamp, const Matrix4d& T);

    visualization_msgs::msg::MarkerArray makeMarkersFromCache(
        const std::vector<GmmLocalData>& components,
        const Matrix4d& T_world,
        const rclcpp::Time& stamp,
        const std::string& ns,
        const std::array<double,3>& color_rgb,
        double alpha,
        int id_start = 0,
        double lifetime_s = 0.0) const;

    Eigen::MatrixXf makeGlobalMapCloudPoints(
        const GmmModel& model,
        const Matrix4d& T_world) const;
    Eigen::MatrixXf voxelDownsamplePoints(
        const Eigen::MatrixXf& pts,
        double voxel_size) const;

    FixedLagBackend& smoother_;
    RegistrationManager& registration_;
    GlobalPoseGraph* global_graph_;

    std::string odom_frame_;
    std::string base_frame_;
    std::string map_frame_;
    double gmm_marker_sigma_;
    bool global_gmm_markers_enable_;
    double global_gmm_publish_period_s_;
    double output_pose_lpf_cutoff_hz_;
    double map_cloud_publish_period_s_;
    int map_cloud_max_chunks_;
    bool global_map_cloud_enable_;
    double global_map_cloud_publish_period_s_;
    double global_map_cloud_voxel_size_m_;
    bool prune_debug_markers_enable_;
    int prune_debug_max_markers_;

    Publishers pubs_;

    static constexpr std::size_t kMaxGraphNodeMarkers = 2000;

    nav_msgs::msg::Path path_;
    bool output_pose_filter_initialized_ = false;
    Matrix4d output_pose_filtered_ = Matrix4d::Identity();
    rclcpp::Time output_pose_filter_stamp_;
    double global_gmm_markers_last_pub_t_ = 0.0;
    bool global_gmm_markers_cleared_ = false;
    int last_global_gmm_processed_idx_ = -1;
    std::vector<std::pair<int, std::vector<GmmLocalData>>> global_gmm_cache_;
    double graph_nodes_last_pub_t_ = 0.0;
    double global_graph_markers_last_pub_t_ = 0.0;

    std::mutex map_cloud_mutex_;
    std::deque<MapCloudChunk> map_cloud_chunks_;
    double map_cloud_last_pub_t_ = -1e30;
    std::map<int, GlobalMapCloudCache> global_map_cloud_cache_;
    std::map<int, GlobalGmmMarkerCache> global_gmm_marker_cache_;
    double global_map_cloud_last_pub_t_ = -1e30;
    int prune_debug_last_generation_ = -1;
    rclcpp::Time last_vis_stamp_;

    ThreadSafeQueue<VisFrame> vis_queue_{1};
};

} // namespace gmmslam
