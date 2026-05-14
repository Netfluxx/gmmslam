#include "gmmslam/config.hpp"
#include "gmmslam/types.hpp"
#include "gmmslam/ros_helpers.hpp"
#include "gmmslam/fixed_lag_backend.hpp"
#include "gmmslam/global_pose_graph.hpp"
#include "gmmslam/registration_manager.hpp"
#include "gmmslam/visualizer.hpp"
#include "gmmslam/util/gmm_utils.hpp"

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <unistd.h>

namespace gmmslam {

class GMMSLAMNode {
public:
    GMMSLAMNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~GMMSLAMNode();

private:
    void pclCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void gtCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void noisyGtCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);

    bool ensureGtOriginInitialized(const ros::Time& stamp);
    bool shouldAddKeyframe(const ros::Time& stamp, const Matrix4d& current_pose);
    std::optional<Matrix4d> sampleNoisyGtRelativePose(const ros::Time& stamp);
    std::optional<Matrix4d> lookupNoisyGtAt(double t_sec) const;
    std::vector<std::tuple<double, Vector3d, Vector3d>> imuMeasurementsBetween(
        double t_prev, double t_curr) const;
    std::optional<Matrix4d> submapTrajDeltaBetween(
        int prev_key, int curr_key, double prev_t, double curr_t);
    std::optional<std::pair<GmmModel, Matrix4d>> getKeyframeGmm(int key_idx);
    void logBackpressure(const ros::Time& stamp);
    std::optional<Matrix4d> currentGtPoseOdomFrame(const ros::Time& stamp) const;
    void maybeReanchorSmootherFromGt(const ros::Time& stamp, double t_cloud);
    std::optional<Matrix4d> loadRestartPose() const;
    void saveRestartPose(const Matrix4d& pose, double t_sec, int key_idx);

    Config cfg_;

    std::unique_ptr<FixedLagBackend> smoother_;
    std::unique_ptr<GlobalPoseGraph> global_graph_;
    std::unique_ptr<RegistrationManager> registration_;
    std::unique_ptr<Visualizer> visualizer_;

    ros::Subscriber lidar_sub_, gt_sub_, noisy_gt_sub_, imu_sub_, reg_result_sub_;
    ros::Publisher gt_path_pub_, gt_pose_pub_;

    int frame_count_ = 0;
    int odom_idx_ = 0;
    int keyframe_count_ = 0;
    std::vector<int> keyframe_odom_indices_;
    Matrix4d last_keyframe_pose_ = Matrix4d::Identity();
    double last_keyframe_t_sec_ = 0.0;
    bool first_cloud_seen_ = false;
    int smoother_frame_counter_ = 0;
    ros::Time last_smoother_stamp_;
    bool has_last_smoother_stamp_ = false;
    std::optional<Matrix4d> accumulated_gt_rel_;
    int reg_enqueue_resume_frame_ = 0;
    double last_backpressure_log_t_ = 0.0;
    double last_cloud_t_sec_processed_ = -1.0;
    bool has_last_keyframe_ = false;
    std::optional<Matrix4d> restart_pose_;
    double last_restart_state_write_t_ = -1.0;

    geometry_msgs::PoseStamped::ConstPtr latest_gt_pose_raw_;
    std::optional<Matrix4d> gt_origin_inv_;
    ros::Time gt_init_start_time_;
    std::optional<Matrix4d> last_gt_T_for_factor_;
    double last_gt_T_stamp_sec_ = 0.0;
    nav_msgs::Path gt_path_;

    struct NoisyGtEntry {
        double t_sec;
        Matrix4d pose;
    };
    std::deque<NoisyGtEntry> noisy_gt_buffer_;
    static constexpr int kNoisyGtBufferMax = 200;
    geometry_msgs::PoseStamped::ConstPtr latest_noisy_gt_msg_;

    struct ImuEntry {
        ros::Time stamp;
        Vector3d acc;
        Vector3d gyro;
    };
    std::deque<ImuEntry> imu_buffer_;

    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_{0.0, 1.0};

    std::atomic<bool> shutdown_{false};
    std::vector<std::thread> worker_threads_;
};

// =====================================================================
// Constructor
// =====================================================================

GMMSLAMNode::GMMSLAMNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
{
    // --- Load configuration ---
    std::string config_path;
    pnh.param<std::string>("config_file", config_path, "");
    if (!config_path.empty()) {
        cfg_ = loadConfig(config_path);
    } else {
        // ROS param fallback — mirrors the Python node's rospy.get_param("~xxx", default)
        pnh.param<std::string>("lidar_topic",  cfg_.ros.lidar_topic,  "/m500_1/mpa/depth/points");
        pnh.param<std::string>("gt_topic",     cfg_.ros.gt_topic,     "/m500_1/mavros/local_position/pose");
        pnh.param<std::string>("imu_topic",    cfg_.ros.imu_topic,    "/m500_1/mavros/imu/data");
        pnh.param<std::string>("sensor_frame", cfg_.ros.sensor_frame, "depth_camera_1");
        pnh.param<std::string>("odom_frame",   cfg_.ros.odom_frame,   "world");
        pnh.param<std::string>("base_frame",   cfg_.ros.base_frame,   "m500_1_base_link");
        if (!pnh.getParam("odometry_input", cfg_.ros.odometry_input)) {
            if (!pnh.getParam("noisy_gt_topic", cfg_.ros.odometry_input)) {
                cfg_.ros.odometry_input = "/gmmslam_node/noisy_gt_pose";
            }
        }
        pnh.param<std::string>("restart_state_path", cfg_.ros.restart_state_path,
                               cfg_.ros.restart_state_path);
        pnh.param<std::string>("registration_request_topic",
                               cfg_.ros.registration_request_topic,
                               "/gmmslam_node/registration/request");
        pnh.param<std::string>("registration_result_topic",
                               cfg_.ros.registration_result_topic,
                               "/gmmslam_node/registration/result");

        pnh.param("min_range",       cfg_.preprocess.min_range,       0.1);
        pnh.param("max_range",       cfg_.preprocess.max_range,       10.0);
        pnh.param("voxel_leaf_size", cfg_.preprocess.voxel_leaf_size, 0.05);
        pnh.param("target_points",   cfg_.preprocess.target_points,   0);
        pnh.param("min_points",      cfg_.preprocess.min_points,      50);

        pnh.param("sogmm_bandwidth",     cfg_.sogmm.bandwidth,     0.02);
        pnh.param("sogmm_max_points",    cfg_.sogmm.max_points,    2000);
        pnh.param("sogmm_n_components",  cfg_.sogmm.n_components,  0);
        pnh.param<std::string>("sogmm_compute", cfg_.sogmm.compute, "GPU");

        pnh.param("fixed_lag_s",         cfg_.smoother.fixed_lag_s,         4.0);
        pnh.param("smoother_stride",     cfg_.smoother.smoother_stride,     3);
        pnh.param("odom_noise_sigma_t",  cfg_.smoother.odom_noise_sigma_t,  0.03);
        pnh.param("odom_noise_sigma_r",  cfg_.smoother.odom_noise_sigma_r,  0.03);
        pnh.param("lost_scale",          cfg_.smoother.lost_scale,          10.0);
        pnh.param("prior_scale",         cfg_.smoother.prior_scale,         0.1);
        pnh.param("pose_history_keep_keyframes", cfg_.smoother.pose_history_keep, 5000);

        pnh.param("enable_async_registration",  cfg_.registration.enable_async, true);
        pnh.param("registration_request_every_n_frames",
                  cfg_.registration.request_every_n_frames, 3);
        pnh.param("registration_factor_every_n_frames",
                  cfg_.registration.factor_every_n_frames, 3);
        pnh.param("registration_score_threshold",
                  cfg_.registration.score_threshold, 0.3);
        pnh.param("strong_factor_score_threshold",
                  cfg_.registration.strong_factor_score_threshold, 2.0);
        pnh.param("strong_factor_sigma_t",
                  cfg_.registration.strong_factor_sigma_t, 0.01);
        pnh.param("strong_factor_sigma_r",
                  cfg_.registration.strong_factor_sigma_r, 0.01);
        pnh.param("registration_queue_size",
                  cfg_.registration.queue_size, 8);
        pnh.param("registration_result_queue_size",
                  cfg_.registration.result_queue_size, 64);
        pnh.param("registration_enqueue_cooldown_frames",
                  cfg_.registration.enqueue_cooldown_frames, 8);
        pnh.param("registration_workers",  cfg_.registration.workers, 6);
        pnh.param("compensate_fit_latency_in_map",
                  cfg_.registration.compensate_fit_latency, true);
        pnh.param("score_sigma_low",  cfg_.registration.score_sigma_low,  0.6);
        pnh.param("score_sigma_high", cfg_.registration.score_sigma_high, 1.7);
        pnh.param("seq_sigma_t_min",  cfg_.registration.seq_sigma_t_min,  0.02);
        pnh.param("seq_sigma_t_max",  cfg_.registration.seq_sigma_t_max,  0.20);
        pnh.param("seq_sigma_r_min",  cfg_.registration.seq_sigma_r_min,  0.01);
        pnh.param("seq_sigma_r_max",  cfg_.registration.seq_sigma_r_max,  0.15);
        pnh.param("loop_sigma_t_min", cfg_.registration.loop_sigma_t_min, 0.03);
        pnh.param("loop_sigma_t_max", cfg_.registration.loop_sigma_t_max, 0.40);
        pnh.param("loop_sigma_r_min", cfg_.registration.loop_sigma_r_min, 0.02);
        pnh.param("loop_sigma_r_max", cfg_.registration.loop_sigma_r_max, 0.25);

        pnh.param("enable_loop_closure_detection",  cfg_.loop_closure.enable, true);
        pnh.param("loop_closure_search_radius_m",   cfg_.loop_closure.search_radius_m, 2.5);
        pnh.param("loop_closure_min_keyframe_gap",   cfg_.loop_closure.min_keyframe_gap, 10);
        pnh.param("loop_closure_max_candidates",     cfg_.loop_closure.max_candidates, 5);
        pnh.param("loop_closure_request_every_n_keyframes",
                  cfg_.loop_closure.request_every_n_keyframes, 2);
        pnh.param("loop_closure_search_cooldown_keyframes",
                  cfg_.loop_closure.search_cooldown_keyframes, 2);
        pnh.param("loop_closure_min_separation_m",
                  cfg_.loop_closure.min_separation_m, 0.8);
        pnh.param("loop_closure_min_separation_deg",
                  cfg_.loop_closure.min_separation_deg, 20.0);
        pnh.param("loop_closure_max_age_s",          cfg_.loop_closure.max_age_s, 720.0);
        pnh.param("loop_closure_gmm_keep_keyframes", cfg_.loop_closure.gmm_keep_keyframes, 15000);
        pnh.param("loop_closure_detect_score_threshold",
                  cfg_.loop_closure.detect_score_threshold, 1.1);
        pnh.param("loop_closure_super_sigma_t",  cfg_.loop_closure.super_sigma_t, 0.01);
        pnh.param("loop_closure_super_sigma_r",  cfg_.loop_closure.super_sigma_r, 0.01);

        pnh.param("solid_enable",             cfg_.solid.enable, true);
        pnh.param("solid_fov_up_deg",         cfg_.solid.fov_up_deg, 30.0);
        pnh.param("solid_fov_down_deg",       cfg_.solid.fov_down_deg, -30.0);
        pnh.param("solid_min_distance_m",     cfg_.solid.min_distance_m, 0.5);
        pnh.param("solid_max_distance_m",     cfg_.solid.max_distance_m, 25.0);
        pnh.param("solid_num_angle",          cfg_.solid.num_angle, 36);
        pnh.param("solid_num_range",          cfg_.solid.num_range, 32);
        pnh.param("solid_num_height",         cfg_.solid.num_height, 16);
        pnh.param("solid_voxel_size_m",       cfg_.solid.voxel_size_m, 0.0);
        pnh.param("solid_cos_similarity_threshold",
                  cfg_.solid.cos_similarity_threshold, 0.85);
        pnh.param("solid_radius_weight",      cfg_.solid.radius_weight, 0.5);
        pnh.param("solid_appearance_weight",  cfg_.solid.appearance_weight, 0.5);
        pnh.param("solid_provide_yaw_prior",  cfg_.solid.provide_yaw_prior, true);
        pnh.param("solid_overlap_min",        cfg_.solid.overlap_min, 0.3);
        pnh.param("solid_max_abs_yaw_deg",    cfg_.solid.max_abs_yaw_deg, 180.0);
        pnh.param("solid_yaw_sign",           cfg_.solid.yaw_sign, 1);
        pnh.param("solid_rescue_enable",      cfg_.solid.rescue_enable, true);
        pnh.param("solid_rescue_every_n_kf",  cfg_.solid.rescue_every_n_kf, 10);
        pnh.param("solid_rescue_trigger_silence_kf",
                  cfg_.solid.rescue_trigger_silence_kf, 30);
        pnh.param("solid_rescue_top_k",       cfg_.solid.rescue_top_k, 3);
        pnh.param("solid_rescue_cos_threshold",
                  cfg_.solid.rescue_cos_threshold, 0.9);
        pnh.param("solid_keep_descriptors",   cfg_.solid.keep_descriptors, 0);

        pnh.param("keyframe_translation_thresh_m",
                  cfg_.keyframe.translation_thresh_m, 0.3);
        pnh.param("keyframe_rotation_thresh_deg",
                  cfg_.keyframe.rotation_thresh_deg, 7.0);
        pnh.param("keyframe_use_time_trigger",
                  cfg_.keyframe.use_time_trigger, false);
        pnh.param("keyframe_max_interval_s",
                  cfg_.keyframe.max_interval_s, 10.0);

        pnh.param("enable_global_pose_graph",       cfg_.global_graph.enable, true);
        pnh.param("submap_keyframes_per_submap",    cfg_.global_graph.submap_keyframes_per_submap, 10);
        pnh.param("submap_between_sigma_t",         cfg_.global_graph.between_sigma_t, 0.08);
        pnh.param("submap_between_sigma_r",         cfg_.global_graph.between_sigma_r, 0.08);
        pnh.param("submap_prior_sigma_t",           cfg_.global_graph.prior_sigma_t, 0.02);
        pnh.param("submap_prior_sigma_r",           cfg_.global_graph.prior_sigma_r, 0.02);
        pnh.param("submap_overlap_radius_m",        cfg_.global_graph.overlap_radius_m, 3.0);
        pnh.param("submap_reg_score_threshold",     cfg_.global_graph.reg_score_threshold, 0.5);
        pnh.param("submap_traj_sigma_t",            cfg_.global_graph.traj_sigma_t, 0.15);
        pnh.param("submap_traj_sigma_r",            cfg_.global_graph.traj_sigma_r, 0.15);
        pnh.param("submap_aux_gate_abs_trans_m",    cfg_.global_graph.aux_gate_abs_trans_m, 6.0);
        pnh.param("submap_aux_gate_abs_rot_deg",    cfg_.global_graph.aux_gate_abs_rot_deg, 50.0);
        pnh.param("submap_aux_gate_consistency_trans_m",
                  cfg_.global_graph.aux_gate_consistency_trans_m, 2.0);
        pnh.param("submap_aux_gate_consistency_rot_deg",
                  cfg_.global_graph.aux_gate_consistency_rot_deg, 35.0);
        pnh.param("submap_traj_aux_gate_abs_trans_m",
                  cfg_.global_graph.traj_aux_gate_abs_trans_m, 6.0);
        pnh.param("submap_traj_aux_gate_abs_rot_deg",
                  cfg_.global_graph.traj_aux_gate_abs_rot_deg, 55.0);
        pnh.param("submap_traj_aux_gate_consistency_trans_m",
                  cfg_.global_graph.traj_aux_gate_consistency_trans_m, 2.0);
        pnh.param("submap_traj_aux_gate_consistency_rot_deg",
                  cfg_.global_graph.traj_aux_gate_consistency_rot_deg, 40.0);
        pnh.param("submap_loop_sigma_t_min", cfg_.global_graph.submap_loop_sigma_t_min, 0.05);
        pnh.param("submap_loop_sigma_t_max", cfg_.global_graph.submap_loop_sigma_t_max, 0.50);
        pnh.param("submap_loop_sigma_r_min", cfg_.global_graph.submap_loop_sigma_r_min, 0.03);
        pnh.param("submap_loop_sigma_r_max", cfg_.global_graph.submap_loop_sigma_r_max, 0.30);
        pnh.param("submap_reanchor_smoother_on_traj_gate_fail",
                  cfg_.global_graph.reanchor_smoother_on_traj_gate_fail, true);
        pnh.param("submap_finalize_min_ready_keyframes",
                  cfg_.global_graph.submap_finalize_min_ready_keyframes, 1);
        pnh.param("submap_finalize_min_ready_fraction",
                  cfg_.global_graph.submap_finalize_min_ready_fraction, 0.0);
        pnh.param("submap_finalize_max_wait_s",
                  cfg_.global_graph.submap_finalize_max_wait_s, 0.0);

        pnh.param("gt_init_wait_s",    cfg_.gt_noise.init_wait_s, 3.0);
        pnh.param("gt_noise_sigma_t",  cfg_.gt_noise.sigma_t, 0.03);
        pnh.param("gt_noise_sigma_r",  cfg_.gt_noise.sigma_r, 0.03);
        pnh.param("gt_factor_sigma_t", cfg_.gt_noise.factor_sigma_t, 0.045);
        pnh.param("gt_factor_sigma_r", cfg_.gt_noise.factor_sigma_r, 0.045);
        pnh.param("gt_noise_seed",     cfg_.gt_noise.seed, -1);

        pnh.param("enable_imu_preintegration",  cfg_.imu.enable_preintegration, false);
        pnh.param("imu_gravity_mps2",           cfg_.imu.gravity_mps2, 9.81);
        pnh.param("imu_buffer_keep_s",          cfg_.imu.buffer_keep_s, 30.0);
        pnh.param("imu_acc_noise_sigma",        cfg_.imu.acc_noise_sigma, 0.2);
        pnh.param("imu_gyro_noise_sigma",       cfg_.imu.gyro_noise_sigma, 0.05);
        pnh.param("imu_integration_sigma",      cfg_.imu.integration_sigma, 0.0001);
        pnh.param("imu_bias_acc_rw_sigma",      cfg_.imu.bias_acc_rw_sigma, 0.001);
        pnh.param("imu_bias_gyro_rw_sigma",     cfg_.imu.bias_gyro_rw_sigma, 0.0001);
        pnh.param("imu_velocity_prior_sigma",   cfg_.imu.velocity_prior_sigma, 1.0);
        pnh.param("imu_bias_prior_sigma",       cfg_.imu.bias_prior_sigma, 0.1);

        pnh.param("gmm_marker_sigma",            cfg_.visualization.gmm_marker_sigma, 3.0);
        pnh.param("global_gmm_publish_period_s",  cfg_.visualization.global_gmm_publish_period_s, 1.0);
        pnh.param("map_cloud_publish_hz",        cfg_.visualization.map_cloud_publish_hz, 0.5);
        pnh.param("map_cloud_max_chunks",        cfg_.visualization.map_cloud_max_chunks, 3000);

        pnh.param("map_prune_enable",            cfg_.map.prune_enable, cfg_.map.prune_enable);
        pnh.param("map_prune_bhatt_threshold",   cfg_.map.prune_bhatt_threshold, cfg_.map.prune_bhatt_threshold);
        pnh.param("map_prune_search_radius_m",   cfg_.map.prune_search_radius_m, cfg_.map.prune_search_radius_m);
        pnh.param("map_prune_use_rtree",         cfg_.map.prune_use_rtree, cfg_.map.prune_use_rtree);
        pnh.param("map_prune_rtree_chi_sq",      cfg_.map.prune_rtree_chi_sq, cfg_.map.prune_rtree_chi_sq);
        pnh.param("map_prune_max_passes",        cfg_.map.prune_max_passes, cfg_.map.prune_max_passes);
        pnh.param("map_prune_cov_reg",           cfg_.map.prune_cov_reg, cfg_.map.prune_cov_reg);

        pnh.param<std::string>("gmm_dir", cfg_.gmm_dir, "/tmp/gmmslam_gmms");
    }

    {
        std::string oin;
        if (pnh.getParam("odometry_input", oin)) {
            cfg_.ros.odometry_input = oin;
        } else if (pnh.getParam("noisy_gt_topic", oin)) {
            cfg_.ros.odometry_input = oin;
        }
        pnh.param<std::string>("restart_state_path", cfg_.ros.restart_state_path,
                               cfg_.ros.restart_state_path);
    }

    restart_pose_ = loadRestartPose();

    ROS_INFO("[gmmslam] lidar_topic   : %s", cfg_.ros.lidar_topic.c_str());
    ROS_INFO("[gmmslam] odom_frame    : %s", cfg_.ros.odom_frame.c_str());
    ROS_INFO("[gmmslam] fixed_lag_s   : %.2f", cfg_.smoother.fixed_lag_s);
    ROS_INFO("[gmmslam] smoother_stride: %d", cfg_.smoother.smoother_stride);
    ROS_INFO("[gmmslam] gt_factor_sigma: t=%.6f r=%.6f (gt_noise: t=%.4f r=%.4f)",
             cfg_.gt_noise.factor_sigma_t, cfg_.gt_noise.factor_sigma_r,
             cfg_.gt_noise.sigma_t, cfg_.gt_noise.sigma_r);
    ROS_INFO("[gmmslam] async reg     : %s", cfg_.registration.enable_async ? "true" : "false");
    ROS_INFO("[gmmslam] keyframe (global graph / reg only): %.3f m | %.2f deg",
             cfg_.keyframe.translation_thresh_m, cfg_.keyframe.rotation_thresh_deg);

    // --- RNG ---
    if (cfg_.gt_noise.seed >= 0) {
        rng_.seed(static_cast<uint64_t>(cfg_.gt_noise.seed));
    } else {
        rng_.seed(std::random_device{}());
    }

    // --- GT state ---
    gt_path_.header.frame_id = cfg_.ros.odom_frame;

    // --- Publishers ---
    const auto path_pub         = pnh.advertise<nav_msgs::Path>("path", 1);
    const auto gg_path_pub      = pnh.advertise<nav_msgs::Path>("global_graph_path", 1);
    const auto odom_pub         = pnh.advertise<nav_msgs::Odometry>("odom", 1);
    const auto latest_frame_pub = pnh.advertise<sensor_msgs::PointCloud2>("latest_frame_cloud", 1);
    const auto map_cloud_pub    = pnh.advertise<sensor_msgs::PointCloud2>("map_cloud", 1, true);
    gt_path_pub_                = pnh.advertise<nav_msgs::Path>("gt_path", 1);
    gt_pose_pub_                = pnh.advertise<geometry_msgs::PoseStamped>("gt_pose", 1);
    const auto gmm_markers_pub  = pnh.advertise<visualization_msgs::MarkerArray>("gmm_markers", 1);
    const auto gmm_global_pub   = pnh.advertise<visualization_msgs::MarkerArray>("gmm_global_markers", 1, true);
    const auto gg_markers_pub   = pnh.advertise<visualization_msgs::MarkerArray>("global_graph_markers", 1, true);
    const auto graph_nodes_pub  = pnh.advertise<visualization_msgs::MarkerArray>("graph_nodes", 1, true);
    const auto loop_closure_markers_pub =
        pnh.advertise<visualization_msgs::MarkerArray>("loop_closure_markers", 10, true);
    const auto reg_request_pub  = nh.advertise<std_msgs::String>(
        cfg_.ros.registration_request_topic, 10);

    // --- Subsystems ---
    smoother_ = std::make_unique<FixedLagBackend>(
        cfg_.smoother, cfg_.gt_noise, cfg_.loop_closure, cfg_.imu);

    global_graph_ = std::make_unique<GlobalPoseGraph>(
        cfg_.global_graph,
        cfg_.registration,
        cfg_.loop_closure,
        cfg_.map,
        cfg_.ros.odom_frame,
        cfg_.gmm_dir,
        gg_path_pub,
        reg_request_pub,
        /*get_pose_fn=*/[this](int idx) -> std::optional<Matrix4d> {
            std::lock_guard<std::mutex> lk(smoother_->graph_lock);
            auto it = smoother_->pose_by_idx.find(idx);
            if (it == smoother_->pose_by_idx.end()) return std::nullopt;
            return it->second;
        },
        /*get_gmm_fn=*/[this](int idx) { return getKeyframeGmm(idx); },
        /*get_pose_uncertainty_fn=*/[this](int idx) -> std::optional<double> {
            std::lock_guard<std::mutex> lk(smoother_->graph_lock);
            auto it = smoother_->pose_uncertainty_by_idx.find(idx);
            if (it == smoother_->pose_uncertainty_by_idx.end()) return std::nullopt;
            return it->second;
        },
        /*get_traj_delta_fn=*/[this](int prev_key, int curr_key,
                                      double prev_t, double curr_t) {
            return submapTrajDeltaBetween(prev_key, curr_key, prev_t, curr_t);
        });

    registration_ = std::make_unique<RegistrationManager>(
        *smoother_,
        global_graph_.get(),
        reg_request_pub,
        cfg_.registration,
        cfg_.loop_closure,
        cfg_.solid,
        cfg_.sogmm,
        cfg_.gmm_dir,
        loop_closure_markers_pub,
        cfg_.ros.odom_frame);
    registration_->setOnFitComplete(
        [this](int /*frame_idx*/, const ros::Time& stamp) {
            if (global_graph_ && global_graph_->enable) {
                global_graph_->processPendingSubmapFinalizations(stamp);
            }
        });
    ROS_INFO("[gmmslam] loop_closure_markers -> %s (MarkerArray, latched; "
             "frame_id=%s)",
             (pnh.getNamespace() + "/loop_closure_markers").c_str(),
             cfg_.ros.odom_frame.c_str());

    Visualizer::Publishers vis_pubs;
    vis_pubs.path              = path_pub;
    vis_pubs.odom              = odom_pub;
    vis_pubs.latest_frame_cloud = latest_frame_pub;
    vis_pubs.map_cloud          = map_cloud_pub;
    vis_pubs.gmm_markers       = gmm_markers_pub;
    vis_pubs.gmm_global_markers = gmm_global_pub;
    vis_pubs.global_graph_markers = gg_markers_pub;
    vis_pubs.graph_nodes       = graph_nodes_pub;

    visualizer_ = std::make_unique<Visualizer>(
        *smoother_,
        *registration_,
        global_graph_.get(),
        cfg_.ros.odom_frame,
        cfg_.ros.base_frame,
        cfg_.visualization,
        std::move(vis_pubs));

    // --- Subscribers ---
    lidar_sub_ = nh.subscribe(
        cfg_.ros.lidar_topic, 1,
        &GMMSLAMNode::pclCallback, this);

    gt_sub_ = nh.subscribe(
        cfg_.ros.gt_topic, 1,
        &GMMSLAMNode::gtCallback, this);

    noisy_gt_sub_ = nh.subscribe(
        cfg_.ros.odometry_input, 1,
        &GMMSLAMNode::noisyGtCallback, this);

    if (!cfg_.ros.imu_topic.empty()) {
        imu_sub_ = nh.subscribe(
            cfg_.ros.imu_topic, 1,
            &GMMSLAMNode::imuCallback, this);
    }

    reg_result_sub_ = nh.subscribe(
        cfg_.ros.registration_result_topic, 50,
        &RegistrationManager::resultCallback, registration_.get());

    // --- Worker threads ---
    worker_threads_.emplace_back(&FixedLagBackend::backendLoop,
                                 smoother_.get(), std::cref(shutdown_));

    if (cfg_.registration.enable_async) {
        const int n_fit = std::clamp(cfg_.registration.workers, 1, 16);
        for (int i = 0; i < n_fit; ++i) {
            worker_threads_.emplace_back(&RegistrationManager::fitWorkerLoop,
                                         registration_.get(), std::cref(shutdown_));
        }
        ROS_INFO("[gmmslam] started %d SOGMM fit worker thread(s)", n_fit);
    }

    worker_threads_.emplace_back(&Visualizer::visLoop,
                                 visualizer_.get(), std::cref(shutdown_));

    // --- Wait for /clock when using sim time ---
    bool use_sim_time = false;
    nh.param("/use_sim_time", use_sim_time, false);
    if (use_sim_time) {
        ROS_INFO("[gmmslam] use_sim_time=true, waiting for /clock ...");
        while (!ros::isShuttingDown() && ros::Time::now().isZero()) {
            ros::Duration(0.1).sleep();
        }
        ROS_INFO("[gmmslam] clock started");
    }

    gt_init_start_time_ = ros::Time::now();
    ROS_INFO("[gmmslam] node ready, waiting for point clouds ...");
}

// =====================================================================
// Destructor
// =====================================================================

GMMSLAMNode::~GMMSLAMNode()
{
    shutdown_.store(true);
    for (auto& t : worker_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// =====================================================================
// GT callback
// =====================================================================

void GMMSLAMNode::gtCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    latest_gt_pose_raw_ = msg;
    if (!ensureGtOriginInitialized(msg->header.stamp)) {
        return;
    }

    const Matrix4d T_gt = poseMsgToMatrix(msg->pose);
    const Matrix4d T_odom = gt_origin_inv_.value() * T_gt;
    const auto ps = poseToPoseStamped(T_odom, msg->header.stamp, cfg_.ros.odom_frame);

    gt_path_.header.stamp = ps.header.stamp;
    gt_path_.poses.push_back(ps);
    gt_pose_pub_.publish(ps);
    gt_path_pub_.publish(gt_path_);
}

// =====================================================================
// Noisy GT callback
// =====================================================================

void GMMSLAMNode::noisyGtCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    latest_noisy_gt_msg_ = msg;
    const double t = stampToSec(msg->header.stamp);
    const Matrix4d T = poseMsgToMatrix(msg->pose);

    noisy_gt_buffer_.push_back({t, T});
    while (static_cast<int>(noisy_gt_buffer_.size()) > kNoisyGtBufferMax) {
        noisy_gt_buffer_.pop_front();
    }
}

// =====================================================================
// IMU callback
// =====================================================================

void GMMSLAMNode::imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
    const Vector3d acc(msg->linear_acceleration.x,
                       msg->linear_acceleration.y,
                       msg->linear_acceleration.z);
    const Vector3d gyro(msg->angular_velocity.x,
                        msg->angular_velocity.y,
                        msg->angular_velocity.z);

    imu_buffer_.push_back({msg->header.stamp, acc, gyro});

    const double t_now = stampToSec(msg->header.stamp);
    const double keep_s = std::max(1.0, cfg_.imu.buffer_keep_s);
    while (!imu_buffer_.empty() &&
           (t_now - stampToSec(imu_buffer_.front().stamp)) > keep_s) {
        imu_buffer_.pop_front();
    }
}

// =====================================================================
// ensureGtOriginInitialized
// =====================================================================

bool GMMSLAMNode::ensureGtOriginInitialized(const ros::Time& stamp)
{
    if (gt_origin_inv_.has_value()) {
        return true;
    }

    ros::Time now = ros::Time::now();
    if (now.isZero()) {
        now = stamp;
    }
    const double elapsed = stampToSec(now) - stampToSec(gt_init_start_time_);
    if (elapsed < cfg_.gt_noise.init_wait_s) {
        ROS_WARN_THROTTLE(2.0, "[gmmslam] GT init window (%.2f/%.2fs)",
                          elapsed, cfg_.gt_noise.init_wait_s);
        return false;
    }
    if (!latest_gt_pose_raw_) {
        ROS_WARN_THROTTLE(2.0,
            "[gmmslam] GT init window elapsed but no GT pose received yet");
        return false;
    }

    const Matrix4d T0 = poseMsgToMatrix(latest_gt_pose_raw_->pose);
    gt_origin_inv_ = T0.inverse();

    gt_path_ = nav_msgs::Path();
    gt_path_.header.frame_id = cfg_.ros.odom_frame;

    ROS_INFO("[gmmslam] GT origin initialized after %.2fs", elapsed);
    return true;
}

// =====================================================================
// shouldAddKeyframe
// =====================================================================

bool GMMSLAMNode::shouldAddKeyframe(const ros::Time& stamp,
                                    const Matrix4d& current_pose)
{
    if (keyframe_count_ == 0 || !has_last_keyframe_) {
        return true;
    }

    if (cfg_.keyframe.use_time_trigger) {
        const double dt = stampToSec(stamp) - last_keyframe_t_sec_;
        if (dt >= cfg_.keyframe.max_interval_s) {
            return true;
        }
    }

    const Matrix4d T_rel = last_keyframe_pose_.inverse() * current_pose;

    const double dtrans = T_rel.block<3,1>(0,3).norm();
    if (dtrans >= cfg_.keyframe.translation_thresh_m) {
        return true;
    }

    const Eigen::AngleAxisd aa(T_rel.block<3,3>(0,0));
    const double drot_deg = std::abs(aa.angle()) * 180.0 / M_PI;
    if (drot_deg >= cfg_.keyframe.rotation_thresh_deg) {
        return true;
    }

    return false;
}

// =====================================================================
// lookupNoisyGtAt — nearest-neighbour lookup in the ring buffer
// =====================================================================

std::optional<Matrix4d> GMMSLAMNode::lookupNoisyGtAt(double t_sec) const
{
    if (noisy_gt_buffer_.empty()) {
        return std::nullopt;
    }

    int best_i = 0;
    double best_dt = std::abs(noisy_gt_buffer_[0].t_sec - t_sec);
    for (int i = 1; i < static_cast<int>(noisy_gt_buffer_.size()); ++i) {
        const double dt = std::abs(noisy_gt_buffer_[i].t_sec - t_sec);
        if (dt < best_dt) {
            best_dt = dt;
            best_i = i;
        } else if (noisy_gt_buffer_[i].t_sec > t_sec) {
            break;
        }
    }
    return noisy_gt_buffer_[best_i].pose;
}

// =====================================================================
// currentGtPoseOdomFrame — absolute pose in odom frame (same convention as factors)
// =====================================================================

std::optional<Matrix4d> GMMSLAMNode::currentGtPoseOdomFrame(
    const ros::Time& stamp) const {
    const double t_cloud = stampToSec(stamp);
    if (!noisy_gt_buffer_.empty()) {
        std::optional<Matrix4d> T_curr = lookupNoisyGtAt(t_cloud);
        if (T_curr.has_value()) {
            return T_curr;
        }
        if (latest_noisy_gt_msg_) {
            return poseMsgToMatrix(latest_noisy_gt_msg_->pose);
        }
        return std::nullopt;
    }
    if (!latest_gt_pose_raw_ || !gt_origin_inv_.has_value()) {
        return std::nullopt;
    }
    const Matrix4d T_gt = poseMsgToMatrix(latest_gt_pose_raw_->pose);
    return gt_origin_inv_.value() * T_gt;
}

// =====================================================================
// maybeReanchorSmootherFromGt
// =====================================================================

std::optional<Matrix4d> GMMSLAMNode::loadRestartPose() const {
    if (cfg_.ros.restart_state_path.empty()) {
        return std::nullopt;
    }

    std::ifstream in(cfg_.ros.restart_state_path);
    if (!in.good()) {
        return std::nullopt;
    }

    double t_sec = 0.0;
    int key_idx = -1;
    Matrix4d T = Matrix4d::Identity();
    if (!(in >> t_sec >> key_idx)) {
        ROS_WARN("[gmmslam] restart state %s is malformed; ignoring",
                 cfg_.ros.restart_state_path.c_str());
        return std::nullopt;
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!(in >> T(r, c))) {
                ROS_WARN("[gmmslam] restart state %s has incomplete pose; ignoring",
                         cfg_.ros.restart_state_path.c_str());
                return std::nullopt;
            }
        }
    }
    if (!T.allFinite()) {
        ROS_WARN("[gmmslam] restart state %s has non-finite pose; ignoring",
                 cfg_.ros.restart_state_path.c_str());
        return std::nullopt;
    }

    const auto p = T.block<3,1>(0,3);
    ROS_WARN("[gmmslam] loaded restart pose from %s (saved X(%d), t=%.3f): "
             "pos=[%.3f, %.3f, %.3f]",
             cfg_.ros.restart_state_path.c_str(), key_idx, t_sec,
             p(0), p(1), p(2));
    return T;
}

void GMMSLAMNode::saveRestartPose(const Matrix4d& T, double t_sec, int key_idx) {
    if (cfg_.ros.restart_state_path.empty() || !T.allFinite()) {
        return;
    }
    if (last_restart_state_write_t_ >= 0.0 &&
        (t_sec - last_restart_state_write_t_) < 0.10) {
        return;
    }
    last_restart_state_write_t_ = t_sec;

    try {
        const std::filesystem::path path(cfg_.ros.restart_state_path);
        const auto parent = path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        const std::filesystem::path tmp =
            path.string() + ".tmp." + std::to_string(::getpid());

        {
            std::ofstream out(tmp);
            out << std::setprecision(17) << t_sec << ' ' << key_idx;
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    out << ' ' << T(r, c);
                }
            }
            out << '\n';
            out.flush();
            if (!out.good()) {
                throw std::runtime_error("write failed");
            }
        }

        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(tmp, path, ec);
        }
        if (ec) {
            ROS_WARN_THROTTLE(5.0,
                "[gmmslam] failed to update restart state %s: %s",
                cfg_.ros.restart_state_path.c_str(), ec.message().c_str());
        }
    } catch (const std::exception& e) {
        ROS_WARN_THROTTLE(5.0,
            "[gmmslam] failed to save restart state %s: %s",
            cfg_.ros.restart_state_path.c_str(), e.what());
    }
}

void GMMSLAMNode::maybeReanchorSmootherFromGt(const ros::Time& stamp,
                                              double t_cloud) {
    if (!cfg_.global_graph.reanchor_smoother_on_traj_gate_fail || !global_graph_ ||
        !global_graph_->enable) {
        return;
    }
    if (!global_graph_->consumeSmootherReanchorRequest()) {
        return;
    }
    if (odom_idx_ <= 0) {
        return;
    }
    const std::optional<Matrix4d> T_gt = currentGtPoseOdomFrame(stamp);
    if (!T_gt.has_value()) {
        ROS_WARN("[gmmslam] smoother re-anchor requested but GT pose unavailable");
        return;
    }
    const int anchor = odom_idx_ - 1;
    smoother_->scheduleReanchorToGt(anchor, T_gt.value(), t_cloud);
    last_gt_T_for_factor_ = T_gt.value();
    accumulated_gt_rel_.reset();
    ROS_WARN("[gmmslam] Scheduled fixed-lag re-anchor to GT at X(%d) "
             "(smoothed trajectory hit submap aux gate — SLAM estimate was reset to GT)",
             anchor);
}

// =====================================================================
// sampleNoisyGtRelativePose
// =====================================================================

std::optional<Matrix4d> GMMSLAMNode::sampleNoisyGtRelativePose(const ros::Time& stamp)
{
    const double t_cloud = stampToSec(stamp);

    // Prefer external noisy GT topic
    if (!noisy_gt_buffer_.empty()) {
        std::optional<Matrix4d> T_curr_opt = lookupNoisyGtAt(t_cloud);
        if (!T_curr_opt.has_value() && latest_noisy_gt_msg_) {
            T_curr_opt = poseMsgToMatrix(latest_noisy_gt_msg_->pose);
        }
        if (!T_curr_opt.has_value()) {
            return std::nullopt;
        }
        const Matrix4d T_curr = T_curr_opt.value();

        if (!last_gt_T_for_factor_.has_value()) {
            last_gt_T_for_factor_ = T_curr;
            last_gt_T_stamp_sec_ = t_cloud;
            return std::nullopt;
        }
        const Matrix4d T_prev = last_gt_T_for_factor_.value();
        last_gt_T_for_factor_ = T_curr;
        last_gt_T_stamp_sec_ = t_cloud;
        return T_prev.inverse() * T_curr;
    }

    // Fallback: use raw GT + internally sampled noise
    if (!latest_gt_pose_raw_ || !gt_origin_inv_.has_value()) {
        return std::nullopt;
    }
    ROS_WARN_THROTTLE(5.0,
        "[gmmslam] odometry_input unavailable; falling back to internal GT noise");

    const Matrix4d T_gt = poseMsgToMatrix(latest_gt_pose_raw_->pose);
    const Matrix4d T_curr_gt = gt_origin_inv_.value() * T_gt;

    if (!last_gt_T_for_factor_.has_value()) {
        last_gt_T_for_factor_ = T_curr_gt;
        return std::nullopt;
    }
    const Matrix4d T_prev_gt = last_gt_T_for_factor_.value();
    const Matrix4d T_rel_gt = T_prev_gt.inverse() * T_curr_gt;

    // Sample rotation noise as a small-angle axis-angle
    const Vector3d rot_noise(
        normal_dist_(rng_) * cfg_.gt_noise.sigma_r,
        normal_dist_(rng_) * cfg_.gt_noise.sigma_r,
        normal_dist_(rng_) * cfg_.gt_noise.sigma_r);
    const Eigen::AngleAxisd aa(rot_noise.norm(),
                               rot_noise.norm() > 1e-12
                                   ? rot_noise.normalized()
                                   : Vector3d::UnitX());
    const Matrix3d R_noise = aa.toRotationMatrix();

    const Vector3d trans_noise(
        normal_dist_(rng_) * cfg_.gt_noise.sigma_t,
        normal_dist_(rng_) * cfg_.gt_noise.sigma_t,
        normal_dist_(rng_) * cfg_.gt_noise.sigma_t);

    Matrix4d T_noisy = Matrix4d::Identity();
    T_noisy.block<3,3>(0,0) = R_noise * T_rel_gt.block<3,3>(0,0);
    T_noisy.block<3,1>(0,3) = T_rel_gt.block<3,1>(0,3) + trans_noise;

    last_gt_T_for_factor_ = T_curr_gt;
    return T_noisy;
}

// =====================================================================
// imuMeasurementsBetween
// =====================================================================

std::vector<std::tuple<double, Vector3d, Vector3d>>
GMMSLAMNode::imuMeasurementsBetween(double t_prev, double t_curr) const
{
    std::vector<std::tuple<double, Vector3d, Vector3d>> samples;
    if (t_curr <= t_prev || imu_buffer_.empty()) {
        return samples;
    }

    struct Entry { double t; Vector3d acc; Vector3d gyro; };

    // Latest IMU sample at or before t_prev (scan full buffer). The old logic
    // only looked in [t_prev-50ms, t_curr]; if every sample there has t > t_prev
    // (low-rate IMU, sim timing, or short stride dt), seed was null → samples=0.
    bool have_seed = false;
    Entry seed{};
    for (const auto& e : imu_buffer_) {
        const double ts = stampToSec(e.stamp);
        if (ts <= t_prev) {
            if (!have_seed || ts >= seed.t) {
                seed = Entry{ts, e.acc, e.gyro};
                have_seed = true;
            }
        }
    }
    if (!have_seed) {
        return samples;
    }

    std::vector<Entry> after_prev;
    after_prev.reserve(imu_buffer_.size());
    for (const auto& e : imu_buffer_) {
        const double ts = stampToSec(e.stamp);
        if (ts > t_prev && ts <= t_curr) {
            after_prev.push_back({ts, e.acc, e.gyro});
        }
    }
    std::sort(after_prev.begin(), after_prev.end(),
              [](const Entry& a, const Entry& b) { return a.t < b.t; });

    double t_last = t_prev;
    Vector3d acc_last = seed.acc;
    Vector3d gyro_last = seed.gyro;

    for (const auto& e : after_prev) {
        const double dt = e.t - t_last;
        if (dt > 1e-6) {
            samples.emplace_back(dt, acc_last, gyro_last);
            t_last = e.t;
        }
        acc_last = e.acc;
        gyro_last = e.gyro;
    }

    const double dt_tail = t_curr - t_last;
    if (dt_tail > 1e-6) {
        samples.emplace_back(dt_tail, acc_last, gyro_last);
    }
    return samples;
}

// =====================================================================
// submapTrajDeltaBetween
// =====================================================================

std::optional<Matrix4d> GMMSLAMNode::submapTrajDeltaBetween(
    int prev_key, int curr_key, double /*prev_t*/, double /*curr_t*/)
{
    std::lock_guard<std::mutex> lk(smoother_->graph_lock);
    auto it_prev = smoother_->pose_by_idx.find(prev_key);
    auto it_curr = smoother_->pose_by_idx.find(curr_key);
    if (it_prev == smoother_->pose_by_idx.end() ||
        it_curr == smoother_->pose_by_idx.end()) {
        return std::nullopt;
    }
    return it_prev->second.inverse() * it_curr->second;
}

// =====================================================================
// getKeyframeGmm
// =====================================================================

std::optional<std::pair<GmmModel, Matrix4d>>
GMMSLAMNode::getKeyframeGmm(int key_idx)
{
    std::lock_guard<std::mutex> lk(registration_->lock);
    auto it = registration_->local_gmms_by_idx.find(key_idx);
    if (it == registration_->local_gmms_by_idx.end()) {
        return std::nullopt;
    }
    const auto& entry = it->second;
    const Matrix4d best_pose = entry.has_map_pose ? entry.map_pose
                             : entry.has_capture_pose ? entry.capture_pose
                             : Matrix4d::Identity();
    return std::make_pair(entry.model, best_pose);
}

// =====================================================================
// logBackpressure
// =====================================================================

void GMMSLAMNode::logBackpressure(const ros::Time& stamp)
{
    const double t_sec = stampToSec(stamp);
    if ((t_sec - last_backpressure_log_t_) < 5.0) {
        return;
    }
    last_backpressure_log_t_ = t_sec;

    const int dropped_fits = registration_->dropped_fit_frames.exchange(0);
    const int dropped_results = registration_->dropped_result_msgs.exchange(0);
    if (dropped_fits > 0 || dropped_results > 0) {
        ROS_WARN("[gmmslam] async reg overloaded: "
                 "dropped fits=%d, dropped results=%d",
                 dropped_fits, dropped_results);
    }

    const int deferred = smoother_->deferred_batches.exchange(0);
    if (deferred > 0) {
        ROS_WARN("[gmmslam] backend overloaded: deferred updates=%d", deferred);
    }
}

// =====================================================================
// pclCallback — main pipeline
// =====================================================================

void GMMSLAMNode::pclCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    try {
        if (!first_cloud_seen_) {
            ROS_INFO("[gmmslam] first point cloud received, processing started");
            first_cloud_seen_ = true;
        }

        const ros::Time stamp = msg->header.stamp;
        const double t_cloud = stampToSec(stamp);

        // Monotonicity check
        if (last_cloud_t_sec_processed_ >= 0.0 &&
            t_cloud <= last_cloud_t_sec_processed_) {
            ROS_WARN_THROTTLE(2.0,
                "[gmmslam] non-monotonic cloud stamp "
                "(%.6f <= %.6f); skipping frame",
                t_cloud, last_cloud_t_sec_processed_);
            return;
        }
        last_cloud_t_sec_processed_ = t_cloud;

        if (!ensureGtOriginInitialized(stamp)) {
            return;
        }

        maybeReanchorSmootherFromGt(stamp, t_cloud);

        // 1. Convert & preprocess
        Eigen::MatrixXf pts = pc2ToEigen(*msg);
        if (pts.rows() == 0) {
            ROS_WARN_THROTTLE(5.0,
                "[gmmslam] received empty point cloud, skipping");
            return;
        }
        pts = preprocess(pts, cfg_.preprocess.min_range,
                         cfg_.preprocess.max_range,
                         cfg_.preprocess.voxel_leaf_size,
                         cfg_.preprocess.target_points);
        if (pts.rows() < cfg_.preprocess.min_points) {
            ROS_WARN_THROTTLE(5.0,
                "[gmmslam] only %d points after filtering, skipping",
                static_cast<int>(pts.rows()));
            return;
        }

        // 2. Drain pending async registration results
        if (cfg_.registration.enable_async) {
            registration_->drainResults(stamp);
        }

        // 3. Per-frame GT relative motion
        const auto gt_rel_opt = sampleNoisyGtRelativePose(stamp);

        // 4. Predicted pose = current smoother pose + GT motion
        Matrix4d predicted;
        {
            std::lock_guard<std::mutex> lk(smoother_->graph_lock);
            if (!smoother_->initialized() && restart_pose_.has_value()) {
                predicted = restart_pose_.value();
            } else if (gt_rel_opt.has_value()) {
                predicted = smoother_->pose * gt_rel_opt.value();
            } else {
                predicted = smoother_->pose;
            }
        }

        // 5. Accumulate GT relative motion for the next smoother update
        if (gt_rel_opt.has_value()) {
            if (accumulated_gt_rel_.has_value()) {
                accumulated_gt_rel_ = accumulated_gt_rel_.value() * gt_rel_opt.value();
            } else {
                accumulated_gt_rel_ = gt_rel_opt.value();
            }
        }

        // 6. Feed the fixed-lag smoother every N-th frame
        ++smoother_frame_counter_;
        const bool is_smoother_frame =
            (smoother_frame_counter_ % cfg_.smoother.smoother_stride) == 0;

        Matrix4d T_pub;
        int smoother_pose_key = -1;
        if (is_smoother_frame) {
            std::vector<std::tuple<double, Vector3d, Vector3d>> imu_measurements;
            if (has_last_smoother_stamp_) {
                imu_measurements = imuMeasurementsBetween(
                    stampToSec(last_smoother_stamp_), stampToSec(stamp));
            }
            if (cfg_.imu.enable_preintegration) {
                ROS_DEBUG("[gmmslam] frame %d: imu samples for preintegration=%zu",
                          frame_count_, imu_measurements.size());
            }

            const int curr_odom = odom_idx_;
            smoother_pose_key = curr_odom;
            const int prev_odom = curr_odom - 1;

            const Matrix4d* gt_rel_ptr = accumulated_gt_rel_.has_value()
                                         ? &accumulated_gt_rel_.value()
                                         : nullptr;
            const auto* imu_ptr = imu_measurements.empty()
                                  ? nullptr : &imu_measurements;

            smoother_->addFrame(prev_odom, curr_odom, stamp,
                                predicted, gt_rel_ptr, imu_ptr);

            last_smoother_stamp_ = stamp;
            has_last_smoother_stamp_ = true;
            ++odom_idx_;
            accumulated_gt_rel_.reset();

            if (curr_odom % 10 == 0) {
                const auto p = predicted.block<3,1>(0,3);
                const bool has_odom = gt_rel_opt.has_value();
                ROS_INFO("[gmmslam] X(%d) pos=[%.3f, %.3f, %.3f] odom=%s",
                         curr_odom, p(0), p(1), p(2),
                         has_odom ? "GT" : "LOST");
            }

            // Single post-addFrame lock: read current pose and pose_by_idx
            Matrix4d current_pose, T_curr;
            {
                std::lock_guard<std::mutex> lk(smoother_->graph_lock);
                current_pose = smoother_->pose;
                auto it = smoother_->pose_by_idx.find(curr_odom);
                T_curr = (it != smoother_->pose_by_idx.end())
                         ? it->second : current_pose;
            }
            saveRestartPose(T_curr, stampToSec(stamp), curr_odom);

            // 7. Keyframe check (global graph + registration only)
            if (shouldAddKeyframe(stamp, current_pose)) {
                last_keyframe_pose_ = current_pose;
                last_keyframe_t_sec_ = stampToSec(stamp);
                has_last_keyframe_ = true;
                keyframe_odom_indices_.push_back(curr_odom);
                ++keyframe_count_;

                ROS_INFO("[gmmslam][DBG] NEW KEYFRAME X(%d) "
                         "keyframe_count=%d pts=%ld",
                         curr_odom, keyframe_count_,
                         static_cast<long>(pts.rows()));

                global_graph_->updateWithKeyframe(
                    curr_odom, stamp, T_curr, stampToSec(stamp));

                // SOLiD descriptor first — cheap and synchronous — so that
                // the subsequent async GMM fit's loop search (triggered
                // from finishFit) can already query by appearance.
                registration_->submitKeyframeDescriptor(curr_odom, pts);

                const bool gate_async  = cfg_.registration.enable_async;
                const bool gate_modulo =
                    (keyframe_count_ % cfg_.registration.request_every_n_frames == 0);
                const bool gate_resume =
                    (curr_odom >= reg_enqueue_resume_frame_);
                ROS_INFO("[gmmslam][DBG] enqueue gate X(%d) async=%d "
                         "kf_count%%%d==0: %d (kf_count=%d) "
                         "resume_frame=%d (curr=%d, pass=%d)",
                         curr_odom, gate_async ? 1 : 0,
                         cfg_.registration.request_every_n_frames,
                         gate_modulo ? 1 : 0, keyframe_count_,
                         reg_enqueue_resume_frame_, curr_odom,
                         gate_resume ? 1 : 0);

                if (gate_async && gate_modulo && gate_resume)
                {
                    const double capture_t = stampToSec(stamp);
                    const bool ok = registration_->enqueueFit(
                        curr_odom, stamp, pts, capture_t, T_curr);
                    ROS_INFO("[gmmslam][DBG] enqueueFit result X(%d) ok=%d",
                             curr_odom, ok ? 1 : 0);
                    if (!ok && cfg_.registration.enqueue_cooldown_frames > 0) {
                        reg_enqueue_resume_frame_ =
                            curr_odom + cfg_.registration.enqueue_cooldown_frames;
                        ROS_WARN("[gmmslam][DBG] enqueue cooldown: "
                                 "resume_frame set to %d",
                                 reg_enqueue_resume_frame_);
                    }
                }
            }

            T_pub = current_pose;
        } else {
            // Non-smoother frame: propagate prediction without GTSAM
            {
                std::lock_guard<std::mutex> lk(smoother_->graph_lock);
                smoother_->pose = predicted;
            }
            T_pub = predicted;
        }

        // 8. Publish
        visualizer_->publishPoseOnly(T_pub, stamp);
        visualizer_->enqueueFrame(stamp, pts, frame_count_, T_pub, smoother_pose_key);

        logBackpressure(stamp);
        ++frame_count_;

    } catch (const std::exception& e) {
        ROS_ERROR("[gmmslam] exception in cloud callback: %s", e.what());
    }
}

} // namespace gmmslam

// =====================================================================
// main
// =====================================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gmmslam_node");
    ros::NodeHandle nh, pnh("~");

    gmmslam::GMMSLAMNode node(nh, pnh);
    ros::spin();
    return 0;
}
