#pragma once
#include <string>

namespace gmmslam {

struct RosConfig {
    std::string lidar_topic = "/m500_1/mpa/depth/points";
    std::string gt_topic = "/m500_1/mavros/local_position/pose";
    std::string imu_topic = "";
    std::string sensor_frame = "depth_camera_1";
    std::string odom_frame = "world";
    std::string base_frame = "m500_1_base_link";
    /// geometry_msgs/PoseStamped odometry-style input (e.g. noisy GT publisher).
    std::string odometry_input = "/gmmslam_node/noisy_gt_pose";
    /// Persist latest odom-frame pose here; on respawn, seed the smoother from it.
    /// Empty string disables restart pose persistence.
    std::string restart_state_path = "/tmp/gmmslam_restart_pose.txt";
    std::string registration_request_topic = "/gmmslam_node/registration/request";
    std::string registration_result_topic = "/gmmslam_node/registration/result";
};

struct PreprocessConfig {
    double min_range = 0.1;
    double max_range = 25.0;
    double voxel_leaf_size = 0.05;
    /// If > 0, randomly subsample to at most this many points after range filter
    /// and optional voxel (uniform subset without replacement).
    int target_points = 0;
    int min_points = 50;
};

struct SogmmConfig {
    double bandwidth = 0.02;
    int max_points = 2000;
    int n_components = 0;
    std::string compute = "GPU";
    /// "sogmm" preserves the legacy GIRA/SOGMM fitter. "gmmap" uses the
    /// optional GMMap adapter when the input cloud is organized and GMMap was
    /// enabled at build time.
    std::string backend = "sogmm";
    std::string gmmap_dataset = "tum";
    int gmmap_num_threads = 0;
    bool gmmap_measure_memory = true;
    std::string gmmap_frame_alg_name = "spgf_extended";
    double gmmap_max_depth = 0.0;
    double gmmap_hell_thresh_squared_oversized_gau = 0.20;
    double gmmap_hell_thresh_squared_free = 0.10;
    double gmmap_hell_thresh_squared_obs_scale = 1.0;
    double gmmap_hell_thresh_squared_min = 0.01;
    double gmmap_min_gaussian_length = 0.05;
    double gmmap_frame_max_scale = 4.0;
    double gmmap_fusion_max_scale = 4.0;
    double gmmap_fusion_bound = 2.0;
    double gmmap_rtree_bound_scale = 2.0;
    double gmmap_depth_scale = 1.0;
    bool gmmap_track_color = false;
    bool gmmap_track_intensity = false;
    bool gmmap_cur_debug_frame = false;
    int gmmap_min_num_neighbor_clusters = 3;
    int gmmap_occ_x_threshold = 2;
    double gmmap_noise_threshold = 0.05;
    double gmmap_line_threshold = 0.05;
    int gmmap_sparse_threshold = 8;
    int gmmap_ncheck_threshold = 20;
    int gmmap_num_line_threshold = 3;
    int gmmap_num_pixels_threshold = 6;
    int gmmap_max_incomplete_clusters = 20;
    double gmmap_adaptive_threshold_scale = 1.2;
    double gmmap_noise_floor = 0.05;
    double gmmap_angle_threshold = 0.95;
    double gmmap_free_space_dist_scale = 1.0;
    int gmmap_debug_row_idx = -1;
    bool gmmap_far_fill_enable = true;
    double gmmap_far_fill_start_m = 10.0;
    double gmmap_far_fill_voxel_m = 1.0;
    int gmmap_far_fill_max_components = 300;
    double gmmap_far_fill_skip_max_depth_margin_m = 0.5;
    bool gmmap_estimate_intrinsics = false;
    double gmmap_horizontal_fov_deg = 120.0;
    double gmmap_fx = 0.0;
    double gmmap_fy = 0.0;
    double gmmap_cx = -1.0;
    double gmmap_cy = -1.0;
};

struct SmootherConfig {
    double fixed_lag_s = 4.0;
    int smoother_stride = 3;
    double odom_noise_sigma_t = 0.03;
    double odom_noise_sigma_r = 0.03;
    double lost_scale = 10.0;
    double prior_scale = 0.1;
    int pose_history_keep = 5000;
};

struct RegistrationConfig {
    bool enable_async = true;
    int request_every_n_frames = 3;
    int factor_every_n_frames = 3;
    double score_threshold = 0.3;
    double strong_factor_score_threshold = 2.0;
    double strong_factor_sigma_t = 0.01;
    double strong_factor_sigma_r = 0.01;
    int queue_size = 8;
    int result_queue_size = 64;
    int enqueue_cooldown_frames = 8;
    int workers = 6;
    bool compensate_fit_latency = true;
    double score_sigma_low = 0.6;
    double score_sigma_high = 1.7;
    double seq_sigma_t_min = 0.02;
    double seq_sigma_t_max = 0.20;
    double seq_sigma_r_min = 0.01;
    double seq_sigma_r_max = 0.15;
    double loop_sigma_t_min = 0.03;
    double loop_sigma_t_max = 0.40;
    double loop_sigma_r_min = 0.02;
    double loop_sigma_r_max = 0.25;
};

struct LoopClosureConfig {
    bool enable = true;
    double search_radius_m = 2.5;
    int min_keyframe_gap = 10;
    int max_candidates = 5;
    int request_every_n_keyframes = 2;
    int search_cooldown_keyframes = 2;
    double min_separation_m = 0.8;
    double min_separation_deg = 20.0;
    double max_age_s = 720.0;
    int gmm_keep_keyframes = 15000;
    double detect_score_threshold = 1.1;
    double super_sigma_t = 0.02;
    double super_sigma_r = 0.02;
    double consistency_gate_trans_m = 2.0;
    double consistency_gate_rot_deg = 45.0;
};

// SOLiD (Kim et al. 2024) — spatially organized, lightweight global descriptor
// used as an appearance-based gate for the radius-based loop search and as a
// rescue index when the smoother's pose is not trustworthy. Tuned here for a
// 120° horizontal FOV depth camera, not for 360° rotating LiDAR.
struct SolidConfig {
    bool enable = true;

    // Geometry / binning. Must match the physical sensor when the descriptor
    // is produced in the sensor frame.
    double fov_up_deg    = 30.0;   // upper vertical FOV boundary
    double fov_down_deg  = -30.0;  // lower vertical FOV boundary
    double min_distance_m = 0.5;
    double max_distance_m = 25.0;
    int num_angle  = 36;           // azimuth bins over 360° (10°/bin)
    int num_range  = 32;           // radial bins up to max_distance_m
    int num_height = 16;           // vertical bins across [fov_down, fov_up]
    // Extra voxel leaf applied to the already-preprocessed cloud. <=0 disables.
    double voxel_size_m = 0.0;

    // Matching gate.
    double cos_similarity_threshold = 0.85;  // range-vec cosine accept cutoff
    double radius_weight     = 0.5;          // α in fused score
    double appearance_weight = 0.5;          // β in fused score

    // Yaw estimation for T_init seeding.
    bool   provide_yaw_prior   = true;
    double overlap_min         = 0.3;   // min non-zero overlap fraction accepted
    double max_abs_yaw_deg     = 180.0; // restrict shift search to |yaw| <= this
    int    yaw_sign            = +1;    // empirical sign flip: +1 or -1

    // Rescue path: run a full-index top-K search when no loop has been added
    // in the last `rescue_trigger_silence_kf` keyframes, throttled by
    // `rescue_every_n_kf`.
    bool   rescue_enable          = true;
    int    rescue_every_n_kf      = 10;
    int    rescue_trigger_silence_kf = 30;
    int    rescue_top_k           = 3;
    double rescue_cos_threshold   = 0.9;

    // In-memory cache sizing. 0 => unlimited (trimmed by gmm_keep_keyframes).
    int keep_descriptors = 0;
};

struct KeyframeConfig {
    double translation_thresh_m = 0.3;
    double rotation_thresh_deg = 7.0;
    bool use_time_trigger = false;
    double max_interval_s = 10.0;
};

struct GlobalGraphConfig {
    bool enable = true;
    int submap_keyframes_per_submap = 10;
    double between_sigma_t = 0.08;
    double between_sigma_r = 0.08;
    double prior_sigma_t = 0.02;
    double prior_sigma_r = 0.02;
    double overlap_radius_m = 3.0;
    int max_overlap_registrations = 3;
    double reg_score_threshold = 0.5;
    int min_loop_submap_gap = 5;
    bool enable_traj_aux_factors = false;
    double traj_sigma_t = 0.15;
    double traj_sigma_r = 0.15;
    double aux_gate_abs_trans_m = 6.0;
    double aux_gate_abs_rot_deg = 50.0;
    double aux_gate_consistency_trans_m = 2.0;
    double aux_gate_consistency_rot_deg = 35.0;
    /// Separate limits for the sequential submap *trajectory* aux factor (spanning
    /// ~submap_keyframes_per_submap keyframes). If unset in YAML, loadGlobalGraph
    /// copies from aux_gate_* after those are read.
    double traj_aux_gate_abs_trans_m = 6.0;
    double traj_aux_gate_abs_rot_deg = 55.0;
    double traj_aux_gate_consistency_trans_m = 2.0;
    double traj_aux_gate_consistency_rot_deg = 40.0;
    double submap_loop_sigma_t_min = 0.05;
    double submap_loop_sigma_t_max = 0.50;
    double submap_loop_sigma_r_min = 0.03;
    double submap_loop_sigma_r_max = 0.30;
    /// When the submap trajectory auxiliary factor is rejected because the fixed-lag
    /// estimate implies an absurd motion (|t|/|r| above aux gates), schedule re-
    /// anchoring the smoother on the current GT pose so published SLAM pose stops
    /// drifting from simulation truth.
    bool reanchor_smoother_on_traj_gate_fail = true;
    /// Before merging keyframe GMMs into a closed submap, require at least this many
    /// keyframes to have finished SOGMM fits (clamped to the submap's keyframe count).
    /// 1 preserves legacy behavior (finalize as soon as any GMM is ready). 0 lets
    /// the fraction rule alone set the threshold (minimum remains 1 ready keyframe).
    int submap_finalize_min_ready_keyframes = 1;
    /// If in (0,1], also require ceil(fraction * N) ready keyframes where N is the number
    /// of keyframes in the submap. Combined with min_ready_keyframes via max(...).
    /// 0 disables the fraction rule.
    double submap_finalize_min_ready_fraction = 0.0;
    /// If >0, after this many seconds waiting on the readiness gate, finalize using
    /// whatever keyframe GMMs are available (>=1). 0 disables the fallback.
    double submap_finalize_max_wait_s = 0.0;
};

struct GtNoiseConfig {
    double init_wait_s = 3.0;
    double sigma_t = 0.01;
    double sigma_r = 0.01;
    double factor_sigma_t = 0.015;
    double factor_sigma_r = 0.015;
    int seed = -1;
    /// If > 0, optional cap on accepted GT step size (m); 0 = unused until wired in node.
    double jump_reject_trans_m = 0.0;
    double jump_reject_rot_deg = 0.0;
};

struct ImuConfig {
    bool enable_preintegration = false;
    double gravity_mps2 = 9.81;
    double buffer_keep_s = 30.0;
    double acc_noise_sigma = 0.001;
    double gyro_noise_sigma = 0.001;
    double integration_sigma = 0.0001;
    double bias_acc_rw_sigma = 0.001;
    double bias_gyro_rw_sigma = 0.0001;
    double velocity_prior_sigma = 1.0;
    double bias_prior_sigma = 0.1;
};

struct VisualizationConfig {
    double gmm_marker_sigma = 3.0;
    bool global_gmm_markers_enable = true;
    double global_gmm_publish_period_s = 1.0;
    bool d2d_frame_to_frame_text_enable = true;
    bool d2d_submap_overlap_text_enable = true;
    bool d2d_loop_closure_text_enable = true;
    /// First-order low-pass cutoff for published TF/odom/RViz pose. <=0 disables.
    double output_pose_lpf_cutoff_hz = 0.0;
    /// Accumulated map PointCloud2 publish rate (Hz); period = 1/rate.
    double map_cloud_publish_hz = 0.5;
    /// Max smoother-frame scans retained (oldest dropped when exceeded).
    int map_cloud_max_chunks = 3000;
};

// Map-representation knobs. The "map" is the per-submap GMM produced at
// submap finalization; pruning collapses near-duplicate components produced
// by overlapping observations of the same physical structure.
struct MapConfig {
    bool prune_enable = true;
    // If true, map pruning removes duplicate components across different
    // source GMM frames by keeping the older measurement.
    bool prune_frame_to_frame_enable = true;
    // Bhattacharyya distance threshold under which two components are merged.
    // For two equal isotropic gaussians 3 sigma apart D_B ~= 1.125, so a
    // value around 1.5 merges aggressively only on near-duplicates.
    double prune_bhatt_threshold = 1.5;
    // Legacy mean-distance prefilter (meters) when prune_use_rtree is false.
    // When prune_use_rtree is true, this value is added as a uniform margin
    // (meters) on each axis of every component AABB before indexing.
    double prune_search_radius_m = 0.5;
    // If true, use an R-tree on 3D AABBs (from covariance + chi-squared
    // ellipsoid bounds) to find merge candidates; otherwise use mean distance.
    bool prune_use_rtree = true;
    // Chi-squared threshold (3 dof) for the ellipsoid axis lengths used to
    // build each component's axis-aligned bounding box (larger = looser box).
    double prune_rtree_chi_sq = 9.21; // ~99% mass in 3D if Gaussian
    // Number of greedy passes over the component list. 1-2 is normally enough.
    int prune_max_passes = 2;
    // Tikhonov regularization added to covariances before any inversion to
    // keep Cholesky stable on near-singular components.
    double prune_cov_reg = 1e-6;
};

struct Config {
    bool debug_prints = true;
    RosConfig ros;
    PreprocessConfig preprocess;
    SogmmConfig sogmm;
    SmootherConfig smoother;
    RegistrationConfig registration;
    LoopClosureConfig loop_closure;
    SolidConfig solid;
    KeyframeConfig keyframe;
    GlobalGraphConfig global_graph;
    GtNoiseConfig gt_noise;
    ImuConfig imu;
    VisualizationConfig visualization;
    MapConfig map;
    std::string gmm_dir = "/tmp/gmmslam_gmms";
};

Config loadConfig(const std::string& yaml_path);

} // namespace gmmslam
