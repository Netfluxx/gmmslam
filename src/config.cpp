#include "gmmslam/config.hpp"
#include <yaml-cpp/yaml.h>

namespace gmmslam {

namespace {

template <typename T>
T readOr(const YAML::Node& node, const std::string& key, const T& default_val) {
    if (node[key]) return node[key].as<T>();
    return default_val;
}

void loadRos(const YAML::Node& root, RosConfig& c) {
    if (!root["ros"]) return;
    const auto& n = root["ros"];
    c.lidar_topic = readOr<std::string>(n, "lidar_topic", c.lidar_topic);
    c.gt_topic = readOr<std::string>(n, "gt_topic", c.gt_topic);
    c.imu_topic = readOr<std::string>(n, "imu_topic", c.imu_topic);
    c.sensor_frame = readOr<std::string>(n, "sensor_frame", c.sensor_frame);
    c.odom_frame = readOr<std::string>(n, "odom_frame", c.odom_frame);
    c.base_frame = readOr<std::string>(n, "base_frame", c.base_frame);
    c.odometry_input = readOr<std::string>(
        n, "odometry_input",
        readOr<std::string>(n, "noisy_gt_topic", c.odometry_input));
    c.restart_state_path = readOr<std::string>(
        n, "restart_state_path", c.restart_state_path);
    c.registration_request_topic = readOr<std::string>(n, "registration_request_topic", c.registration_request_topic);
    c.registration_result_topic = readOr<std::string>(n, "registration_result_topic", c.registration_result_topic);
}

void loadPreprocess(const YAML::Node& root, PreprocessConfig& c) {
    if (!root["preprocess"]) return;
    const auto& n = root["preprocess"];
    c.min_range = readOr(n, "min_range", c.min_range);
    c.max_range = readOr(n, "max_range", c.max_range);
    c.voxel_leaf_size = readOr(n, "voxel_leaf_size", c.voxel_leaf_size);
    c.target_points = readOr(n, "target_points", c.target_points);
    c.min_points = readOr(n, "min_points", c.min_points);
}

void loadSogmm(const YAML::Node& root, SogmmConfig& c) {
    if (!root["sogmm"]) return;
    const auto& n = root["sogmm"];
    c.bandwidth = readOr(n, "bandwidth", c.bandwidth);
    c.max_points = readOr(n, "max_points", c.max_points);
    c.n_components = readOr(n, "n_components", c.n_components);
    c.compute = readOr<std::string>(n, "compute", c.compute);
    c.backend = readOr<std::string>(n, "backend", c.backend);
}

void loadGmmap(const YAML::Node& root, SogmmConfig& c) {
    if (!root["gmmap"]) return;
    const auto& n = root["gmmap"];
    const auto& frame = n["frame"];
    const auto& map = n["map"];
    const auto& adapter = n["adapter"];

    if (frame) {
        c.gmmap_dataset =
            readOr<std::string>(frame, "dataset", c.gmmap_dataset);
        c.gmmap_occ_x_threshold =
            readOr(frame, "occ_x_t", c.gmmap_occ_x_threshold);
        c.gmmap_noise_threshold =
            readOr(frame, "noise_thresh", c.gmmap_noise_threshold);
        c.gmmap_sparse_threshold =
            readOr(frame, "sparse_t", c.gmmap_sparse_threshold);
        c.gmmap_ncheck_threshold =
            readOr(frame, "ncheck_t", c.gmmap_ncheck_threshold);
        c.gmmap_adaptive_threshold_scale =
            readOr(frame, "adaptive_thresh_scale",
                   c.gmmap_adaptive_threshold_scale);
        c.gmmap_line_threshold =
            readOr(frame, "line_t", c.gmmap_line_threshold);
        c.gmmap_angle_threshold =
            readOr(frame, "angle_t", c.gmmap_angle_threshold);
        c.gmmap_noise_floor =
            readOr(frame, "noise_floor", c.gmmap_noise_floor);
        c.gmmap_num_line_threshold =
            readOr(frame, "num_line_t", c.gmmap_num_line_threshold);
        c.gmmap_num_pixels_threshold =
            readOr(frame, "num_pixels_t", c.gmmap_num_pixels_threshold);
        c.gmmap_max_incomplete_clusters =
            readOr(frame, "max_incomplete_clusters",
                   c.gmmap_max_incomplete_clusters);
        c.gmmap_free_space_dist_scale =
            readOr(frame, "free_space_dist_scale",
                   c.gmmap_free_space_dist_scale);
        c.gmmap_debug_row_idx =
            readOr(frame, "debug_row_idx", c.gmmap_debug_row_idx);
    }

    if (map) {
        c.gmmap_num_threads =
            readOr(map, "num_threads", c.gmmap_num_threads);
        c.gmmap_measure_memory =
            readOr(map, "measure_memory", c.gmmap_measure_memory);
        c.gmmap_frame_alg_name =
            readOr<std::string>(map, "frame_alg_name",
                                c.gmmap_frame_alg_name);
        c.gmmap_max_depth =
            readOr(map, "max_depth", c.gmmap_max_depth);
        c.gmmap_hell_thresh_squared_oversized_gau =
            readOr(map, "hell_thresh_squard_oversized_gau",
                   c.gmmap_hell_thresh_squared_oversized_gau);
        c.gmmap_hell_thresh_squared_free =
            readOr(map, "hell_thresh_squard_free",
                   c.gmmap_hell_thresh_squared_free);
        c.gmmap_hell_thresh_squared_obs_scale =
            readOr(map, "hell_thresh_squard_obs_scale",
                   c.gmmap_hell_thresh_squared_obs_scale);
        c.gmmap_min_gaussian_length =
            readOr(map, "min_gau_len", c.gmmap_min_gaussian_length);
        c.gmmap_frame_max_scale =
            readOr(map, "frame_max_scale", c.gmmap_frame_max_scale);
        c.gmmap_min_num_neighbor_clusters =
            readOr(map, "min_num_neighbor_clusters",
                   c.gmmap_min_num_neighbor_clusters);
        c.gmmap_hell_thresh_squared_min =
            readOr(map, "hell_thresh_squard_min",
                   c.gmmap_hell_thresh_squared_min);
        c.gmmap_fusion_bound =
            readOr(map, "gau_fusion_bd", c.gmmap_fusion_bound);
        c.gmmap_rtree_bound_scale =
            readOr(map, "gau_rtree_bd", c.gmmap_rtree_bound_scale);
        c.gmmap_depth_scale =
            readOr(map, "depth_scale", c.gmmap_depth_scale);
        c.gmmap_track_color =
            readOr(map, "track_color", c.gmmap_track_color);
        c.gmmap_track_intensity =
            readOr(map, "track_intensity", c.gmmap_track_intensity);
        c.gmmap_cur_debug_frame =
            readOr(map, "cur_debug_frame", c.gmmap_cur_debug_frame);
    }

    if (adapter) {
        c.gmmap_far_fill_enable =
            readOr(adapter, "far_fill_enable", c.gmmap_far_fill_enable);
        c.gmmap_far_fill_start_m =
            readOr(adapter, "far_fill_start_m", c.gmmap_far_fill_start_m);
        c.gmmap_far_fill_voxel_m =
            readOr(adapter, "far_fill_voxel_m", c.gmmap_far_fill_voxel_m);
        c.gmmap_far_fill_max_components =
            readOr(adapter, "far_fill_max_components",
                   c.gmmap_far_fill_max_components);
        c.gmmap_far_fill_skip_max_depth_margin_m =
            readOr(adapter, "far_fill_skip_max_depth_margin_m",
                   c.gmmap_far_fill_skip_max_depth_margin_m);
        c.gmmap_estimate_intrinsics =
            readOr(adapter, "estimate_intrinsics",
                   c.gmmap_estimate_intrinsics);
        c.gmmap_horizontal_fov_deg =
            readOr(adapter, "horizontal_fov_deg",
                   c.gmmap_horizontal_fov_deg);
        c.gmmap_fx = readOr(adapter, "fx", c.gmmap_fx);
        c.gmmap_fy = readOr(adapter, "fy", c.gmmap_fy);
        c.gmmap_cx = readOr(adapter, "cx", c.gmmap_cx);
        c.gmmap_cy = readOr(adapter, "cy", c.gmmap_cy);
    }
}

void loadSmoother(const YAML::Node& root, SmootherConfig& c) {
    if (!root["smoother"]) return;
    const auto& n = root["smoother"];
    c.fixed_lag_s = readOr(n, "fixed_lag_s", c.fixed_lag_s);
    c.smoother_stride = readOr(n, "smoother_stride", c.smoother_stride);
    c.odom_noise_sigma_t = readOr(n, "odom_noise_sigma_t", c.odom_noise_sigma_t);
    c.odom_noise_sigma_r = readOr(n, "odom_noise_sigma_r", c.odom_noise_sigma_r);
    c.lost_scale = readOr(n, "lost_scale", c.lost_scale);
    c.prior_scale = readOr(n, "prior_scale", c.prior_scale);
    c.pose_history_keep = readOr(n, "pose_history_keep", c.pose_history_keep);
}

void loadRegistration(const YAML::Node& root, RegistrationConfig& c) {
    if (!root["registration"]) return;
    const auto& n = root["registration"];
    c.enable_async = readOr(n, "enable_async", c.enable_async);
    c.request_every_n_frames = readOr(n, "request_every_n_frames", c.request_every_n_frames);
    c.factor_every_n_frames = readOr(n, "factor_every_n_frames", c.factor_every_n_frames);
    c.score_threshold = readOr(n, "score_threshold", c.score_threshold);
    c.strong_factor_score_threshold = readOr(n, "strong_factor_score_threshold", c.strong_factor_score_threshold);
    c.strong_factor_sigma_t = readOr(n, "strong_factor_sigma_t", c.strong_factor_sigma_t);
    c.strong_factor_sigma_r = readOr(n, "strong_factor_sigma_r", c.strong_factor_sigma_r);
    c.queue_size = readOr(n, "queue_size", c.queue_size);
    c.result_queue_size = readOr(n, "result_queue_size", c.result_queue_size);
    c.enqueue_cooldown_frames = readOr(n, "enqueue_cooldown_frames", c.enqueue_cooldown_frames);
    c.workers = readOr(n, "workers", c.workers);
    c.compensate_fit_latency = readOr(n, "compensate_fit_latency", c.compensate_fit_latency);
    c.score_sigma_low = readOr(n, "score_sigma_low", c.score_sigma_low);
    c.score_sigma_high = readOr(n, "score_sigma_high", c.score_sigma_high);
    c.seq_sigma_t_min = readOr(n, "seq_sigma_t_min", c.seq_sigma_t_min);
    c.seq_sigma_t_max = readOr(n, "seq_sigma_t_max", c.seq_sigma_t_max);
    c.seq_sigma_r_min = readOr(n, "seq_sigma_r_min", c.seq_sigma_r_min);
    c.seq_sigma_r_max = readOr(n, "seq_sigma_r_max", c.seq_sigma_r_max);
    c.loop_sigma_t_min = readOr(n, "loop_sigma_t_min", c.loop_sigma_t_min);
    c.loop_sigma_t_max = readOr(n, "loop_sigma_t_max", c.loop_sigma_t_max);
    c.loop_sigma_r_min = readOr(n, "loop_sigma_r_min", c.loop_sigma_r_min);
    c.loop_sigma_r_max = readOr(n, "loop_sigma_r_max", c.loop_sigma_r_max);
}

void loadLoopClosure(const YAML::Node& root, LoopClosureConfig& c) {
    if (!root["loop_closure"]) return;
    const auto& n = root["loop_closure"];
    c.enable = readOr(n, "enable", c.enable);
    c.search_radius_m = readOr(n, "search_radius_m", c.search_radius_m);
    c.min_keyframe_gap = readOr(n, "min_keyframe_gap", c.min_keyframe_gap);
    c.max_candidates = readOr(n, "max_candidates", c.max_candidates);
    c.request_every_n_keyframes = readOr(n, "request_every_n_keyframes", c.request_every_n_keyframes);
    c.search_cooldown_keyframes = readOr(n, "search_cooldown_keyframes", c.search_cooldown_keyframes);
    c.min_separation_m = readOr(n, "min_separation_m", c.min_separation_m);
    c.min_separation_deg = readOr(n, "min_separation_deg", c.min_separation_deg);
    c.max_age_s = readOr(n, "max_age_s", c.max_age_s);
    c.gmm_keep_keyframes = readOr(n, "gmm_keep_keyframes", c.gmm_keep_keyframes);
    c.detect_score_threshold = readOr(n, "detect_score_threshold", c.detect_score_threshold);
    c.super_sigma_t = readOr(n, "super_sigma_t", c.super_sigma_t);
    c.super_sigma_r = readOr(n, "super_sigma_r", c.super_sigma_r);
    c.consistency_gate_trans_m =
        readOr(n, "consistency_gate_trans_m", c.consistency_gate_trans_m);
    c.consistency_gate_rot_deg =
        readOr(n, "consistency_gate_rot_deg", c.consistency_gate_rot_deg);
}

void loadSolid(const YAML::Node& root, SolidConfig& c) {
    if (!root["solid"]) return;
    const auto& n = root["solid"];
    c.enable = readOr(n, "enable", c.enable);
    c.fov_up_deg = readOr(n, "fov_up_deg", c.fov_up_deg);
    c.fov_down_deg = readOr(n, "fov_down_deg", c.fov_down_deg);
    c.min_distance_m = readOr(n, "min_distance_m", c.min_distance_m);
    c.max_distance_m = readOr(n, "max_distance_m", c.max_distance_m);
    c.num_angle = readOr(n, "num_angle", c.num_angle);
    c.num_range = readOr(n, "num_range", c.num_range);
    c.num_height = readOr(n, "num_height", c.num_height);
    c.voxel_size_m = readOr(n, "voxel_size_m", c.voxel_size_m);
    c.cos_similarity_threshold =
        readOr(n, "cos_similarity_threshold", c.cos_similarity_threshold);
    c.radius_weight = readOr(n, "radius_weight", c.radius_weight);
    c.appearance_weight = readOr(n, "appearance_weight", c.appearance_weight);
    c.provide_yaw_prior = readOr(n, "provide_yaw_prior", c.provide_yaw_prior);
    c.overlap_min = readOr(n, "overlap_min", c.overlap_min);
    c.max_abs_yaw_deg = readOr(n, "max_abs_yaw_deg", c.max_abs_yaw_deg);
    c.yaw_sign = readOr(n, "yaw_sign", c.yaw_sign);
    c.rescue_enable = readOr(n, "rescue_enable", c.rescue_enable);
    c.rescue_every_n_kf = readOr(n, "rescue_every_n_kf", c.rescue_every_n_kf);
    c.rescue_trigger_silence_kf =
        readOr(n, "rescue_trigger_silence_kf", c.rescue_trigger_silence_kf);
    c.rescue_top_k = readOr(n, "rescue_top_k", c.rescue_top_k);
    c.rescue_cos_threshold =
        readOr(n, "rescue_cos_threshold", c.rescue_cos_threshold);
    c.keep_descriptors = readOr(n, "keep_descriptors", c.keep_descriptors);
}

void loadKeyframe(const YAML::Node& root, KeyframeConfig& c) {
    if (!root["keyframe"]) return;
    const auto& n = root["keyframe"];
    c.translation_thresh_m = readOr(n, "translation_thresh_m", c.translation_thresh_m);
    c.rotation_thresh_deg = readOr(n, "rotation_thresh_deg", c.rotation_thresh_deg);
    c.use_time_trigger = readOr(n, "use_time_trigger", c.use_time_trigger);
    c.max_interval_s = readOr(n, "max_interval_s", c.max_interval_s);
}

void loadGlobalGraph(const YAML::Node& root, GlobalGraphConfig& c) {
    if (!root["global_graph"]) return;
    const auto& n = root["global_graph"];
    c.enable = readOr(n, "enable", c.enable);
    c.submap_keyframes_per_submap = readOr(n, "submap_keyframes_per_submap", c.submap_keyframes_per_submap);
    c.between_sigma_t = readOr(n, "between_sigma_t", c.between_sigma_t);
    c.between_sigma_r = readOr(n, "between_sigma_r", c.between_sigma_r);
    c.prior_sigma_t = readOr(n, "prior_sigma_t", c.prior_sigma_t);
    c.prior_sigma_r = readOr(n, "prior_sigma_r", c.prior_sigma_r);
    c.overlap_radius_m = readOr(n, "overlap_radius_m", c.overlap_radius_m);
    c.max_overlap_registrations =
        readOr(n, "max_overlap_registrations", c.max_overlap_registrations);
    c.reg_score_threshold = readOr(n, "reg_score_threshold", c.reg_score_threshold);
    c.min_loop_submap_gap = readOr(n, "min_loop_submap_gap", c.min_loop_submap_gap);
    c.enable_traj_aux_factors = readOr(n, "enable_traj_aux_factors", c.enable_traj_aux_factors);
    c.traj_sigma_t = readOr(n, "traj_sigma_t", c.traj_sigma_t);
    c.traj_sigma_r = readOr(n, "traj_sigma_r", c.traj_sigma_r);
    c.aux_gate_abs_trans_m = readOr(n, "aux_gate_abs_trans_m", c.aux_gate_abs_trans_m);
    c.aux_gate_abs_rot_deg = readOr(n, "aux_gate_abs_rot_deg", c.aux_gate_abs_rot_deg);
    c.aux_gate_consistency_trans_m = readOr(n, "aux_gate_consistency_trans_m", c.aux_gate_consistency_trans_m);
    c.aux_gate_consistency_rot_deg = readOr(n, "aux_gate_consistency_rot_deg", c.aux_gate_consistency_rot_deg);
    c.traj_aux_gate_abs_trans_m =
        readOr(n, "traj_aux_gate_abs_trans_m", c.aux_gate_abs_trans_m);
    c.traj_aux_gate_abs_rot_deg =
        readOr(n, "traj_aux_gate_abs_rot_deg", c.aux_gate_abs_rot_deg);
    c.traj_aux_gate_consistency_trans_m =
        readOr(n, "traj_aux_gate_consistency_trans_m", c.aux_gate_consistency_trans_m);
    c.traj_aux_gate_consistency_rot_deg =
        readOr(n, "traj_aux_gate_consistency_rot_deg", c.aux_gate_consistency_rot_deg);
    c.submap_loop_sigma_t_min = readOr(n, "submap_loop_sigma_t_min", c.submap_loop_sigma_t_min);
    c.submap_loop_sigma_t_max = readOr(n, "submap_loop_sigma_t_max", c.submap_loop_sigma_t_max);
    c.submap_loop_sigma_r_min = readOr(n, "submap_loop_sigma_r_min", c.submap_loop_sigma_r_min);
    c.submap_loop_sigma_r_max = readOr(n, "submap_loop_sigma_r_max", c.submap_loop_sigma_r_max);
    c.reanchor_smoother_on_traj_gate_fail =
        readOr(n, "reanchor_smoother_on_traj_gate_fail", c.reanchor_smoother_on_traj_gate_fail);
    c.submap_finalize_min_ready_keyframes =
        readOr(n, "submap_finalize_min_ready_keyframes", c.submap_finalize_min_ready_keyframes);
    c.submap_finalize_min_ready_fraction =
        readOr(n, "submap_finalize_min_ready_fraction", c.submap_finalize_min_ready_fraction);
    c.submap_finalize_max_wait_s =
        readOr(n, "submap_finalize_max_wait_s", c.submap_finalize_max_wait_s);
}

void loadGtNoise(const YAML::Node& root, GtNoiseConfig& c) {
    if (!root["gt_noise"]) return;
    const auto& n = root["gt_noise"];
    c.init_wait_s = readOr(n, "init_wait_s", c.init_wait_s);
    c.sigma_t = readOr(n, "sigma_t", c.sigma_t);
    c.sigma_r = readOr(n, "sigma_r", c.sigma_r);
    c.factor_sigma_t = readOr(n, "factor_sigma_t", c.factor_sigma_t);
    c.factor_sigma_r = readOr(n, "factor_sigma_r", c.factor_sigma_r);
    c.seed = readOr(n, "seed", c.seed);
    c.jump_reject_trans_m = readOr(n, "jump_reject_trans_m", c.jump_reject_trans_m);
    c.jump_reject_rot_deg = readOr(n, "jump_reject_rot_deg", c.jump_reject_rot_deg);
}

void loadImu(const YAML::Node& root, ImuConfig& c) {
    if (!root["imu"]) return;
    const auto& n = root["imu"];
    c.enable_preintegration = readOr(n, "enable_preintegration", c.enable_preintegration);
    c.gravity_mps2 = readOr(n, "gravity_mps2", c.gravity_mps2);
    c.buffer_keep_s = readOr(n, "buffer_keep_s", c.buffer_keep_s);
    c.acc_noise_sigma = readOr(n, "acc_noise_sigma", c.acc_noise_sigma);
    c.gyro_noise_sigma = readOr(n, "gyro_noise_sigma", c.gyro_noise_sigma);
    c.integration_sigma = readOr(n, "integration_sigma", c.integration_sigma);
    c.bias_acc_rw_sigma = readOr(n, "bias_acc_rw_sigma", c.bias_acc_rw_sigma);
    c.bias_gyro_rw_sigma = readOr(n, "bias_gyro_rw_sigma", c.bias_gyro_rw_sigma);
    c.velocity_prior_sigma = readOr(n, "velocity_prior_sigma", c.velocity_prior_sigma);
    c.bias_prior_sigma = readOr(n, "bias_prior_sigma", c.bias_prior_sigma);
}

void loadVisualization(const YAML::Node& root, VisualizationConfig& c) {
    if (!root["visualization"]) return;
    const auto& n = root["visualization"];
    c.gmm_marker_sigma = readOr(n, "gmm_marker_sigma", c.gmm_marker_sigma);
    c.global_gmm_markers_enable =
        readOr(n, "global_gmm_markers_enable", c.global_gmm_markers_enable);
    c.global_gmm_publish_period_s = readOr(n, "global_gmm_publish_period_s", c.global_gmm_publish_period_s);
    c.d2d_frame_to_frame_text_enable =
        readOr(n, "d2d_frame_to_frame_text_enable",
               c.d2d_frame_to_frame_text_enable);
    c.d2d_submap_overlap_text_enable =
        readOr(n, "d2d_submap_overlap_text_enable",
               c.d2d_submap_overlap_text_enable);
    c.d2d_loop_closure_text_enable =
        readOr(n, "d2d_loop_closure_text_enable",
               c.d2d_loop_closure_text_enable);
    c.output_pose_lpf_cutoff_hz = readOr(n, "output_pose_lpf_cutoff_hz", c.output_pose_lpf_cutoff_hz);
    c.map_cloud_publish_hz = readOr(n, "map_cloud_publish_hz", c.map_cloud_publish_hz);
    c.map_cloud_max_chunks = readOr(n, "map_cloud_max_chunks", c.map_cloud_max_chunks);
}

void loadMap(const YAML::Node& root, MapConfig& c) {
    if (!root["map"]) return;
    const auto& n = root["map"];
    c.prune_enable = readOr(n, "prune_enable", c.prune_enable);
    c.prune_frame_to_frame_enable =
        readOr(n, "prune_frame_to_frame_enable",
               c.prune_frame_to_frame_enable);
    c.prune_bhatt_threshold = readOr(n, "prune_bhatt_threshold", c.prune_bhatt_threshold);
    c.prune_search_radius_m = readOr(n, "prune_search_radius_m", c.prune_search_radius_m);
    c.prune_use_rtree = readOr(n, "prune_use_rtree", c.prune_use_rtree);
    c.prune_rtree_chi_sq = readOr(n, "prune_rtree_chi_sq", c.prune_rtree_chi_sq);
    c.prune_max_passes = readOr(n, "prune_max_passes", c.prune_max_passes);
    c.prune_cov_reg = readOr(n, "prune_cov_reg", c.prune_cov_reg);
}

} // anonymous namespace

Config loadConfig(const std::string& yaml_path) {
    Config cfg;
    YAML::Node root = YAML::LoadFile(yaml_path);

    loadRos(root, cfg.ros);
    loadPreprocess(root, cfg.preprocess);
    loadSogmm(root, cfg.sogmm);
    loadGmmap(root, cfg.sogmm);
    loadSmoother(root, cfg.smoother);
    loadRegistration(root, cfg.registration);
    loadLoopClosure(root, cfg.loop_closure);
    loadSolid(root, cfg.solid);
    loadKeyframe(root, cfg.keyframe);
    loadGlobalGraph(root, cfg.global_graph);
    loadGtNoise(root, cfg.gt_noise);
    loadImu(root, cfg.imu);
    loadVisualization(root, cfg.visualization);
    loadMap(root, cfg.map);

    cfg.debug_prints = readOr(root, "DEBUG_PRINTS", cfg.debug_prints);
    cfg.gmm_dir = readOr<std::string>(root, "gmm_dir", cfg.gmm_dir);

    return cfg;
}

} // namespace gmmslam
