"""GMM-SLAM ROS1 node — slimmed orchestrator.

Reads parameters, wires up FixedLagBackend / GlobalPoseGraph /
RegistrationManager / Visualizer, and runs the main point-cloud callback
which feeds the smoother on **every** frame while keyframes only trigger
global graph updates and async registration.
"""

import threading

import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Image, CameraInfo, Imu
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation

from imports_setup import HAS_SOGMM
from ros_helpers import (
    pc2_to_numpy,
    pose_to_pose_stamped,
    preprocess,
    pose_msg_to_matrix,
    stamp_to_sec,
)
from smoother import FixedLagBackend
from global_graph import GlobalPoseGraph
from registration import RegistrationManager
from visualization import Visualizer

if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm


class GMMSLAMNode:

    def __init__(self):
        rospy.init_node("gmmslam_node", anonymous=False)

        def _pa(new_name: str, old_name: str, default):
            if rospy.has_param(new_name):
                return rospy.get_param(new_name)
            if rospy.has_param(old_name):
                rospy.logwarn_once(
                    f"[gmmslam] param '{old_name}' deprecated, use '{new_name}'"
                )
                return rospy.get_param(old_name)
            return default

        # ==============================================================
        # Parameters
        # ==============================================================
        self.lidar_topic = rospy.get_param("~lidar_topic", "/m500_1/mpa/depth/points")
        self.imu_topic = rospy.get_param("~imu_topic", "/m500_1/mavros/imu/data")
        self._imu_buffer = []
        self._latest_imu = None
        self._last_cloud_stamp_for_imu = None
        self.imu_buffer_keep_s = float(rospy.get_param("~imu_buffer_keep_s", 30.0))
        self.sensor_frame = rospy.get_param("~sensor_frame", "depth_camera_1")
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "m500_1_base_link")

        self.min_range = rospy.get_param("~min_range", 0.1)
        self.max_range = rospy.get_param("~max_range", 10.0)
        self.voxel_size = rospy.get_param("~voxel_leaf_size", 0.05)
        self.min_points = rospy.get_param("~min_points", 50)
        self.global_map_publish_period_s = max(
            0.1, float(rospy.get_param("~global_map_publish_period_s", 1.0))
        )
        self.global_gmm_publish_period_s = max(
            0.1, float(rospy.get_param("~global_gmm_publish_period_s", 1.0))
        )

        # SOGMM
        self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.1)
        self.sogmm_compute = rospy.get_param("~sogmm_compute", "GPU")
        self.sogmm_max_points = rospy.get_param("~sogmm_max_points", 2000)
        self.sogmm_n_components = rospy.get_param("~sogmm_n_components", 0)
        self.gmm_marker_sigma = float(rospy.get_param("~gmm_marker_sigma", 3.0))
        self.plot_first_frame = rospy.get_param("~plot_first_frame", False)

        # Backend
        self.fixed_lag_s = rospy.get_param("~fixed_lag_s", 10.0)
        self.odom_noise_sigma_t = rospy.get_param("~odom_noise_sigma_t", 0.06)
        self.odom_noise_sigma_r = rospy.get_param("~odom_noise_sigma_r", 0.06)
        self.enable_imu_preintegration = bool(
            rospy.get_param("~enable_imu_preintegration", True)
        )
        self.imu_gravity_mps2 = float(rospy.get_param("~imu_gravity_mps2", 9.81))
        self.imu_acc_noise_sigma = float(rospy.get_param("~imu_acc_noise_sigma", 0.2))
        self.imu_gyro_noise_sigma = float(
            rospy.get_param("~imu_gyro_noise_sigma", 0.05)
        )
        self.imu_integration_sigma = float(
            rospy.get_param("~imu_integration_sigma", 1e-4)
        )
        self.imu_bias_acc_rw_sigma = float(
            rospy.get_param("~imu_bias_acc_rw_sigma", 1e-3)
        )
        self.imu_bias_gyro_rw_sigma = float(
            rospy.get_param("~imu_bias_gyro_rw_sigma", 1e-4)
        )
        self.imu_velocity_prior_sigma = float(
            rospy.get_param("~imu_velocity_prior_sigma", 1.0)
        )
        self.imu_bias_prior_sigma = float(rospy.get_param("~imu_bias_prior_sigma", 0.1))

        # Keyframe thresholds (for global graph / registration only)
        self.keyframe_translation_thresh_m = float(
            rospy.get_param("~keyframe_translation_thresh_m", 0.3)
        )
        self.keyframe_rotation_thresh_deg = float(
            rospy.get_param("~keyframe_rotation_thresh_deg", 5.0)
        )
        self.keyframe_use_time_trigger = bool(
            rospy.get_param("~keyframe_use_time_trigger", False)
        )
        self.keyframe_max_interval_s = float(
            rospy.get_param("~keyframe_max_interval_s", 10.0)
        )

        # Global graph
        self.enable_global_pose_graph = bool(
            rospy.get_param("~enable_global_pose_graph", True)
        )
        self.submap_keyframes_per_submap = max(
            1, int(rospy.get_param("~submap_keyframes_per_submap", 15))
        )
        self.submap_between_sigma_t = float(
            rospy.get_param("~submap_between_sigma_t", 0.08)
        )
        self.submap_between_sigma_r = float(
            rospy.get_param("~submap_between_sigma_r", 0.08)
        )
        self.submap_prior_sigma_t = float(
            rospy.get_param("~submap_prior_sigma_t", 0.02)
        )
        self.submap_prior_sigma_r = float(
            rospy.get_param("~submap_prior_sigma_r", 0.02)
        )
        self.submap_traj_sigma_t = float(rospy.get_param("~submap_traj_sigma_t", 0.15))
        self.submap_traj_sigma_r = float(rospy.get_param("~submap_traj_sigma_r", 0.15))
        self.submap_aux_gate_abs_trans_m = float(
            rospy.get_param("~submap_aux_gate_abs_trans_m", 10.0)
        )
        self.submap_aux_gate_abs_rot_deg = float(
            rospy.get_param("~submap_aux_gate_abs_rot_deg", 90.0)
        )
        self.submap_aux_gate_consistency_trans_m = float(
            rospy.get_param("~submap_aux_gate_consistency_trans_m", 2.0)
        )
        self.submap_aux_gate_consistency_rot_deg = float(
            rospy.get_param("~submap_aux_gate_consistency_rot_deg", 35.0)
        )

        # Registration / loop closure
        self.enable_async_registration = rospy.get_param(
            "~enable_async_registration", True
        )
        self.compensate_fit_latency_in_map = bool(
            rospy.get_param("~compensate_fit_latency_in_map", True)
        )
        self.registration_request_every_n_frames = max(
            1,
            int(
                _pa(
                    "~registration_request_every_n_frames",
                    "~registration_request_stride",
                    _pa(
                        "~registration_request_stride",
                        "~registration_fit_stride",
                        2,
                    ),
                )
            ),
        )
        self.registration_score_threshold = float(
            rospy.get_param("~registration_score_threshold", -1e9)
        )
        self.loop_closure_score_threshold = float(
            _pa(
                "~strong_factor_score_threshold",
                "~loop_closure_score_threshold",
                2.0,
            )
        )
        self.loop_closure_sigma_t = float(
            _pa("~strong_factor_sigma_t", "~loop_closure_sigma_t", 0.01)
        )
        self.loop_closure_sigma_r = float(
            _pa("~strong_factor_sigma_r", "~loop_closure_sigma_r", 0.01)
        )
        self.registration_request_topic = rospy.get_param(
            "~registration_request_topic", "/gmmslam_node/registration/request"
        )
        self.registration_result_topic = rospy.get_param(
            "~registration_result_topic", "/gmmslam_node/registration/result"
        )
        self.registration_queue_size = max(
            1, int(rospy.get_param("~registration_queue_size", 8))
        )
        self.registration_result_queue_size = max(
            1, int(rospy.get_param("~registration_result_queue_size", 64))
        )
        self.registration_enqueue_cooldown_frames = max(
            0, int(rospy.get_param("~registration_enqueue_cooldown_frames", 8))
        )
        self.enable_loop_closure_detection = bool(
            rospy.get_param("~enable_loop_closure_detection", True)
        )
        self.loop_closure_search_radius_m = float(
            rospy.get_param("~loop_closure_search_radius_m", 2.0)
        )
        self.loop_closure_min_keyframe_gap = max(
            10, int(rospy.get_param("~loop_closure_min_keyframe_gap", 10))
        )
        self.loop_closure_max_candidates = max(
            1, int(rospy.get_param("~loop_closure_max_candidates", 100))
        )
        self.loop_closure_request_every_n_keyframes = max(
            1,
            int(rospy.get_param("~loop_closure_request_every_n_keyframes", 1)),
        )
        self.loop_closure_search_cooldown_keyframes = max(
            1,
            int(rospy.get_param("~loop_closure_search_cooldown_keyframes", 5)),
        )
        self.loop_closure_min_separation_m = float(
            rospy.get_param("~loop_closure_min_separation_m", 0.8)
        )
        self.loop_closure_min_separation_deg = float(
            rospy.get_param("~loop_closure_min_separation_deg", 20.0)
        )
        self.loop_closure_max_age_s = float(
            rospy.get_param("~loop_closure_max_age_s", 520.0)
        )
        self.loop_closure_gmm_keep_keyframes = max(
            200, int(rospy.get_param("~loop_closure_gmm_keep_keyframes", 2000))
        )
        self.pose_history_keep_keyframes = max(
            600, int(rospy.get_param("~pose_history_keep_keyframes", 5000))
        )
        self.loop_closure_detect_score_threshold = float(
            rospy.get_param("~loop_closure_detect_score_threshold", 0.8)
        )
        self.loop_closure_super_sigma_t = float(
            rospy.get_param("~loop_closure_super_sigma_t", 0.0005)
        )
        self.loop_closure_super_sigma_r = float(
            rospy.get_param("~loop_closure_super_sigma_r", 0.0005)
        )
        self.score_sigma_low = float(
            rospy.get_param(
                "~score_sigma_low", self.loop_closure_detect_score_threshold
            )
        )
        self.score_sigma_high = float(
            rospy.get_param("~score_sigma_high", self.loop_closure_score_threshold)
        )
        self.seq_sigma_t_min = float(rospy.get_param("~seq_sigma_t_min", 0.02))
        self.seq_sigma_t_max = float(rospy.get_param("~seq_sigma_t_max", 0.20))
        self.seq_sigma_r_min = float(rospy.get_param("~seq_sigma_r_min", 0.01))
        self.seq_sigma_r_max = float(rospy.get_param("~seq_sigma_r_max", 0.15))
        self.loop_sigma_t_min = float(rospy.get_param("~loop_sigma_t_min", 0.03))
        self.loop_sigma_t_max = float(rospy.get_param("~loop_sigma_t_max", 0.40))
        self.loop_sigma_r_min = float(rospy.get_param("~loop_sigma_r_min", 0.02))
        self.loop_sigma_r_max = float(rospy.get_param("~loop_sigma_r_max", 0.25))
        self.submap_loop_sigma_t_min = float(
            rospy.get_param("~submap_loop_sigma_t_min", 0.05)
        )
        self.submap_loop_sigma_t_max = float(
            rospy.get_param("~submap_loop_sigma_t_max", 0.50)
        )
        self.submap_loop_sigma_r_min = float(
            rospy.get_param("~submap_loop_sigma_r_min", 0.03)
        )
        self.submap_loop_sigma_r_max = float(
            rospy.get_param("~submap_loop_sigma_r_max", 0.30)
        )
        self.registration_factor_every_n_frames = max(
            1,
            int(
                _pa(
                    "~registration_factor_every_n_frames",
                    "~registration_factor_stride",
                    _pa(
                        "~registration_factor_stride",
                        "~registration_apply_stride",
                        1,
                    ),
                )
            ),
        )

        # GT factors
        self.gt_init_wait_s = float(rospy.get_param("~gt_init_wait_s", 3.0))
        self.use_noisy_gt_factor = rospy.get_param("~use_noisy_gt_factor", True)
        self.gt_noise_sigma_t = rospy.get_param("~gt_noise_sigma_t", 0.00)
        self.gt_noise_sigma_r = rospy.get_param("~gt_noise_sigma_r", 0.00)
        self.gt_factor_sigma_t = rospy.get_param("~gt_factor_sigma_t", 0.0001)
        self.gt_factor_sigma_r = rospy.get_param("~gt_factor_sigma_r", 0.0001)
        self.gt_noise_seed = rospy.get_param("~gt_noise_seed", -1)
        self.noisy_gt_topic = rospy.get_param(
            "~noisy_gt_topic", "/gmmslam_node/noisy_gt_pose"
        )
        self.map_decimate = 5

        rospy.loginfo(f"[gmmslam] lidar_topic  : {self.lidar_topic}")
        rospy.loginfo(f"[gmmslam] odom_frame   : {self.odom_frame}")
        rospy.loginfo(f"[gmmslam] fixed_lag_s  : {self.fixed_lag_s}")
        rospy.loginfo(f"[gmmslam] async reg    : {self.enable_async_registration}")
        rospy.loginfo(
            f"[gmmslam] keyframe (global graph / reg only): "
            f"{self.keyframe_translation_thresh_m:.3f} m | "
            f"{self.keyframe_rotation_thresh_deg:.2f} deg"
        )

        # ==============================================================
        # State
        # ==============================================================
        self._frame_count = 0
        self._odom_idx = 0
        self._keyframe_count = 0
        self._keyframe_odom_indices: list = []
        self._last_keyframe_pose = None
        self._last_keyframe_t_sec = None
        self._first_cloud_seen = False
        self._reg_enqueue_resume_frame = 0
        self._last_backpressure_log_t = 0.0
        self._last_cloud_t_sec_processed = None

        # GT state
        self._latest_gt_pose_raw = None
        self._latest_gt_pose = None
        self._latest_noisy_gt_pose_msg = None
        self._gt_origin_inv = None
        self._gt_init_start_time = None
        self._last_gt_T_for_factor = None
        self._gt_path = Path()
        self._gt_path.header.frame_id = self.odom_frame
        self._rng = (
            np.random.default_rng(self.gt_noise_seed)
            if self.gt_noise_seed >= 0
            else np.random.default_rng()
        )

        # ==============================================================
        # Subsystem creation
        # ==============================================================
        self.smoother = FixedLagBackend(
            lag_s=self.fixed_lag_s,
            odom_sigma_t=self.odom_noise_sigma_t,
            odom_sigma_r=self.odom_noise_sigma_r,
            gt_factor_sigma_t=self.gt_factor_sigma_t,
            gt_factor_sigma_r=self.gt_factor_sigma_r,
            loop_sigma_t=self.loop_closure_sigma_t,
            loop_sigma_r=self.loop_closure_sigma_r,
            loop_super_sigma_t=self.loop_closure_super_sigma_t,
            loop_super_sigma_r=self.loop_closure_super_sigma_r,
            pose_history_keep=self.pose_history_keep_keyframes,
            enable_imu_preintegration=self.enable_imu_preintegration,
            imu_gravity_mps2=self.imu_gravity_mps2,
            imu_acc_noise_sigma=self.imu_acc_noise_sigma,
            imu_gyro_noise_sigma=self.imu_gyro_noise_sigma,
            imu_integration_sigma=self.imu_integration_sigma,
            imu_bias_acc_rw_sigma=self.imu_bias_acc_rw_sigma,
            imu_bias_gyro_rw_sigma=self.imu_bias_gyro_rw_sigma,
            imu_velocity_prior_sigma=self.imu_velocity_prior_sigma,
            imu_bias_prior_sigma=self.imu_bias_prior_sigma,
        )

        # TF broadcaster + publishers
        tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.gt_topic = rospy.get_param(
            "~gt_topic", "/m500_1/mavros/local_position/pose"
        )
        path_pub = rospy.Publisher("~path", Path, queue_size=1)
        global_graph_path_pub = rospy.Publisher(
            "~global_graph_path", Path, queue_size=1
        )
        odom_pub = rospy.Publisher("~odom", Odometry, queue_size=1)
        cloud_pub = rospy.Publisher("~map_cloud", PointCloud2, queue_size=1)
        latest_frame_cloud_pub = rospy.Publisher(
            "~latest_frame_cloud", PointCloud2, queue_size=1
        )
        self.gt_path_pub = rospy.Publisher("~gt_path", Path, queue_size=1)
        self.gt_pose_pub = rospy.Publisher("~gt_pose", PoseStamped, queue_size=1)
        gmm_markers_pub = rospy.Publisher("~gmm_markers", MarkerArray, queue_size=1)
        gmm_global_markers_pub = rospy.Publisher(
            "~gmm_global_markers", MarkerArray, queue_size=1, latch=True
        )
        global_graph_markers_pub = rospy.Publisher(
            "~global_graph_markers", MarkerArray, queue_size=1, latch=True
        )
        graph_nodes_pub = rospy.Publisher(
            "~graph_nodes", MarkerArray, queue_size=1, latch=True
        )
        reg_request_pub = rospy.Publisher(
            self.registration_request_topic, String, queue_size=10
        )

        self.global_graph = GlobalPoseGraph(
            odom_frame=self.odom_frame,
            submap_between_sigma_t=self.submap_between_sigma_t,
            submap_between_sigma_r=self.submap_between_sigma_r,
            submap_prior_sigma_t=self.submap_prior_sigma_t,
            submap_prior_sigma_r=self.submap_prior_sigma_r,
            loop_super_sigma_t=self.loop_closure_super_sigma_t,
            loop_super_sigma_r=self.loop_closure_super_sigma_r,
            submap_keyframes_per_submap=self.submap_keyframes_per_submap,
            enable=self.enable_global_pose_graph,
            path_pub=global_graph_path_pub,
            get_pose_fn=lambda idx: self.smoother.pose_by_idx.get(idx),
            get_gmm_fn=lambda idx: self._get_keyframe_gmm(idx),
            get_submap_traj_delta_fn=self._submap_traj_delta_between,
            reg_request_pub=reg_request_pub,
            gmm_dir=rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms"),
            overlap_radius_m=float(rospy.get_param("~submap_overlap_radius_m", 5.0)),
            submap_reg_score_threshold=float(
                rospy.get_param("~submap_reg_score_threshold", 0.5)
            ),
            submap_traj_sigma_t=self.submap_traj_sigma_t,
            submap_traj_sigma_r=self.submap_traj_sigma_r,
            submap_aux_gate_abs_trans_m=self.submap_aux_gate_abs_trans_m,
            submap_aux_gate_abs_rot_deg=self.submap_aux_gate_abs_rot_deg,
            submap_aux_gate_consistency_trans_m=self.submap_aux_gate_consistency_trans_m,
            submap_aux_gate_consistency_rot_deg=self.submap_aux_gate_consistency_rot_deg,
            score_sigma_low=self.score_sigma_low,
            score_sigma_high=self.score_sigma_high,
            loop_sigma_t_min=self.loop_sigma_t_min,
            loop_sigma_t_max=self.loop_sigma_t_max,
            loop_sigma_r_min=self.loop_sigma_r_min,
            loop_sigma_r_max=self.loop_sigma_r_max,
            submap_loop_sigma_t_min=self.submap_loop_sigma_t_min,
            submap_loop_sigma_t_max=self.submap_loop_sigma_t_max,
            submap_loop_sigma_r_min=self.submap_loop_sigma_r_min,
            submap_loop_sigma_r_max=self.submap_loop_sigma_r_max,
        )

        self.registration = RegistrationManager(
            smoother=self.smoother,
            global_graph=self.global_graph,
            reg_request_pub=reg_request_pub,
            sogmm_bandwidth=self.sogmm_bandwidth,
            sogmm_compute=self.sogmm_compute,
            sogmm_max_points=self.sogmm_max_points,
            sogmm_n_components=self.sogmm_n_components,
            gmm_dir=rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms"),
            plot_first_frame=self.plot_first_frame,
            compensate_fit_latency_in_map=self.compensate_fit_latency_in_map,
            registration_score_threshold=self.registration_score_threshold,
            registration_factor_every_n_frames=self.registration_factor_every_n_frames,
            loop_closure_score_threshold=self.loop_closure_score_threshold,
            loop_closure_detect_score_threshold=self.loop_closure_detect_score_threshold,
            enable_loop_closure_detection=self.enable_loop_closure_detection,
            loop_closure_min_keyframe_gap=self.loop_closure_min_keyframe_gap,
            loop_closure_max_candidates=self.loop_closure_max_candidates,
            loop_closure_search_radius_m=self.loop_closure_search_radius_m,
            loop_closure_search_cooldown_keyframes=self.loop_closure_search_cooldown_keyframes,
            loop_closure_request_every_n_keyframes=self.loop_closure_request_every_n_keyframes,
            loop_closure_min_separation_m=self.loop_closure_min_separation_m,
            loop_closure_min_separation_deg=self.loop_closure_min_separation_deg,
            loop_closure_max_age_s=self.loop_closure_max_age_s,
            loop_closure_gmm_keep_keyframes=self.loop_closure_gmm_keep_keyframes,
            registration_queue_size=self.registration_queue_size,
            registration_result_queue_size=self.registration_result_queue_size,
            fit_pool_workers=max(1, int(rospy.get_param("~fit_pool_workers", 2))),
            map_decimate=self.map_decimate,
            score_sigma_low=self.score_sigma_low,
            score_sigma_high=self.score_sigma_high,
            seq_sigma_t_min=self.seq_sigma_t_min,
            seq_sigma_t_max=self.seq_sigma_t_max,
            seq_sigma_r_min=self.seq_sigma_r_min,
            seq_sigma_r_max=self.seq_sigma_r_max,
            loop_sigma_t_min=self.loop_sigma_t_min,
            loop_sigma_t_max=self.loop_sigma_t_max,
            loop_sigma_r_min=self.loop_sigma_r_min,
            loop_sigma_r_max=self.loop_sigma_r_max,
        )

        self.visualizer = Visualizer(
            smoother=self.smoother,
            registration=self.registration,
            global_graph=self.global_graph,
            odom_frame=self.odom_frame,
            base_frame=self.base_frame,
            gmm_marker_sigma=self.gmm_marker_sigma,
            map_decimate=self.map_decimate,
            global_map_publish_period_s=self.global_map_publish_period_s,
            global_gmm_publish_period_s=self.global_gmm_publish_period_s,
            publishers={
                "path": path_pub,
                "odom": odom_pub,
                "cloud": cloud_pub,
                "latest_frame_cloud": latest_frame_cloud_pub,
                "gmm_markers": gmm_markers_pub,
                "gmm_global_markers": gmm_global_markers_pub,
                "global_graph_markers": global_graph_markers_pub,
                "graph_nodes": graph_nodes_pub,
                "tf_broadcaster": tf_broadcaster,
            },
        )

        # ==============================================================
        # Subscribers
        # ==============================================================
        rospy.Subscriber(
            self.lidar_topic, PointCloud2, self._pcl_callback, queue_size=1
        )
        rospy.Subscriber(self.gt_topic, PoseStamped, self._gt_callback, queue_size=1)
        rospy.Subscriber(
            self.registration_result_topic,
            String,
            self.registration.result_callback,
            queue_size=50,
        )
        rospy.Subscriber(
            self.noisy_gt_topic,
            PoseStamped,
            self._noisy_gt_input_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.imu_topic, Imu, self._imu_callback, queue_size=1)
        depth_ns = "/".join(self.lidar_topic.split("/")[:-1])
        rospy.Subscriber(
            depth_ns + "/image_raw", Image, self._depth_callback, queue_size=1
        )
        rospy.Subscriber(
            depth_ns + "/camera_info",
            CameraInfo,
            self._cam_info_callback,
            queue_size=1,
        )

        # ==============================================================
        # Worker threads
        # ==============================================================
        if self.enable_async_registration:
            threading.Thread(
                target=self.registration.fit_worker_loop, daemon=True
            ).start()
        threading.Thread(target=self.visualizer.vis_loop, daemon=True).start()
        threading.Thread(target=self.smoother.backend_loop, daemon=True).start()

        # Wait for /clock if sim time
        if rospy.get_param("/use_sim_time", False):
            rospy.loginfo("[gmmslam] use_sim_time=true, waiting for /clock …")
            while not rospy.is_shutdown() and rospy.Time.now().is_zero():
                rospy.sleep(0.1)
            rospy.loginfo("[gmmslam] clock started")
        self._gt_init_start_time = rospy.Time.now()
        rospy.loginfo("[gmmslam] node ready, waiting for point clouds …")

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _gt_callback(self, msg: PoseStamped):
        self._latest_gt_pose_raw = msg
        if not self._ensure_gt_origin_initialized(msg.header.stamp):
            return
        T_gt = pose_msg_to_matrix(msg.pose)
        T_odom = self._gt_origin_inv @ T_gt
        ps = pose_to_pose_stamped(T_odom, msg.header.stamp, self.odom_frame)
        self._latest_gt_pose = ps
        self._gt_path.header.stamp = ps.header.stamp
        self._gt_path.poses.append(ps)
        self.gt_pose_pub.publish(ps)
        self.gt_path_pub.publish(self._gt_path)

    def _noisy_gt_input_callback(self, msg: PoseStamped):
        self._latest_noisy_gt_pose_msg = msg

    def _ensure_gt_origin_initialized(self, stamp) -> bool:
        if self._gt_origin_inv is not None:
            return True
        if self._gt_init_start_time is None:
            self._gt_init_start_time = rospy.Time.now()
        now = rospy.Time.now()
        if now.is_zero():
            now = stamp
        elapsed = stamp_to_sec(now) - stamp_to_sec(self._gt_init_start_time)
        if elapsed < self.gt_init_wait_s:
            rospy.logwarn_throttle(
                2.0,
                f"[gmmslam] GT init window ({elapsed:.2f}/{self.gt_init_wait_s:.2f}s)",
            )
            return False
        if self._latest_gt_pose_raw is None:
            rospy.logwarn_throttle(
                2.0,
                "[gmmslam] GT init window elapsed but no GT pose received yet",
            )
            return False
        T0 = pose_msg_to_matrix(self._latest_gt_pose_raw.pose)
        self._gt_origin_inv = np.linalg.inv(T0)
        self._gt_path = Path()
        self._gt_path.header.frame_id = self.odom_frame
        rospy.loginfo(f"[gmmslam] GT origin initialized after {elapsed:.2f}s")
        return True

    def _depth_callback(self, msg: Image):
        self._latest_depth_image = msg

    def _cam_info_callback(self, msg: CameraInfo):
        self._latest_camera_info = msg

    def _imu_callback(self, msg: Imu):
        self._latest_imu = msg
        acc = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        gyro = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
        )
        self._imu_buffer.append((msg.header.stamp, acc, gyro))
        t_now = stamp_to_sec(msg.header.stamp)
        keep_s = max(1.0, float(self.imu_buffer_keep_s))
        self._imu_buffer = [
            (ts, a, g)
            for ts, a, g in self._imu_buffer
            if (t_now - stamp_to_sec(ts)) <= keep_s
        ]

    def _imu_measurements_between_secs(self, t_prev: float, t_curr: float):
        """Build preintegration samples [(dt, acc, gyro), ...] in (t_prev, t_curr]."""
        if t_curr <= t_prev:
            return []

        imu_snapshot = list(self._imu_buffer)
        if not imu_snapshot:
            return []

        # Keep a slightly wider range so we have context on both boundaries.
        entries = [
            (stamp_to_sec(ts), acc, gyro)
            for ts, acc, gyro in imu_snapshot
            if (stamp_to_sec(ts) >= (t_prev - 0.05)) and (stamp_to_sec(ts) <= t_curr)
        ]
        if not entries:
            return []
        entries.sort(key=lambda x: x[0])

        # Seed with the latest sample at/before t_prev when available.
        seed = None
        for e in entries:
            if e[0] <= t_prev:
                seed = e
            else:
                break
        if seed is None:
            # No boundary sample at interval start: avoid extrapolating
            # backwards with a future IMU reading.
            return []

        samples = []
        t_last = t_prev
        acc_last = np.asarray(seed[1], dtype=np.float64)
        gyro_last = np.asarray(seed[2], dtype=np.float64)

        for t_i, acc_i, gyro_i in entries:
            if t_i <= t_prev:
                acc_last = np.asarray(acc_i, dtype=np.float64)
                gyro_last = np.asarray(gyro_i, dtype=np.float64)
                continue
            dt = float(t_i - t_last)
            if dt > 1e-6:
                samples.append((dt, acc_last, gyro_last))
                t_last = t_i
            acc_last = np.asarray(acc_i, dtype=np.float64)
            gyro_last = np.asarray(gyro_i, dtype=np.float64)

        # Last segment up to the cloud timestamp.
        dt_tail = float(t_curr - t_last)
        if dt_tail > 1e-6:
            samples.append((dt_tail, acc_last, gyro_last))
        return samples

    def _imu_measurements_between(self, stamp_prev, stamp_curr):
        if stamp_prev is None or stamp_curr is None:
            return []
        return self._imu_measurements_between_secs(
            stamp_to_sec(stamp_prev), stamp_to_sec(stamp_curr)
        )

    def _submap_traj_delta_between(
        self, prev_anchor_key, curr_anchor_key, _prev_anchor_t, _curr_anchor_t
    ):
        """Relative transform from fixed-lag trajectory between two anchor keys."""
        with self.smoother.graph_lock:
            T_prev = self.smoother.pose_by_idx.get(prev_anchor_key)
            T_curr = self.smoother.pose_by_idx.get(curr_anchor_key)
        if T_prev is None or T_curr is None:
            return None
        return np.linalg.inv(T_prev) @ T_curr

    def _get_keyframe_gmm(self, key_idx):
        """Retrieve (gmm, capture_pose) for a keyframe, or None."""
        with self.registration.lock:
            entry = self.registration.local_gmms_by_idx.get(key_idx)
        if entry is None:
            return None
        gmm = entry[1]
        capture_pose = entry[5] if len(entry) > 5 else None
        return (gmm, capture_pose)

    # ==================================================================
    # Main point-cloud callback
    # ==================================================================

    def _pcl_callback(self, msg: PointCloud2):
        try:
            self._pcl_callback_inner(msg)
        except Exception as e:
            rospy.logerr(f"[gmmslam] exception in cloud callback: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())

    def _pcl_callback_inner(self, msg: PointCloud2):
        if not self._first_cloud_seen:
            rospy.loginfo("[gmmslam] first point cloud received, processing started")
            self._first_cloud_seen = True

        stamp = msg.header.stamp
        t_cloud = stamp_to_sec(stamp)
        if (
            self._last_cloud_t_sec_processed is not None
            and t_cloud <= self._last_cloud_t_sec_processed
        ):
            rospy.logwarn_throttle(
                2.0,
                f"[gmmslam] non-monotonic cloud stamp "
                f"({t_cloud:.6f} <= {self._last_cloud_t_sec_processed:.6f}); skipping frame",
            )
            return
        self._last_cloud_t_sec_processed = t_cloud
        if not self._ensure_gt_origin_initialized(stamp):
            return

        # 1. Convert & preprocess
        pts = pc2_to_numpy(msg)
        if pts.shape[0] == 0:
            rospy.logwarn_throttle(
                5.0, "[gmmslam] received empty point cloud, skipping"
            )
            return
        pts = preprocess(pts, self.min_range, self.max_range, self.voxel_size)
        if pts.shape[0] < self.min_points:
            rospy.logwarn_throttle(
                5.0,
                f"[gmmslam] only {pts.shape[0]} points after filtering, skipping",
            )
            return

        # 2. Drain pending async registration results into smoother factors
        if self.enable_async_registration:
            self.registration.drain_results(stamp)

        # 3. Per-frame GT relative motion
        gt_rel_mat = self._sample_noisy_gt_relative_pose3(stamp)

        # 4. Predicted pose = current smoother pose + GT motion
        sm = self.smoother
        with sm.graph_lock:
            if gt_rel_mat is not None:
                predicted = sm.pose @ gt_rel_mat
            else:
                predicted = sm.pose.copy()

        imu_measurements = self._imu_measurements_between(
            self._last_cloud_stamp_for_imu, stamp
        )
        if self.enable_imu_preintegration:
            rospy.logdebug(
                f"[gmmslam] frame {self._frame_count}: imu samples for preintegration="
                f"{len(imu_measurements)}"
            )

        # 5. **EVERY FRAME**: feed the fixed-lag smoother
        curr_odom = self._odom_idx
        prev_odom = curr_odom - 1
        sm.add_frame(
            prev_idx=prev_odom,
            curr_idx=curr_odom,
            stamp=stamp,
            predicted_pose=predicted,
            gt_rel_mat=gt_rel_mat,
            imu_measurements=imu_measurements,
        )
        self._last_cloud_stamp_for_imu = stamp
        self._odom_idx += 1

        if curr_odom % 10 == 0:
            p = predicted[:3, 3]
            has_odom = gt_rel_mat is not None
            rospy.loginfo(
                f"[gmmslam] X({curr_odom}) pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
                f"odom={'GT' if has_odom else 'LOST'}"
            )

        # 6. Keyframe check (global graph + registration ONLY)
        if self._should_add_keyframe(stamp):
            with sm.graph_lock:
                self._last_keyframe_pose = sm.pose.copy()
            self._last_keyframe_t_sec = stamp_to_sec(stamp)
            self._keyframe_odom_indices.append(curr_odom)
            self._keyframe_count += 1

            with sm.graph_lock:
                T_curr = sm.pose_by_idx.get(curr_odom, sm.pose).copy()
            self.global_graph.update_with_keyframe(
                curr_odom, stamp, T_curr, stamp_to_sec(stamp)
            )

            if (
                self.enable_async_registration
                and HAS_SOGMM
                and (
                    self._keyframe_count % self.registration_request_every_n_frames == 0
                )
                and (curr_odom >= self._reg_enqueue_resume_frame)
            ):
                with sm.graph_lock:
                    capture_pose = sm.pose_by_idx.get(curr_odom, sm.pose).copy()
                capture_t = stamp_to_sec(stamp)
                ok = self.registration.enqueue_fit(
                    curr_odom, stamp, pts, capture_t, capture_pose
                )
                if not ok and self.registration_enqueue_cooldown_frames > 0:
                    self._reg_enqueue_resume_frame = (
                        curr_odom + self.registration_enqueue_cooldown_frames
                    )

        # 7. Publish
        try:
            with sm.graph_lock:
                T_pub = sm.pose.copy()
                T_cloud = sm.pose_by_idx.get(curr_odom, T_pub).copy()
            self.visualizer.publish_pose_only(T_pub, stamp)
            self.visualizer.enqueue_frame(stamp, pts, self._frame_count, T_cloud)
        finally:
            self._log_backpressure_periodic(stamp)
            self._frame_count += 1

    # ==================================================================
    # GT noise sampling
    # ==================================================================

    def _sample_noisy_gt_relative_pose3(self, stamp):
        """Build GT-relative motion matrix for BetweenFactor.

        Prefers externally published noisy GT topic. Falls back to internally
        sampled noisy GT if external topic is unavailable.
        """
        if not self.use_noisy_gt_factor:
            return None

        if self._latest_noisy_gt_pose_msg is not None:
            T_curr = pose_msg_to_matrix(self._latest_noisy_gt_pose_msg.pose)
            if self._last_gt_T_for_factor is None:
                self._last_gt_T_for_factor = T_curr
                return None
            T_prev = self._last_gt_T_for_factor
            self._last_gt_T_for_factor = T_curr
            return np.linalg.inv(T_prev) @ T_curr

        if self._latest_gt_pose is None:
            return None
        rospy.logwarn_throttle(
            5.0,
            "[gmmslam] noisy_gt_topic unavailable; falling back to internal GT noise",
        )
        T_curr_gt = pose_msg_to_matrix(self._latest_gt_pose.pose)
        if self._last_gt_T_for_factor is None:
            self._last_gt_T_for_factor = T_curr_gt
            return None
        T_prev_gt = self._last_gt_T_for_factor
        T_rel_gt = np.linalg.inv(T_prev_gt) @ T_curr_gt

        rot_noise = self._rng.normal(0.0, self.gt_noise_sigma_r, size=3)
        trans_noise = self._rng.normal(0.0, self.gt_noise_sigma_t, size=3)
        R_noise = Rotation.from_rotvec(rot_noise).as_matrix()
        T_noisy = np.eye(4, dtype=np.float64)
        T_noisy[:3, :3] = R_noise @ T_rel_gt[:3, :3]
        T_noisy[:3, 3] = T_rel_gt[:3, 3] + trans_noise
        self._last_gt_T_for_factor = T_curr_gt
        return T_noisy

    # ==================================================================
    # Keyframe logic
    # ==================================================================

    def _should_add_keyframe(self, stamp) -> bool:
        """Motion/time-triggered keyframe insertion (for global graph only)."""
        if self._keyframe_count == 0:
            return True
        if self._last_keyframe_pose is None or self._last_keyframe_t_sec is None:
            return True

        if self.keyframe_use_time_trigger:
            dt = stamp_to_sec(stamp) - self._last_keyframe_t_sec
            if dt >= self.keyframe_max_interval_s:
                return True

        with self.smoother.graph_lock:
            T_now = self.smoother.pose.copy()
        T_rel = np.linalg.inv(self._last_keyframe_pose) @ T_now
        dtrans = float(np.linalg.norm(T_rel[:3, 3]))
        drot_deg = float(np.degrees(Rotation.from_matrix(T_rel[:3, :3]).magnitude()))
        if dtrans >= self.keyframe_translation_thresh_m:
            return True
        if drot_deg >= self.keyframe_rotation_thresh_deg:
            return True
        return False

    # ==================================================================
    # Monitoring
    # ==================================================================

    def _log_backpressure_periodic(self, stamp):
        t_sec = stamp_to_sec(stamp)
        if (t_sec - self._last_backpressure_log_t) < 5.0:
            return
        self._last_backpressure_log_t = t_sec
        reg = self.registration
        if reg._dropped_fit_frames > 0 or reg._dropped_result_msgs > 0:
            rospy.logwarn(
                f"[gmmslam] async reg overloaded: "
                f"dropped fits={reg._dropped_fit_frames}, "
                f"dropped results={reg._dropped_result_msgs}"
            )
            reg._dropped_fit_frames = 0
            reg._dropped_result_msgs = 0
        if self.smoother._deferred_batches > 0:
            rospy.logwarn(
                f"[gmmslam] backend overloaded: "
                f"deferred updates={self.smoother._deferred_batches}"
            )
            self.smoother._deferred_batches = 0


def main():
    node = GMMSLAMNode()  # noqa: F841
    rospy.spin()


if __name__ == "__main__":
    main()
