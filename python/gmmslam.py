#!/usr/bin/env python3
"""
GMM-SLAM ROS1 node
------------------
Subscribes to the depth point cloud published by the m500 drone
(/m500_1/mpa/depth/points)

Subscribed topics (names read from ROS params / launch file):
  <lidar_topic>        sensor_msgs/PointCloud2   depth point cloud
  depth/image_raw      sensor_msgs/Image          32FC1 depth image (optional)
  depth/camera_info    sensor_msgs/CameraInfo     depth camera intrinsics (optional)

Published topics:
  ~path                nav_msgs/Path              accumulated trajectory
  ~odom                nav_msgs/Odometry          current odometry estimate
  ~map_cloud           sensor_msgs/PointCloud2    accumulated map points (decimated)
  ~noisy_gt_pose       geometry_msgs/PoseStamped  accumulated noisy GT odometry pose
  ~noisy_gt_path       nav_msgs/Path              accumulated noisy GT trajectory
  ~gmm_markers         visualization_msgs/MarkerArray  per-component spheres scaled by λ₁

Broadcast TF:
  odom_frame → base_frame   current pose
"""

import json
import queue
import threading

import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Image, CameraInfo, Imu
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation

# scipy < 1.4 uses from_dcm; >= 1.4 uses from_matrix
if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm


# ---------------------------------------------------------------------------
# SOGMM import and venv injection (expects gira3d-reconstruction/.venv with sogmm_py installed)
# ---------------------------------------------------------------------------
import sys, os, glob


def _inject_venv(venv_root: str):
    """Add site-packages from a venv into sys.path (so that the modules can be imported once the venv is activated)."""
    pattern = os.path.join(venv_root, "lib", "python3*", "site-packages")
    for sp in glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)


_GIRA_WS = os.environ.get("GIRA_WS", "/root/gira_ws")
_RECONSTRUCTION_VENV = os.path.join(_GIRA_WS, "gira3d-reconstruction", ".venv")
_REGISTRATION_VENV = os.path.join(_GIRA_WS, "gira3d-registration", ".venv")

_inject_venv(_RECONSTRUCTION_VENV)
_inject_venv(_REGISTRATION_VENV)

try:
    from sogmm_py.sogmm import SOGMM

    HAS_SOGMM = True
except ImportError:
    HAS_SOGMM = False
    import logging as _logging

    _logging.warning(
        f"sogmm_py not found - SOGMM fitting disabled. "
        f"Expected venv at {_RECONSTRUCTION_VENV}"
    )

try:
    import gmm_d2d_registration_py
    from utils.save_gmm import save as save_gmm_official

    HAS_GMM_REGISTRATION = True
except ImportError:
    HAS_GMM_REGISTRATION = False
    save_gmm_official = None
    import logging as _logging

    _logging.warning(
        f"gmm_d2d_registration_py not found - D2D registration disabled. "
        f"Expected venv at {_REGISTRATION_VENV}"
    )

try:
    import gtsam
    from gtsam.symbol_shorthand import X

    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    gtsam = None
    X = None
    import logging as _logging

    _logging.warning("gtsam not found - fixed-lag backend disabled.")

try:
    import gtsam_unstable

    HAS_GTSAM_UNSTABLE = True
except ImportError:
    gtsam_unstable = None
    HAS_GTSAM_UNSTABLE = False


# ===========================================================================
# Helpers
# ===========================================================================


def pose_to_transform_stamped(
    T: np.ndarray, stamp, parent: str, child: str
) -> TransformStamped:
    """Convert a 4x4 homogeneous transform to a TransformStamped message."""
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = T[0, 3]
    ts.transform.translation.y = T[1, 3]
    ts.transform.translation.z = T[2, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    ts.transform.rotation.x = q[0]
    ts.transform.rotation.y = q[1]
    ts.transform.rotation.z = q[2]
    ts.transform.rotation.w = q[3]
    return ts


def pose_to_pose_stamped(T: np.ndarray, stamp, frame: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame
    ps.pose.position.x = T[0, 3]
    ps.pose.position.y = T[1, 3]
    ps.pose.position.z = T[2, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    ps.pose.orientation.x = q[0]
    ps.pose.orientation.y = q[1]
    ps.pose.orientation.z = q[2]
    ps.pose.orientation.w = q[3]
    return ps


def pc2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Extract XYZ points from a PointCloud2 message as an (N, 3) float32 array.

    Works with both plain-tuple and numpy-structured-scalar generators returned
    by different versions of sensor_msgs.point_cloud2.read_points.
    """
    rows = [
        [p[0], p[1], p[2]]
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    ]
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(rows, dtype=np.float32)  # shape (N, 3)


def numpy_to_pc2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Convert an (N, 3) float32 array to a PointCloud2 message."""
    header = rospy.Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return pc2.create_cloud_xyz32(header, pts.tolist())


def make_pcld_4d(pts: np.ndarray) -> np.ndarray:
    """Append L2-norm range as 4th feature column expected by SOGMM.

    Parameters
    ----------
    pts : (N, 3) float32  –  XYZ in sensor frame

    Returns
    -------
    pcld_4d : (N, 4) float64  –  [x, y, z, range]
    """
    ranges = np.linalg.norm(pts, axis=1, keepdims=True)  # (N, 1)
    return np.hstack([pts, ranges]).astype(np.float64)


def project_gmm_4d_to_3d(gmm_4d):
    """Project a 4D SOGMM to 3D for registration.

    SOGMM produces 4D GMMs (x, y, z, range), but registration expects 3D (x, y, z).
    Creates a new sklearn GaussianMixture with only the first 3 dimensions.

    Parameters
    ----------
    gmm_4d : sklearn.mixture.GaussianMixture
        4D GMM from SOGMM with shape (N, 4)

    Returns
    -------
    gmm_3d : sklearn.mixture.GaussianMixture
        3D GMM with shape (N, 3)
    """
    from sklearn.mixture import GaussianMixture

    K = gmm_4d.n_components_
    gmm_3d = GaussianMixture(n_components=K, covariance_type="full")

    # Check 4D GMM for NaN/Inf before projection
    if np.any(np.isnan(gmm_4d.means_)):
        rospy.logerr(f"[project_gmm] Input 4D GMM has NaN means!")
        raise ValueError("Input GMM contains NaN values")
    if np.any(np.isnan(gmm_4d.covariances_)):
        rospy.logerr(f"[project_gmm] Input 4D GMM has NaN covariances!")
        raise ValueError("Input GMM contains NaN values")

    # Project means: (K, 4) -> (K, 3)
    gmm_3d.means_ = gmm_4d.means_[:, :3].copy()

    # Project covariances: (K, 4, 4) -> (K, 3, 3)
    # Handle both 3D array and flattened cases
    if gmm_4d.covariances_.ndim == 3:
        gmm_3d.covariances_ = gmm_4d.covariances_[:, :3, :3].copy()
    elif gmm_4d.covariances_.ndim == 2:
        # Already (K, D*D) flattened - reshape, slice, flatten again
        D_orig = gmm_4d.means_.shape[1]
        cov_3d = np.zeros((K, 3, 3))
        for k in range(K):
            cov_4d = gmm_4d.covariances_[k].reshape(D_orig, D_orig)
            cov_3d[k] = cov_4d[:3, :3]
        gmm_3d.covariances_ = cov_3d
    else:
        raise ValueError(f"Unexpected covariance shape: {gmm_4d.covariances_.shape}")

    # Copy weights
    gmm_3d.weights_ = gmm_4d.weights_.copy()
    return filter_well_conditioned_gmm(gmm_3d)


def filter_well_conditioned_gmm(gmm_3d, reg: float = 1e-4):
    """Keep only components with well-conditioned 3x3 covariances.

    Only drops components with NaN/Inf or non-positive eigenvalues after
    regularization.  Planar (near-singular) components are intentionally kept
    because isoplanar_registration specifically relies on them.
    """
    from sklearn.mixture import GaussianMixture

    if hasattr(gmm_3d, "means_"):
        K = int(gmm_3d.means_.shape[0])
    else:
        K = int(getattr(gmm_3d, "n_components_", getattr(gmm_3d, "n_components")))
    keep = []
    dropped_bad = []

    for k in range(K):
        cov = gmm_3d.covariances_[k].copy()
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            dropped_bad.append(k)
            continue

        # Symmetrize and regularize before eigenvalue check.
        cov = 0.5 * (cov + cov.T) + reg * np.eye(3)
        try:
            eigvals = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            dropped_bad.append(k)
            continue

        if (
            np.any(np.isnan(eigvals))
            or np.any(np.isinf(eigvals))
            or eigvals.min() <= 0.0
        ):
            dropped_bad.append(k)
            continue

        keep.append((k, cov))

    if dropped_bad:
        rospy.logwarn_throttle(
            10.0,
            f"[filter_gmm] dropped {len(dropped_bad)}/{K} bad components (NaN/Inf/non-PD): {dropped_bad[:5]}",
        )

    if not keep:
        raise ValueError("No well-conditioned covariance matrices left after filtering")

    keep_idx = [i for i, _ in keep]
    covs = np.stack([cov for _, cov in keep], axis=0)  # reg already applied above
    weights = gmm_3d.weights_[keep_idx].astype(np.float64)
    weights_sum = weights.sum()
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        raise ValueError("Invalid weights after filtering GMM components")
    weights /= weights_sum

    filtered = GaussianMixture(n_components=len(keep_idx), covariance_type="full")
    filtered.means_ = gmm_3d.means_[keep_idx].copy()
    filtered.covariances_ = covs
    filtered.weights_ = weights
    filtered.n_components_ = len(keep_idx)
    filtered.converged_ = True
    filtered.n_iter_ = 0
    filtered.lower_bound_ = -np.inf
    # Compute precisions_cholesky_ with extra regularization for near-planar covs.
    try:
        filtered.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
    except np.linalg.LinAlgError:
        covs_safe = covs + 1e-3 * np.eye(3)
        filtered.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs_safe))

    if len(keep_idx) != K:
        rospy.loginfo(
            f"[filter_gmm] keeping {len(keep_idx)}/{K} well-conditioned components"
        )

    return filtered


def save_gmm_to_file(gmm, filepath: str):
    """Save a GaussianMixture to .gmm format using gira3d's official save utility.

    SOGMM produces 4D GMMs, so we project to 3D before saving.
    """
    if not HAS_GMM_REGISTRATION or save_gmm_official is None:
        raise RuntimeError("save_gmm utility not available")

    # Project 4D -> 3D if needed
    if gmm.means_.shape[1] == 4:
        gmm_3d = project_gmm_4d_to_3d(gmm)
    else:
        gmm_3d = filter_well_conditioned_gmm(gmm)

    if gmm_3d is None:
        return None

    # Validate GMM before saving
    nan_count = 0
    inf_count = 0
    for k in range(gmm_3d.n_components_):
        if np.any(np.isnan(gmm_3d.means_[k])):
            nan_count += 1
            rospy.logwarn(f"[save_gmm] Component {k} has NaN mean")
        if np.any(np.isnan(gmm_3d.covariances_[k])):
            nan_count += 1
            rospy.logwarn(f"[save_gmm] Component {k} has NaN covariance")
        if np.any(np.isinf(gmm_3d.covariances_[k])):
            inf_count += 1
            rospy.logwarn(f"[save_gmm] Component {k} has Inf covariance")

    if nan_count > 0 or inf_count > 0:
        rospy.logerr(
            f"[save_gmm] GMM has {nan_count} NaN and {inf_count} Inf components!"
        )

    # Use official gira3d save function
    save_gmm_official(filepath, gmm_3d)

    # Log file save with stats
    # rospy.loginfo(
    #     f"[save_gmm] Saved {gmm_3d.n_components_} components to {os.path.basename(filepath)}"
    # )


def plot_gmm_3d(
    gmm, sigma: float = 1.0, max_ellipsoids: int = 50, title: str = "GMM 3D Gaussians"
):
    """Visualize a 3D GMM as ellipsoids in a native matplotlib window.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Fitted 3D (or 4D) GMM — 4D is auto-projected to 3D.
    sigma : float
        Ellipsoid half-axis in standard deviations (default: 1.0).
    max_ellipsoids : int
        Draw only the top-N components by weight (default: 50).
    title : str
        Figure title.
    """
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gmm.means_.shape[1] == 4:
        gmm = project_gmm_4d_to_3d(gmm)

    K = gmm.n_components_
    n_u, n_v = 20, 14
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)).ravel(),
            np.outer(np.sin(u), np.sin(v)).ravel(),
            np.outer(np.ones(n_u), np.cos(v)).ravel(),
        ],
        axis=1,
    )

    top_idx = np.argsort(gmm.weights_)[::-1][:max_ellipsoids]
    max_w = gmm.weights_[top_idx[0]]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{title}  (K={K}, showing top-{min(max_ellipsoids, K)})")

    for k in top_idx:
        mu = gmm.means_[k]
        cov = gmm.covariances_[k]
        w = gmm.weights_[k]
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        axes = eigvecs * (np.sqrt(eigvals) * sigma)
        epts = (axes @ sphere.T).T + mu
        ex = epts[:, 0].reshape(n_u, n_v)
        ey = epts[:, 1].reshape(n_u, n_v)
        ez = epts[:, 2].reshape(n_u, n_v)
        alpha = 0.15 + 0.45 * (w / max_w)
        ax.plot_surface(ex, ey, ez, color="steelblue", alpha=alpha, linewidth=0)

    ax.scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        gmm.means_[:, 2],
        c="red",
        s=40,
        zorder=5,
        label="Means",
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    return fig


# ===========================================================================
# Pre-processing
# ===========================================================================


def preprocess(
    pts: np.ndarray, min_range: float, max_range: float, voxel_size: float
) -> np.ndarray:
    """Range filter + voxel-grid downsampling."""
    ranges = np.linalg.norm(pts, axis=1)
    mask = (ranges >= min_range) & (ranges <= max_range)
    pts = pts[mask]

    if pts.shape[0] == 0:
        return pts

    # if voxel_size > 0.0 and HAS_OPEN3D:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    #     pcd = pcd.voxel_down_sample(voxel_size)
    #     pts = np.asarray(pcd.points, dtype=np.float32)

    return pts


# ===========================================================================
# GMM-SLAM node
# ===========================================================================


class GMMSLAMNode:

    def __init__(self):
        rospy.init_node("gmmslam_node", anonymous=False)

        def _param_with_alias(new_name: str, old_name: str, default):
            """Read new param name first, then legacy name."""
            if rospy.has_param(new_name):
                return rospy.get_param(new_name)
            if rospy.has_param(old_name):
                rospy.logwarn_once(
                    f"[gmmslam] parameter '{old_name}' is deprecated, use '{new_name}'"
                )
                return rospy.get_param(old_name)
            return default

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.lidar_topic = rospy.get_param("~lidar_topic", "/m500_1/mpa/depth/points")
        self.imu_topic = rospy.get_param("~imu_topic", "/m500_1/mavros/imu/data")
        self._imu_buffer = []
        self._latest_imu = None
        self.sensor_frame = rospy.get_param("~sensor_frame", "depth_camera_1")
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "m500_1_base_link")

        self.min_range = rospy.get_param("~min_range", 0.1)
        self.max_range = rospy.get_param("~max_range", 10.0)
        self.voxel_size = rospy.get_param("~voxel_leaf_size", 0.05)
        self.min_points = rospy.get_param("~min_points", 50)

        # SOGMM parameters
        # bandwidth = bandwidth of the kernel used for the Gaussian Blurring
        # Mean Shift (GBMS) within the SOGMM algorithm
        # so higher => fewer components, lower => more components
        self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.95)
        self.sogmm_compute = rospy.get_param("~sogmm_compute", "CPU")  # "CPU" or "GPU"
        self.sogmm_max_points = rospy.get_param(
            "~sogmm_max_points", 400
        )  # subsample before fitting (0 = no cap)
        self.sogmm_n_components = rospy.get_param(
            "~sogmm_n_components", 100
        )  # fixed component count for now
        self.gmm_marker_sigma = float(
            rospy.get_param("~gmm_marker_sigma", 3.0)
        )  # sigma multiplier for ellipsoid scale in RViz (3.0 = 3-sigma)
        self.plot_first_frame = rospy.get_param("~plot_first_frame", False)
        # Backend parameters (fixed-lag smoothing only)
        self.use_fixed_lag_smoother = rospy.get_param("~use_fixed_lag_smoother", True)
        self.fixed_lag_s = rospy.get_param("~fixed_lag_s", 0.25)
        self.odom_noise_sigma_t = rospy.get_param("~odom_noise_sigma_t", 0.06)
        self.odom_noise_sigma_r = rospy.get_param("~odom_noise_sigma_r", 0.06)
        self.enable_async_registration = rospy.get_param(
            "~enable_async_registration", True
        )
        # Enqueue one registration request every N frames.
        self.registration_request_every_n_frames = max(
            1,
            int(
                _param_with_alias(
                    "~registration_request_every_n_frames",
                    "~registration_request_stride",
                    _param_with_alias(
                        "~registration_request_stride", "~registration_fit_stride", 3
                    ),
                )
            ),
        )
        self.registration_score_threshold = float(
            rospy.get_param("~registration_score_threshold", -1.0e9)
        )
        # If score exceeds this threshold, add an extra strong factor on that edge.
        self.loop_closure_score_threshold = float(
            _param_with_alias(
                "~strong_factor_score_threshold", "~loop_closure_score_threshold", 2.0
            )
        )
        self.loop_closure_sigma_t = float(
            _param_with_alias("~strong_factor_sigma_t", "~loop_closure_sigma_t", 0.01)
        )
        self.loop_closure_sigma_r = float(
            _param_with_alias("~strong_factor_sigma_r", "~loop_closure_sigma_r", 0.01)
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
        # Insert one async registration factor every N frames.
        self.registration_factor_every_n_frames = max(
            1,
            int(
                _param_with_alias(
                    "~registration_factor_every_n_frames",
                    "~registration_factor_stride",
                    _param_with_alias(
                        "~registration_factor_stride", "~registration_apply_stride", 1
                    ),
                )
            ),
        )
        # Whether to solve and publish immediately when an async registration arrives.
        self.solve_on_registration_result = bool(
            _param_with_alias(
                "~solve_after_registration_factor",
                "~solve_on_registration_result",
                False,
            )
        )
        self.gt_init_wait_s = float(rospy.get_param("~gt_init_wait_s", 3.0))
        self.use_noisy_gt_factor = rospy.get_param("~use_noisy_gt_factor", True)
        self.gt_noise_sigma_t = rospy.get_param("~gt_noise_sigma_t", 0.00)
        self.gt_noise_sigma_r = rospy.get_param("~gt_noise_sigma_r", 0.00)
        self.gt_factor_sigma_t = rospy.get_param("~gt_factor_sigma_t", 0.0001)
        self.gt_factor_sigma_r = rospy.get_param("~gt_factor_sigma_r", 0.0001)
        self.gt_noise_seed = rospy.get_param("~gt_noise_seed", -1)

        rospy.loginfo(f"[gmmslam] lidar_topic  : {self.lidar_topic}")
        rospy.loginfo(f"[gmmslam] sensor_frame : {self.sensor_frame}")
        rospy.loginfo(f"[gmmslam] odom_frame   : {self.odom_frame}")
        rospy.loginfo(f"[gmmslam] base_frame   : {self.base_frame}")
        rospy.loginfo(
            f"[gmmslam] range        : [{self.min_range}, {self.max_range}] m"
        )
        rospy.loginfo(f"[gmmslam] voxel_size   : {self.voxel_size} m")

        if HAS_SOGMM:
            rospy.loginfo(f"[gmmslam] SOGMM bandwidth     : {self.sogmm_bandwidth}")
            rospy.loginfo(f"[gmmslam] SOGMM compute       : {self.sogmm_compute}")
            rospy.loginfo(
                f"[gmmslam] SOGMM max_points    : {self.sogmm_max_points} (0=no cap)"
            )
            if self.sogmm_n_components > 0:
                rospy.loginfo(
                    f"[gmmslam] SOGMM n_components  : {self.sogmm_n_components} (fixed)"
                )
            else:
                rospy.loginfo(
                    "[gmmslam] SOGMM n_components  : auto (MeanShift enabled)"
                )
        else:
            rospy.logwarn("[gmmslam] SOGMM unavailable – fitting will be skipped")
        rospy.loginfo(f"[gmmslam] use_fixed_lag    : {self.use_fixed_lag_smoother}")
        rospy.loginfo(f"[gmmslam] fixed_lag_s      : {self.fixed_lag_s}")
        rospy.loginfo(f"[gmmslam] async reg enabled : {self.enable_async_registration}")
        rospy.loginfo(
            f"[gmmslam] reg request every N frames: {self.registration_request_every_n_frames}"
        )
        rospy.loginfo(
            f"[gmmslam] reg score threshold: {self.registration_score_threshold}"
        )
        rospy.loginfo(
            f"[gmmslam] reg factor every N frames : {self.registration_factor_every_n_frames}"
        )
        rospy.loginfo(
            f"[gmmslam] solve after reg factor: {self.solve_on_registration_result}"
        )
        rospy.loginfo(
            f"[gmmslam] strong factor threshold: {self.loop_closure_score_threshold}"
        )
        rospy.loginfo(
            f"[gmmslam] odom noise (t/r) : {self.odom_noise_sigma_t} / {self.odom_noise_sigma_r}"
        )
        rospy.loginfo(f"[gmmslam] noisy GT factor : {self.use_noisy_gt_factor}")
        if self.use_noisy_gt_factor:
            rospy.loginfo(
                f"[gmmslam] GT noise (t/r)   : {self.gt_noise_sigma_t} / {self.gt_noise_sigma_r}"
            )
            rospy.loginfo(
                f"[gmmslam] GT factor (t/r)  : {self.gt_factor_sigma_t} / {self.gt_factor_sigma_r}"
            )
        elif not self.enable_async_registration:
            rospy.logwarn(
                "[gmmslam] both noisy GT and async registration are disabled; "
                "the graph will only receive identity base edges."
            )
        if not self.use_fixed_lag_smoother:
            rospy.logwarn("[gmmslam] forcing fixed-lag backend ON as requested")
            self.use_fixed_lag_smoother = True

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.pose = np.eye(4, dtype=np.float64)  # T_world_base (cumulative)
        self.prev_pcd = None  # open3d PointCloud (previous scan)
        self.path = Path()
        self.map_pts: list = []  # accumulated map points (list of np arrays)
        self.map_decimate = 5  # add 1 out of N frames to map
        self._map_cloud_last_pub_t = 0.0  # throttle np.vstack+publish to 2 Hz

        self._frame_count = 0
        self._latest_key_idx = 0
        self._first_cloud_seen = False
        self._reg_lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self._key_t_sec: dict = {}  # frame_idx -> t_sec for lag-window checks
        self._reg_fit_queue: queue.Queue = queue.Queue(
            maxsize=self.registration_queue_size
        )
        self._reg_result_queue: queue.Queue = queue.Queue(
            maxsize=self.registration_result_queue_size
        )
        self._gmm_paths_by_idx = {}
        self._dropped_reg_fit_frames = 0
        self._dropped_reg_result_msgs = 0
        self._reg_enqueue_resume_frame = 0
        self._last_backpressure_log_t = 0.0

        # SOGMM state
        # sg         – incremental global SOGMM (builds up over all scans)
        # local_gmms – list of per-scan local GMMf4 models (for D2D registration)
        if HAS_SOGMM:
            self.sg = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            self.local_gmms = []  # list[(stamp, GMMf4)]
        else:
            self.sg = None
            self.local_gmms = []

        # D2D registration state
        self.prev_gmm_path = None  # path to previous frame's .gmm file
        self.gmm_dir = rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms")
        os.makedirs(self.gmm_dir, exist_ok=True)
        # Fixed-lag backend state (same pattern as GTSAM example)
        self._odom_noise = None
        self._prior_noise = None
        self._loop_closure_noise = None
        self._fixed_lag = None
        self._new_factors = None
        self._new_values = None
        self._new_timestamps = None
        self._fixed_lag_initialized = False
        self._gt_factor_noise = None
        self._last_gt_T_for_factor = None
        self._rng = (
            np.random.default_rng(self.gt_noise_seed)
            if self.gt_noise_seed >= 0
            else np.random.default_rng()
        )
        if not HAS_GTSAM:
            raise RuntimeError("Fixed-lag backend requested but gtsam is unavailable")
        if not (
            HAS_GTSAM_UNSTABLE
            and hasattr(gtsam_unstable, "IncrementalFixedLagSmoother")
        ):
            raise RuntimeError(
                "Fixed-lag backend requested but gtsam_unstable.IncrementalFixedLagSmoother is unavailable"
            )
        sigmas = np.array(
            [
                self.odom_noise_sigma_r,
                self.odom_noise_sigma_r,
                self.odom_noise_sigma_r,
                self.odom_noise_sigma_t,
                self.odom_noise_sigma_t,
                self.odom_noise_sigma_t,
            ],
            dtype=np.float64,
        )
        self._odom_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self._odom_noise_lost = gtsam.noiseModel.Diagonal.Sigmas(sigmas * 10.0)
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas * 0.1)
        loop_sigmas = np.array(
            [
                self.loop_closure_sigma_r,
                self.loop_closure_sigma_r,
                self.loop_closure_sigma_r,
                self.loop_closure_sigma_t,
                self.loop_closure_sigma_t,
                self.loop_closure_sigma_t,
            ],
            dtype=np.float64,
        )
        self._loop_closure_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)
        gt_sigmas = np.array(
            [
                self.gt_factor_sigma_r,
                self.gt_factor_sigma_r,
                self.gt_factor_sigma_r,
                self.gt_factor_sigma_t,
                self.gt_factor_sigma_t,
                self.gt_factor_sigma_t,
            ],
            dtype=np.float64,
        )
        self._gt_factor_noise = gtsam.noiseModel.Diagonal.Sigmas(gt_sigmas)
        self._fixed_lag = gtsam_unstable.IncrementalFixedLagSmoother(self.fixed_lag_s)
        map_ctor = getattr(gtsam_unstable, "FixedLagSmootherKeyTimestampMap", None)
        if map_ctor is None:
            raise RuntimeError(
                "Fixed-lag backend requested but gtsam_unstable.FixedLagSmootherKeyTimestampMap is unavailable"
            )
        self._map_ctor = map_ctor  # saved for container re-creation in the async path
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
        self._new_timestamps = map_ctor()
        # Queue for the dedicated GTSAM backend thread (maxsize=2 keeps latency low).
        self._gtsam_queue: queue.Queue = queue.Queue(maxsize=2)
        rospy.loginfo(
            f"[gmmslam] backend initialised: IncrementalFixedLagSmoother (lag={self.fixed_lag_s:.2f}s)"
        )

        # ------------------------------------------------------------------
        # TF broadcaster
        # ------------------------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Ground truth
        self.gt_topic = rospy.get_param(
            "~gt_topic", "/m500_1/mavros/local_position/pose"
        )
        self._gt_path = Path()
        self._gt_path.header.frame_id = self.odom_frame
        self._latest_gt_pose = None
        self._latest_gt_pose_raw = None
        self._gt_origin_inv = None
        self._gt_init_start_time = None
        self._noisy_gt_pose = np.eye(4, dtype=np.float64)
        self._noisy_gt_path = Path()
        self._noisy_gt_path.header.frame_id = self.odom_frame
        self._graph_node_markers = MarkerArray()
        self._graph_nodes_last_pub_t = 0.0  # throttle full-array publish to 2 Hz

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.path_pub = rospy.Publisher("~path", Path, queue_size=1)
        self.odom_pub = rospy.Publisher("~odom", Odometry, queue_size=1)
        self.cloud_pub = rospy.Publisher("~map_cloud", PointCloud2, queue_size=1)
        self.gt_path_pub = rospy.Publisher("~gt_path", Path, queue_size=1)
        self.gt_pose_pub = rospy.Publisher("~gt_pose", PoseStamped, queue_size=1)
        self.noisy_gt_pose_pub = rospy.Publisher(
            "~noisy_gt_pose", PoseStamped, queue_size=1
        )
        self.noisy_gt_path_pub = rospy.Publisher("~noisy_gt_path", Path, queue_size=1)
        self.gmm_markers_pub = rospy.Publisher(
            "~gmm_markers", MarkerArray, queue_size=1
        )
        self.graph_nodes_pub = rospy.Publisher(
            "~graph_nodes", MarkerArray, queue_size=1, latch=True
        )
        self.reg_request_pub = rospy.Publisher(
            self.registration_request_topic, String, queue_size=10
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        rospy.Subscriber(
            self.lidar_topic, PointCloud2, self._pcl_callback, queue_size=1
        )
        rospy.Subscriber(self.gt_topic, PoseStamped, self._gt_callback, queue_size=1)
        rospy.Subscriber(
            self.registration_result_topic,
            String,
            self._registration_result_callback,
            queue_size=50,
        )

        rospy.Subscriber(self.imu_topic, Imu, self._imu_callback, queue_size=1)

        # Depth image and camera info (for future use / logging)
        depth_ns = "/".join(self.lidar_topic.split("/")[:-1])  # strip "points"
        rospy.Subscriber(
            depth_ns + "/image_raw", Image, self._depth_callback, queue_size=1
        )
        rospy.Subscriber(
            depth_ns + "/camera_info", CameraInfo, self._cam_info_callback, queue_size=1
        )
        rospy.loginfo(f"[gmmslam] gt_topic     : {self.gt_topic}")
        rospy.loginfo(
            f"[gmmslam] reg topics   : {self.registration_request_topic} -> {self.registration_result_topic}"
        )

        if self.enable_async_registration:
            self._reg_worker = threading.Thread(
                target=self._registration_fit_worker_loop, daemon=True
            )
            self._reg_worker.start()
            # NOTE: registration results are drained on the lidar callback thread
            # (_pcl_callback_inner) to avoid _graph_lock contention with the odom
            # update. We do NOT start a separate result-worker thread here.

        # Dedicated visualization thread — runs _publish_scan_products so that
        # growing-message serialization (path, map cloud, GMM markers) never
        # blocks the lidar callback thread.  maxsize=1 means we always drop the
        # oldest pending frame if the vis thread is still busy.
        self._vis_queue: queue.Queue = queue.Queue(maxsize=1)
        self._vis_worker = threading.Thread(target=self._vis_loop, daemon=True)
        self._vis_worker.start()

        # Dedicated GTSAM backend thread — keeps _fixed_lag.update() and
        # calculateEstimate() off the lidar callback thread so publishing never
        # blocks on GTSAM latency or GIL contention with SOGMM.
        self._gtsam_worker = threading.Thread(
            target=self._gtsam_backend_loop, daemon=True
        )
        self._gtsam_worker.start()

        # With use_sim_time=true rospy holds callbacks until the clock ticks.
        # Block here until time is valid so we don't silently drop messages.
        if rospy.get_param("/use_sim_time", False):
            rospy.loginfo("[gmmslam] use_sim_time=true detected, waiting for /clock …")
            while not rospy.is_shutdown() and rospy.Time.now().is_zero():
                rospy.sleep(0.1)
            rospy.loginfo("[gmmslam] clock started, ready")
        self._gt_init_start_time = rospy.Time.now()
        rospy.loginfo(
            f"[gmmslam] waiting {self.gt_init_wait_s:.1f}s before GT frame reset initialization"
        )

        rospy.loginfo("[gmmslam] node ready, waiting for point clouds …")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _gt_callback(self, msg: PoseStamped):
        """Accumulate GT path after resetting first valid pose to odom origin."""
        self._latest_gt_pose_raw = msg
        if not self._ensure_gt_origin_initialized(msg.header.stamp):
            return

        T_gt_webots = self._pose_msg_to_matrix(msg.pose)
        T_gt_odom = self._gt_origin_inv @ T_gt_webots
        ps = pose_to_pose_stamped(T_gt_odom, msg.header.stamp, self.odom_frame)
        self._latest_gt_pose = ps
        self._gt_path.header.stamp = ps.header.stamp
        self._gt_path.poses.append(ps)
        self.gt_pose_pub.publish(ps)
        self.gt_path_pub.publish(self._gt_path)

    def _ensure_gt_origin_initialized(self, stamp) -> bool:
        """Initialize GT origin after configured startup wait."""
        if self._gt_origin_inv is not None:
            return True
        if self._gt_init_start_time is None:
            self._gt_init_start_time = rospy.Time.now()
        now = rospy.Time.now()
        if now.is_zero():
            now = stamp
        elapsed = self._stamp_to_sec(now) - self._stamp_to_sec(self._gt_init_start_time)
        if elapsed < self.gt_init_wait_s:
            rospy.logwarn_throttle(
                2.0,
                f"[gmmslam] waiting for GT initialization window ({elapsed:.2f}/{self.gt_init_wait_s:.2f}s)",
            )
            return False
        if self._latest_gt_pose_raw is None:
            rospy.logwarn_throttle(
                2.0,
                "[gmmslam] GT init window elapsed but no GT pose received yet on gt_topic",
            )
            return False

        T0 = self._pose_msg_to_matrix(self._latest_gt_pose_raw.pose)
        self._gt_origin_inv = np.linalg.inv(T0)
        self._gt_path = Path()
        self._gt_path.header.frame_id = self.odom_frame
        rospy.loginfo(
            f"[gmmslam] GT origin initialized after {elapsed:.2f}s (GT reset to odom origin)"
        )
        return True

    def _depth_callback(self, msg: Image):
        """Store the latest 32FC1 depth image (reserved for future processing)."""
        self._latest_depth_image = msg

    def _cam_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics (reserved for future use)."""
        self._latest_camera_info = msg

    def _imu_callback(self, msg: Imu):
        """Buffer IMU measurements for preintegration / deskewing."""
        self._latest_imu = msg
        acc = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        gyro = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )
        self._imu_buffer.append((msg.header.stamp, acc, gyro))
        # Keep only the latest 1 second of data, 200hz
        self._imu_buffer = self._imu_buffer[-200:]

    def _registration_fit_worker_loop(self):
        """Background worker: fit GMMs and emit async registration requests."""
        while not rospy.is_shutdown():
            try:
                item = self._reg_fit_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            try:
                frame_idx, stamp, pts = item
                gmm_path = self._fit_sogmm(stamp, pts, frame_idx=frame_idx)
                if gmm_path is None:
                    continue

                with self._reg_lock:
                    self._gmm_paths_by_idx[frame_idx] = gmm_path
                    # Find the most recently fitted frame before this one.
                    prev_idx = max(
                        (k for k in self._gmm_paths_by_idx.keys() if k < frame_idx),
                        default=None,
                    )
                    prev_path = (
                        self._gmm_paths_by_idx.get(prev_idx)
                        if prev_idx is not None
                        else None
                    )

                if prev_path is None:
                    continue

                payload = {
                    "prev_idx": int(prev_idx),
                    "curr_idx": int(frame_idx),
                    "stamp": float(self._stamp_to_sec(stamp)),
                    "source_path": gmm_path,
                    "target_path": prev_path,
                }
                self.reg_request_pub.publish(String(data=json.dumps(payload)))

                # Keep only recent indices in memory.
                with self._reg_lock:
                    min_keep = max(0, frame_idx - 200)
                    stale = [k for k in self._gmm_paths_by_idx.keys() if k < min_keep]
                    for k in stale:
                        del self._gmm_paths_by_idx[k]
            except Exception as e:
                rospy.logerr_throttle(
                    2.0, f"[gmmslam] registration fit worker error: {e}"
                )

    def _registration_result_callback(self, msg: String):
        """Parse async registration result and enqueue graph update."""
        try:
            rospy.logdebug(f"[gmmslam] received registration result")
            data = json.loads(msg.data)
            if not bool(data.get("success", False)):
                return
            score = float(data.get("score", float("-inf")))
            if score < self.registration_score_threshold:
                return
            prev_idx = int(data["prev_idx"])
            curr_idx = int(data["curr_idx"])
            if (curr_idx % self.registration_factor_every_n_frames) != 0:
                return
            T_list = data["transform"]
            T = np.array(T_list, dtype=np.float64).reshape(4, 4)
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, f"[gmmslam] bad registration result message: {e}"
            )
            return

        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            return

        force_loop = score >= self.loop_closure_score_threshold
        try:
            self._reg_result_queue.put_nowait((prev_idx, curr_idx, T, force_loop))
        except queue.Full:
            self._dropped_reg_result_msgs += 1

    def _log_backpressure_periodic(self, stamp):
        """Emit sparse queue-backpressure logs outside hot loops."""
        t_sec = self._stamp_to_sec(stamp)
        if (t_sec - self._last_backpressure_log_t) < 5.0:
            return
        self._last_backpressure_log_t = t_sec
        if self._dropped_reg_fit_frames > 0 or self._dropped_reg_result_msgs > 0:
            rospy.logwarn(
                "[gmmslam] async registration overloaded: "
                f"dropped fit frames={self._dropped_reg_fit_frames}, "
                f"dropped reg results={self._dropped_reg_result_msgs}"
            )
            self._dropped_reg_fit_frames = 0
            self._dropped_reg_result_msgs = 0

    def _registration_result_worker_loop(self):
        """Unused — registration results are now drained on the lidar callback thread."""
        pass

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
        if not self._ensure_gt_origin_initialized(stamp):
            return

        # 1. Convert to numpy
        pts = pc2_to_numpy(msg)
        rospy.logdebug(f"[gmmslam] raw pts: {pts.shape[0]}")
        if pts.shape[0] == 0:
            rospy.logwarn_throttle(
                5.0, "[gmmslam] received empty point cloud, skipping"
            )
            return

        # 2. Pre-process
        pts = preprocess(pts, self.min_range, self.max_range, self.voxel_size)
        if pts.shape[0] < self.min_points:
            rospy.logwarn_throttle(
                5.0,
                f"[gmmslam] only {pts.shape[0]} points after filtering (< {self.min_points}), skipping",
            )
            return
        # Stage any pending registration factors into _new_factors BEFORE the
        # odom update so everything is committed in a single _fixed_lag.update()
        # call.  This avoids a second expensive GTSAM update per callback.
        if self.enable_async_registration:
            while True:
                try:
                    r_prev, r_curr, r_T, r_loop = self._reg_result_queue.get_nowait()
                    self._stage_registration_factor(
                        r_prev, r_curr, r_T, force_loop=r_loop
                    )
                except queue.Empty:
                    break
                except Exception as e:
                    rospy.logwarn_throttle(
                        2.0, f"[gmmslam] registration drain error: {e}"
                    )
                    break

        # Stage factors and enqueue GTSAM solve (non-blocking — solve runs in
        # _gtsam_backend_loop thread).  _latest_key_idx is updated optimistically
        # here on the lidar thread so that registration factor lag-window checks
        # on the next callback see the correct value immediately.
        _staged_ok = self._fixed_lag_update_from_odom(
            prev_idx=self._frame_count - 1,
            curr_idx=self._frame_count,
            stamp=stamp,
        )
        if _staged_ok:
            self._latest_key_idx = self._frame_count
        else:
            rospy.logwarn_throttle(
                2.0,
                "[gmmslam] factor staging failed; keeping previous published pose",
            )

        # Offload expensive fitting and registration request generation.
        if (
            self.enable_async_registration
            and HAS_SOGMM
            and (self._frame_count % self.registration_request_every_n_frames == 0)
            and (self._frame_count >= self._reg_enqueue_resume_frame)
        ):
            try:
                self._reg_fit_queue.put_nowait((self._frame_count, stamp, pts.copy()))
            except queue.Full:
                self._dropped_reg_fit_frames += 1
                if self.registration_enqueue_cooldown_frames > 0:
                    self._reg_enqueue_resume_frame = (
                        self._frame_count + self.registration_enqueue_cooldown_frames
                    )

        # _publish_pose_only (TF + odom) stays on the lidar thread — it is
        # small and latency-sensitive.  Everything else is handed to the vis
        # thread via a drop queue so growing messages never stall this thread.
        try:
            self._publish_pose_only(stamp)
            try:
                self._vis_queue.put_nowait((stamp, pts.copy(), self._frame_count))
            except queue.Full:
                pass  # vis thread is busy; drop this frame's scan products
        finally:
            self._log_backpressure_periodic(stamp)
            self._frame_count += 1

    # ------------------------------------------------------------------
    # SOGMM fitting
    # https://github.com/gira3d/sogmm_py/blob/697cbeaf10c60fa80445c25e05dd9d720a296869/src/sogmm_py/sogmm.py
    # https://github.com/gira3d/gmm_d2d_registration_examples/blob/082377d71d49201955b253e796ac6ca94fbfa904/python/create_and_save_gmm_example.py
    # Sklean GMM Fitting: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
    # ------------------------------------------------------------------

    def _fit_sogmm(self, stamp, pts: np.ndarray, frame_idx: int = None) -> str:
        """Fit a SOGMM to the current preprocessed scan.

        Builds the 4D point cloud [x, y, z, range], runs SOGMM.fit() which:
          1. Uses MeanShift (bandwidth) to estimate the number of components.
          2. Initialises responsibilities with K-Means++.
          3. Runs EM to obtain the local GMMf4 model.
          4. Merges the local model into the global incremental model (sg.model).

        The local model is saved to a .gmm file and returned.

        Returns
        -------
        gmm_path : str or None
            Path to the saved .gmm file, or None if fitting failed.
        """
        idx = self._frame_count if frame_idx is None else int(frame_idx)
        rospy.logdebug(
            f"[gmmslam] _fit_sogmm called for frame {idx}, HAS_SOGMM={HAS_SOGMM}"
        )

        if not HAS_SOGMM:
            return None

        pcld_4d = make_pcld_4d(pts)  # (N, 4)  [x, y, z, range]

        # Subsample to cap fitting time (MeanShift + EM scale badly with N).
        # sogmm_max_points=0 disables the cap.
        if self.sogmm_max_points > 0 and pcld_4d.shape[0] > self.sogmm_max_points:
            sample_idx = np.random.choice(
                pcld_4d.shape[0], self.sogmm_max_points, replace=False
            )
            pcld_4d = pcld_4d[sample_idx]

        try:
            sg_local = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            if self.sogmm_n_components > 0:
                local_model = sg_local.gmm_fit(pcld_4d, self.sogmm_n_components)
            else:
                local_model = sg_local.fit(pcld_4d)
        except Exception as e:
            rospy.logerr(f"[gmmslam] SOGMM fit failed on frame {idx}: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

        if local_model is None:
            rospy.logwarn_throttle(
                5.0, f"[gmmslam] SOGMM fit returned None on frame {idx}"
            )
            return None

        self.local_gmms.append((stamp, local_model))
        n_local = local_model.n_components_
        rospy.loginfo(
            f"[gmmslam] frame {idx:4d} | "
            f"local GMM: {n_local:3d} components (fresh per-frame fit)"
        )

        if self.plot_first_frame and idx == 0:
            rospy.loginfo("[gmmslam] plot_first_frame=True — opening GMM visualisation")
            plot_gmm_3d(
                local_model,
                sigma=1.0,
                title=f"First frame local GMM (frame 0, K={n_local})",
            )

        gmm_path = os.path.join(self.gmm_dir, f"frame_{idx:06d}.gmm")
        try:
            save_gmm_to_file(local_model, gmm_path)

            # if os.path.exists(gmm_path):
            #     file_size = os.path.getsize(gmm_path)
            #     rospy.loginfo(f"[gmmslam] GMM file size: {file_size} bytes")
            #     if file_size == 0:
            #         rospy.logerr(f"[gmmslam] GMM file is empty!")
            #         return None
            #     # Read first line to verify CSV format (should have 13 columns)
            #     with open(gmm_path, "r") as f:
            #         first_line = f.readline().strip()
            #         values = first_line.split(",")
            #         rospy.loginfo(f"[gmmslam] GMM CSV has {len(values)} columns")
            #         if len(values) != 13:
            #             rospy.logerr(
            #                 f"[gmmslam] Expected 13 columns (3 mean + 9 cov + 1 weight)!"
            #             )
            #             return None
            # else:
            #     rospy.logerr(f"[gmmslam] GMM file was not created!")
            #     return None

            return gmm_path
        except Exception as e:
            rospy.logerr(f"[gmmslam] failed to save GMM: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # D2D registration
    # see: https://github.com/gira3d/gmm_d2d_registration_examples/blob/082377d71d49201955b253e796ac6ca94fbfa904/python/registration_example.py
    # ------------------------------------------------------------------

    # def _register_gmm(self, source_path: str, target_path: str) -> np.ndarray:
    #     """Register source GMM to target GMM using D2D registration.

    #     Performs two-stage registration:
    #       1. isoplanar_registration - fast alignment
    #       2. anisotropic_registration - refinement

    #     Parameters
    #     ----------
    #     source_path : str
    #         Path to current frame's .gmm file
    #     target_path : str
    #         Path to previous frame's .gmm file

    #     Returns
    #     -------
    #     T : (4,4) np.ndarray or None
    #         returns the source in the target frame so T_target -> source
    #         Returns None if registration failed.
    #     """
    #     if not HAS_GMM_REGISTRATION:
    #         return None

    #     # Validate input files exist
    #     if not os.path.exists(source_path):
    #         rospy.logerr(f"[gmmslam] Source GMM file not found: {source_path}")
    #         return None
    #     if not os.path.exists(target_path):
    #         rospy.logerr(f"[gmmslam] Target GMM file not found: {target_path}")
    #         return None

    #     try:
    #         # Identity is the correct neutral initialization for SE(3).
    #         # A zero matrix can break/degenerate optimization, especially on pure rotations.
    #         T_init = np.eye(4, dtype=np.float64)

    #         # Stage 1: isoplanar registration: fast initial alignment
    #         result_iso = gmm_d2d_registration_py.isoplanar_registration(
    #             T_init, source_path, target_path
    #         )
    #         T_iso = result_iso[0]
    #         score_iso = result_iso[1]

    #         # Check for NaN scores or invalid transform from isoplanar stage.
    #         # When isoplanar fails (no planar components in the scan) fall back
    #         # to anisotropic-only, initialised from identity.
    #         if (
    #             np.isnan(score_iso)
    #             or np.isinf(score_iso)
    #             or np.any(np.isnan(T_iso))
    #             or np.any(np.isinf(T_iso))
    #         ):
    #             rospy.logwarn(
    #                 "[gmmslam] isoplanar registration failed (score: nan/inf or invalid T) "
    #                 "– falling back to anisotropic-only from identity"
    #             )
    #             T_iso = T_init  # reset to identity so anisotropic starts clean

    #         # Stage 2: anisotropic refinement
    #         result_aniso = gmm_d2d_registration_py.anisotropic_registration(
    #             T_iso, source_path, target_path
    #         )
    #         T_final = result_aniso[0]
    #         score_final = result_aniso[1]

    #         # Check for NaN scores (indicates ill-conditioned matrices)
    #         if np.isnan(score_final) or np.isinf(score_final):
    #             rospy.logwarn(f"[gmmslam] D2D aniso failed - aniso score: nan")
    #             return None

    #         # Check for NaN/Inf in final transformation
    #         if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
    #             rospy.logwarn(f"[gmmslam] D2D aniso failed - ill-conditioned matrices")
    #             return None

    #         rospy.loginfo(
    #             f"[gmmslam] D2D registration | iso score: {score_iso:.4f} | aniso score: {score_final:.4f}"
    #         )

    #         # Log transformation for debugging
    #         rospy.logdebug(
    #             f"[gmmslam] T_final translation: [{T_final[0,3]:.3f}, {T_final[1,3]:.3f}, {T_final[2,3]:.3f}]"
    #         )

    #         return T_final
    #     except Exception as e:
    #         rospy.logerr(f"[gmmslam] D2D registration failed: {e}")
    #         import traceback

    #         rospy.logerr(traceback.format_exc())
    #         return None

    def _pose3_from_matrix(self, T: np.ndarray):
        """Build gtsam.Pose3 from a 4x4 SE(3) matrix."""
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        return gtsam.Pose3(R, t)

    def _matrix_from_pose3(self, pose):
        """Convert gtsam.Pose3 to a 4x4 numpy matrix."""
        return np.array(pose.matrix(), dtype=np.float64)

    def _stamp_to_sec(self, stamp):
        return float(stamp.secs) + 1e-9 * float(stamp.nsecs)

    def _pose_msg_to_matrix(self, pose_msg) -> np.ndarray:
        """Convert geometry_msgs/Pose to a 4x4 homogeneous matrix."""
        q = np.array(
            [
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ],
            dtype=np.float64,
        )
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            raise ValueError("Invalid quaternion norm in ground-truth pose")
        q /= q_norm
        Rm = Rotation.from_quat(q).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rm
        T[0, 3] = float(pose_msg.position.x)
        T[1, 3] = float(pose_msg.position.y)
        T[2, 3] = float(pose_msg.position.z)
        return T

    def _sample_noisy_gt_relative_pose3(self, stamp):
        """Build a noisy relative GT motion to be used as a BetweenFactor."""
        _ = stamp
        if not self.use_noisy_gt_factor:
            return None
        if self._latest_gt_pose is None:
            return None

        T_curr_gt = self._pose_msg_to_matrix(self._latest_gt_pose.pose)
        if self._last_gt_T_for_factor is None:
            # Need two GT poses to form a relative constraint.
            self._last_gt_T_for_factor = T_curr_gt
            return None
        T_prev_gt = self._last_gt_T_for_factor
        T_rel_gt = np.linalg.inv(T_prev_gt) @ T_curr_gt

        rot_noise_vec = self._rng.normal(0.0, self.gt_noise_sigma_r, size=3)
        trans_noise = self._rng.normal(0.0, self.gt_noise_sigma_t, size=3)

        R_noise = Rotation.from_rotvec(rot_noise_vec).as_matrix()
        T_noisy = np.eye(4, dtype=np.float64)
        T_noisy[:3, :3] = R_noise @ T_rel_gt[:3, :3]
        T_noisy[:3, 3] = T_rel_gt[:3, 3] + trans_noise
        self._last_gt_T_for_factor = T_curr_gt
        return self._pose3_from_matrix(T_noisy)

    def _reset_new_data(self):
        """Clear per-update containers (example style)."""
        self._new_factors.resize(0)
        self._new_values.clear()
        self._new_timestamps.clear()

    def _timestamps_insert(self, key, t_sec: float):
        """Insert (key, timestamp) into FixedLagSmootherKeyTimestampMap."""
        try:
            self._new_timestamps.insert((key, float(t_sec)))
        except TypeError:
            self._new_timestamps.insert(key, float(t_sec))

    def _gtsam_backend_loop(self):
        """Dedicated GTSAM solve thread.

        Drains _gtsam_queue and runs _fixed_lag.update() + calculateEstimate()
        without ever blocking the lidar callback thread.  self.pose is updated
        here under _graph_lock so the lidar callback always reads a consistent
        (possibly one-frame-stale) estimate.
        """
        while not rospy.is_shutdown():
            try:
                task = self._gtsam_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                break
            curr_idx, factors, values, timestamps = task
            try:
                self._fixed_lag.update(factors, values, timestamps)
                estimate = self._fixed_lag.calculateEstimate()
                new_pose = self._matrix_from_pose3(estimate.atPose3(X(curr_idx)))
                with self._graph_lock:
                    self.pose = new_pose
                    self._latest_key_idx = curr_idx
                rospy.logdebug(f"[gmmslam] GTSAM backend solved at X({curr_idx})")
            except Exception as e:
                rospy.logwarn(f"[gmmslam] GTSAM backend failed at X({curr_idx}): {e}")

    def _fixed_lag_initialize(self, stamp):
        """Insert X0 prior and perform first fixed-lag update."""
        if self._fixed_lag_initialized:
            return True
        try:
            with self._graph_lock:
                key0 = X(0)
                t0 = self._stamp_to_sec(stamp)
                pose0 = self._pose3_from_matrix(self.pose)
                self._new_factors.push_back(
                    gtsam.PriorFactorPose3(key0, pose0, self._prior_noise)
                )
                self._new_values.insert(key0, pose0)
                self._timestamps_insert(key0, t0)
                self._fixed_lag.update(
                    self._new_factors, self._new_values, self._new_timestamps
                )
                self._reset_new_data()
            self._fixed_lag_initialized = True
            rospy.loginfo("[gmmslam] fixed-lag smoother initialized with prior X(0)")
            return True
        except Exception as e:
            rospy.logwarn(f"[gmmslam] failed to initialize fixed-lag smoother: {e}")
            return False

    def _fixed_lag_update_from_odom(self, prev_idx: int, curr_idx: int, stamp):
        """Update fixed-lag smoother with one immediate odometry-like edge.

        This path never waits for heavy registration; it inserts an identity
        motion edge with inflated noise, plus optional noisy-GT between factor.
        Heavy GTSAM work (update + calculateEstimate) is offloaded to the
        dedicated _gtsam_backend_loop thread via _gtsam_queue so the lidar
        callback is never blocked.

        When prev_idx < 0 (frame 0), there is no predecessor edge to add; the
        smoother has already been seeded by _fixed_lag_initialize but we still
        record the key timestamp so that registration factors can reference X(0).
        """
        if not self._fixed_lag_initialize(stamp):
            return False
        if prev_idx < 0:
            # Frame 0: prior already committed by _fixed_lag_initialize.
            # Record the timestamp so _stage_registration_factor can look it up.
            self._key_t_sec[curr_idx] = self._stamp_to_sec(stamp)
            return True

        t_sec = self._stamp_to_sec(stamp)
        key_prev = X(prev_idx)
        key_curr = X(curr_idx)

        # Read pose under lock (GTSAM thread may be writing concurrently).
        with self._graph_lock:
            predicted_T = self.pose.copy()

        # --- Stage odom factor (fast, no GTSAM solve) --------------------
        self._new_factors.push_back(
            gtsam.BetweenFactorPose3(
                key_prev, key_curr, gtsam.Pose3.Identity(), self._odom_noise_lost
            )
        )

        # --- Optional noisy-GT between factor ----------------------------
        noisy_gt_rel_pose = self._sample_noisy_gt_relative_pose3(stamp)
        if noisy_gt_rel_pose is not None:
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    key_prev, key_curr, noisy_gt_rel_pose, self._gt_factor_noise
                )
            )
            gt_rel_mat = self._matrix_from_pose3(noisy_gt_rel_pose)
            rospy.logdebug(
                f"[gmmslam] added noisy GT between X({prev_idx})->X({curr_idx}) "
                f"| dtrans={np.linalg.norm(gt_rel_mat[:3,3]):.3f} m"
                f"| drot={Rotation.from_matrix(gt_rel_mat[:3,:3]).magnitude():.3f} rad"
            )
            # Accumulate and publish noisy GT here on the lidar thread — the
            # messages are small so serialization does not block publishing.
            self._noisy_gt_pose = self._noisy_gt_pose @ gt_rel_mat
            noisy_gt_ps = pose_to_pose_stamped(
                self._noisy_gt_pose, stamp, self.odom_frame
            )
            self._noisy_gt_path.header.stamp = stamp
            self._noisy_gt_path.poses.append(noisy_gt_ps)
            # Publish single pose here (small, fast). Path is O(N) — publish
            # from the vis thread to avoid stalling the lidar callback.
            self.noisy_gt_pose_pub.publish(noisy_gt_ps)

        self._new_values.insert(key_curr, self._pose3_from_matrix(predicted_T))
        self._timestamps_insert(key_curr, t_sec)
        # Record timestamp immediately so _stage_registration_factor can use it
        # for lag-window checks on the *next* lidar callback, before the GTSAM
        # backend thread has a chance to run.
        self._key_t_sec[curr_idx] = t_sec

        # --- Snapshot factor containers and hand off to GTSAM thread -----
        # Swap in fresh containers so the next callback can stage immediately.
        factors_snap = self._new_factors
        values_snap = self._new_values
        timestamps_snap = self._new_timestamps
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
        self._new_timestamps = self._map_ctor()

        try:
            self._gtsam_queue.put_nowait(
                (curr_idx, factors_snap, values_snap, timestamps_snap)
            )
        except queue.Full:
            rospy.logwarn_throttle(
                2.0,
                f"[gmmslam] GTSAM queue full; dropping solve for X({curr_idx}) "
                f"(pose estimate will be stale until next accepted frame)",
            )
        return True

    def _stage_registration_factor(
        self,
        prev_idx: int,
        curr_idx: int,
        T_prev_to_curr: np.ndarray,
        force_loop: bool = False,
    ):
        """Push a registration between-factor into _new_factors without calling update().

        Called on the lidar thread before the odom update so that registration
        and odometry factors are committed in a single _fixed_lag.update() call.
        Uses timestamp bookkeeping to check lag-window membership — no
        calculateEstimate() call needed.
        """
        if prev_idx < 0 or curr_idx > self._latest_key_idx:
            return

        # Check both keys are still within the lag window using our own timestamp
        # records.  This is O(1) and avoids an expensive calculateEstimate() call.
        t_prev = self._key_t_sec.get(prev_idx)
        t_curr = self._key_t_sec.get(curr_idx)
        if t_prev is None or t_curr is None:
            return
        t_latest = self._key_t_sec.get(self._latest_key_idx, 0.0)
        if (t_latest - t_prev) > self.fixed_lag_s * 1.1:  # 10% margin
            rospy.logdebug(
                f"[gmmslam] skipping stale registration factor X({prev_idx})->X({curr_idx}): "
                f"key already outside lag window"
            )
            return

        key_prev = X(prev_idx)
        key_curr = X(curr_idx)
        rel_pose = self._pose3_from_matrix(T_prev_to_curr)
        self._new_factors.push_back(
            gtsam.BetweenFactorPose3(key_prev, key_curr, rel_pose, self._odom_noise)
        )
        if force_loop:
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    key_prev, key_curr, rel_pose, self._loop_closure_noise
                )
            )
        if force_loop:
            rospy.loginfo(
                f"[gmmslam] staged loop-closure factor X({prev_idx})->X({curr_idx})"
            )
        else:
            rospy.logdebug(
                f"[gmmslam] staged registration factor X({prev_idx})->X({curr_idx})"
            )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_pose_only(self, stamp):
        T = self.pose

        # --- TF: odom_frame → base_frame ---
        ts = pose_to_transform_stamped(T, stamp, self.odom_frame, self.base_frame)
        self.tf_broadcaster.sendTransform(ts)

        # --- Odometry message ---
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose_to_pose_stamped(T, stamp, self.odom_frame).pose
        self.odom_pub.publish(odom)

    def _vis_loop(self):
        """Dedicated visualization thread: drains _vis_queue and publishes all
        scan-rate products whose serialization cost grows with runtime
        (path, map cloud, GMM markers, graph nodes).
        """
        while not rospy.is_shutdown():
            try:
                item = self._vis_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            stamp, pts, frame_count = item
            try:
                self._publish_scan_products(stamp, pts, frame_count)
            except Exception as e:
                rospy.logerr_throttle(2.0, f"[gmmslam] vis thread error: {e}")

    def _publish_scan_products(self, stamp, pts: np.ndarray, frame_count: int = None):
        """Publish scan-rate products (runs on the vis thread, NOT the lidar thread)."""
        if frame_count is None:
            frame_count = self._frame_count
        with self._graph_lock:
            T = self.pose.copy()

        # --- Path ---
        ps = pose_to_pose_stamped(T, stamp, self.odom_frame)
        self.path.header = ps.header
        self.path.poses.append(ps)
        self.path_pub.publish(self.path)

        # --- Noisy GT path (O(N) — kept off the lidar callback thread) ---
        self.noisy_gt_path_pub.publish(self._noisy_gt_path)

        # --- Accumulated map cloud (throttled) ---
        if frame_count % self.map_decimate == 0:
            # Transform current scan into world frame and append.
            ones = np.ones((pts.shape[0], 1), dtype=np.float64)
            pts_h = np.hstack([pts.astype(np.float64), ones])  # (N,4)
            pts_w = (T @ pts_h.T).T[:, :3].astype(np.float32)
            self.map_pts.append(pts_w)

        # Publish the accumulated cloud at most every 2 s.  np.vstack rebuilds
        # the full O(N) array — doing this every 5 frames caused the vis thread
        # to spend seconds here as the map grows.
        now_t = self._stamp_to_sec(stamp)
        if len(self.map_pts) > 0 and (now_t - self._map_cloud_last_pub_t) >= 2.0:
            all_pts = np.vstack(self.map_pts)
            self.cloud_pub.publish(numpy_to_pc2(all_pts, stamp, self.odom_frame))
            self._map_cloud_last_pub_t = now_t

        # --- Graph node markers ---
        # Accumulate every frame but publish at most 2 Hz to avoid serializing
        # a growing array on every lidar callback (which would block all publishers).
        # latch=True ensures late RViz subscribers receive the full history.
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.odom_frame
        m.ns = "graph_nodes"
        m.id = frame_count
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(T[0, 3])
        m.pose.position.y = float(T[1, 3])
        m.pose.position.z = float(T[2, 3])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        self._graph_node_markers.markers.append(m)
        if (now_t - self._graph_nodes_last_pub_t) >= 0.5:
            self.graph_nodes_pub.publish(self._graph_node_markers)
            self._graph_nodes_last_pub_t = now_t

        # --- GMM component eigenvalue markers ---
        self._publish_gmm_markers(stamp, T)

    def _publish_gmm_markers(self, stamp, T: np.ndarray):
        """Publish a MarkerArray of ellipsoids representing the most recent local GMM.

        Each SPHERE marker is scaled along its three principal axes by 2·√λᵢ
        (1-σ extent) and oriented by the eigenvectors of the 3×3 covariance,
        giving a true covariance ellipsoid in RViz.  The mean is transformed
        into odom_frame via the current pose.
        Color encodes λ₁ (largest eigenvalue): blue = compact, red = spread out.
        """
        if not self.local_gmms:
            return
        _, gmm = self.local_gmms[-1]
        K = gmm.n_components_ if hasattr(gmm, "n_components_") else gmm.n_components

        # Covariances may be stored as (K, D, D) or flattened (K, D*D).
        D = gmm.means_.shape[1]
        covs_raw = gmm.covariances_
        if covs_raw.ndim == 2:  # flattened (K, D*D)
            covs_raw = covs_raw.reshape(K, D, D)

        R_world = T[:3, :3]

        # Pre-scan λ₁ for color normalization.
        lambda1_all = np.empty(K, dtype=np.float64)
        for k in range(K):
            eigvals = np.linalg.eigvalsh(covs_raw[k, :3, :3])
            lambda1_all[k] = max(float(eigvals[-1]), 1e-9)
        lmin, lmax = lambda1_all.min(), lambda1_all.max()
        lrange = max(lmax - lmin, 1e-12)

        ma = MarkerArray()
        for k in range(K):
            cov3 = covs_raw[k, :3, :3]
            # Symmetrize to guard against floating-point asymmetry.
            cov3 = 0.5 * (cov3 + cov3.T)
            eigvals, eigvecs = np.linalg.eigh(cov3)  # ascending; eigvecs are columns
            eigvals = np.maximum(eigvals, 1e-9)

            # Rotate eigenvectors into world frame and build a quaternion.
            # eigvecs columns form a rotation matrix R_local (cov frame → sensor frame).
            # Ensure right-handed by fixing determinant.
            R_local = eigvecs.copy()
            if np.linalg.det(R_local) < 0:
                R_local[:, 0] *= -1
            R_marker = R_world @ R_local
            q = Rotation.from_matrix(R_marker).as_quat()  # [x, y, z, w]

            mu_h = np.array([*gmm.means_[k, :3], 1.0], dtype=np.float64)
            mu_w = (T @ mu_h)[:3]

            t_norm = (lambda1_all[k] - lmin) / lrange  # 0..1

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_frame
            m.ns = "gmm_ellipsoids"
            m.id = k
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(mu_w[0])
            m.pose.position.y = float(mu_w[1])
            m.pose.position.z = float(mu_w[2])
            m.pose.orientation.x = float(q[0])
            m.pose.orientation.y = float(q[1])
            m.pose.orientation.z = float(q[2])
            m.pose.orientation.w = float(q[3])
            # Scale = 2·√λ per axis (1-σ ellipsoid diameters).
            # eigvals are in ascending order; assign to x/y/z of the marker axes.
            s = 2.0 * self.gmm_marker_sigma
            m.scale.x = max(s * np.sqrt(eigvals[0]), 0.02)
            m.scale.y = max(s * np.sqrt(eigvals[1]), 0.02)
            m.scale.z = max(s * np.sqrt(eigvals[2]), 0.02)
            m.color.r = float(t_norm)
            m.color.g = 0.2
            m.color.b = float(1.0 - t_norm)
            m.color.a = float(min(max(gmm.weights_[k] * K, 0.1), 1.0))
            m.lifetime = rospy.Duration(2.0)
            ma.markers.append(m)

        self.gmm_markers_pub.publish(ma)


def main():
    node = GMMSLAMNode()
    rospy.spin()


if __name__ == "__main__":
    main()
