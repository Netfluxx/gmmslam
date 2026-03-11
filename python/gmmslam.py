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

Broadcast TF:
  odom_frame → base_frame   current pose
"""

import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
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


def filter_well_conditioned_gmm(
    gmm_3d, max_condition_number: float = 1e8, min_det: float = 1e-12, reg: float = 1e-6
):
    """Keep only components with numerically well-conditioned 3x3 covariances.

    Thresholds are intentionally loose so that planar (high-condition-number)
    components are kept: isoplanar_registration specifically needs them.
    Only truly degenerate components (NaN/Inf, non-positive eigenvalues) are
    dropped.
    """
    from sklearn.mixture import GaussianMixture

    if hasattr(gmm_3d, "means_"):
        K = int(gmm_3d.means_.shape[0])
    else:
        K = int(getattr(gmm_3d, "n_components_", getattr(gmm_3d, "n_components")))
    keep = []
    dropped_bad = []
    dropped_singular = []
    dropped_ill = []

    for k in range(K):
        cov = gmm_3d.covariances_[k].copy()
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            dropped_bad.append(k)
            continue

        cov = 0.5 * (cov + cov.T)
        try:
            eigvals = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            dropped_bad.append(k)
            continue

        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            dropped_bad.append(k)
            continue

        min_eigval = eigvals.min()
        max_eigval = eigvals.max()
        det = float(np.prod(eigvals))
        if min_eigval <= 0.0 or det <= min_det:
            dropped_singular.append(k)
            continue

        condition_number = max_eigval / min_eigval
        if condition_number > max_condition_number:
            dropped_ill.append(k)
            continue

        keep.append((k, cov))

    if dropped_bad:
        rospy.logwarn(
            f"[filter_gmm] dropped {len(dropped_bad)}/{K} components (NaN/Inf or eigendecomposition failure): {dropped_bad[:10]}"
        )
    if dropped_singular:
        rospy.logwarn(
            f"[filter_gmm] dropped {len(dropped_singular)}/{K} singular/near-singular components (det <= {min_det}): {dropped_singular[:10]}"
        )
    if dropped_ill:
        rospy.logwarn(
            f"[filter_gmm] dropped {len(dropped_ill)}/{K} ill-conditioned components (cond > {max_condition_number}): {dropped_ill[:10]}"
        )

    if not keep:
        raise ValueError("No well-conditioned covariance matrices left after filtering")

    keep_idx = [i for i, _ in keep]
    covs = np.stack([cov + reg * np.eye(3) for _, cov in keep], axis=0)
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
    filtered.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))

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

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.lidar_topic = rospy.get_param("~lidar_topic", "/m500_1/mpa/depth/points")
        self.sensor_frame = rospy.get_param("~sensor_frame", "depth_camera_1")
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "m500_1_base_link")

        self.min_range = rospy.get_param("~min_range", 0.1)
        self.max_range = rospy.get_param("~max_range", 10.0)
        self.voxel_size = rospy.get_param("~voxel_leaf_size", 0.05)
        self.min_points = rospy.get_param("~min_points", 50)

        # SOGMM parameters
        self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.05)
        self.sogmm_compute = rospy.get_param("~sogmm_compute", "CPU")  # "CPU" or "GPU"
        self.sogmm_max_points = rospy.get_param(
            "~sogmm_max_points", 1000
        )  # subsample before fitting (0 = no cap)
        self.sogmm_n_components = rospy.get_param(
            "~sogmm_n_components", 200
        )  # fixed component count for now
        self.plot_first_frame = rospy.get_param("~plot_first_frame", False)
        # Backend parameters (fixed-lag smoothing only)
        self.use_fixed_lag_smoother = rospy.get_param("~use_fixed_lag_smoother", True)
        self.fixed_lag_s = rospy.get_param("~fixed_lag_s", 3.0)
        self.odom_noise_sigma_t = rospy.get_param("~odom_noise_sigma_t", 0.06)
        self.odom_noise_sigma_r = rospy.get_param("~odom_noise_sigma_r", 0.06)

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
        rospy.loginfo(
            f"[gmmslam] odom noise (t/r) : {self.odom_noise_sigma_t} / {self.odom_noise_sigma_r}"
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

        self._frame_count = 0

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
        self._fixed_lag = None
        self._new_factors = None
        self._new_values = None
        self._new_timestamps = None
        self._fixed_lag_initialized = False
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
        self._fixed_lag = gtsam_unstable.IncrementalFixedLagSmoother(self.fixed_lag_s)
        map_ctor = getattr(gtsam_unstable, "FixedLagSmootherKeyTimestampMap", None)
        if map_ctor is None:
            raise RuntimeError(
                "Fixed-lag backend requested but gtsam_unstable.FixedLagSmootherKeyTimestampMap is unavailable"
            )
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
        self._new_timestamps = map_ctor()
        rospy.loginfo(
            f"[gmmslam] backend initialised: IncrementalFixedLagSmoother (lag={self.fixed_lag_s:.2f}s)"
        )

        # ------------------------------------------------------------------
        # TF broadcaster
        # ------------------------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.path_pub = rospy.Publisher("~path", Path, queue_size=1)
        self.odom_pub = rospy.Publisher("~odom", Odometry, queue_size=1)
        self.cloud_pub = rospy.Publisher("~map_cloud", PointCloud2, queue_size=1)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        rospy.Subscriber(
            self.lidar_topic, PointCloud2, self._pcl_callback, queue_size=1
        )

        # Depth image and camera info (for future use / logging)
        depth_ns = "/".join(self.lidar_topic.split("/")[:-1])  # strip "points"
        rospy.Subscriber(
            depth_ns + "/image_raw", Image, self._depth_callback, queue_size=1
        )
        rospy.Subscriber(
            depth_ns + "/camera_info", CameraInfo, self._cam_info_callback, queue_size=1
        )

        # With use_sim_time=true rospy holds callbacks until the clock ticks.
        # Block here until time is valid so we don't silently drop messages.
        if rospy.get_param("/use_sim_time", False):
            rospy.loginfo("[gmmslam] use_sim_time=true detected, waiting for /clock …")
            while not rospy.is_shutdown() and rospy.Time.now().is_zero():
                rospy.sleep(0.1)
            rospy.loginfo("[gmmslam] clock started, ready")

        rospy.loginfo("[gmmslam] node ready, waiting for point clouds …")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _depth_callback(self, msg: Image):
        """Store the latest 32FC1 depth image (reserved for future processing)."""
        self._latest_depth_image = msg

    def _cam_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics (reserved for future use)."""
        self._latest_camera_info = msg

    def _pcl_callback(self, msg: PointCloud2):
        try:
            self._pcl_callback_inner(msg)
        except Exception as e:
            rospy.logerr(f"[gmmslam] exception in cloud callback: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())

    def _pcl_callback_inner(self, msg: PointCloud2):
        if self._frame_count == 0:
            rospy.loginfo("[gmmslam] first point cloud received, processing started")

        stamp = msg.header.stamp

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
        if self._frame_count == 0:
            self._fixed_lag_initialize(stamp)

        # --- SOGMM fitting ---
        curr_gmm_path = self._fit_sogmm(stamp, pts)

        # --- D2D registration + fixed-lag update ---
        if self._frame_count > 0:
            T_prev_to_curr = None
            if curr_gmm_path is not None and self.prev_gmm_path is not None:
                T_prev_to_curr = self._register_gmm(curr_gmm_path, self.prev_gmm_path)

            if not self._fixed_lag_update_from_odom(
                prev_idx=self._frame_count - 1,
                curr_idx=self._frame_count,
                T_prev_to_curr=T_prev_to_curr,
                stamp=stamp,
            ):
                rospy.logwarn_throttle(
                    2.0,
                    "[gmmslam] fixed-lag update failed; keeping previous published pose",
                )

        # Store current GMM path for next iteration
        if curr_gmm_path is not None:
            self.prev_gmm_path = curr_gmm_path

        # Publish
        self._publish(stamp, pts)
        self._frame_count += 1

    # ------------------------------------------------------------------
    # SOGMM fitting
    # https://github.com/gira3d/sogmm_py/blob/697cbeaf10c60fa80445c25e05dd9d720a296869/src/sogmm_py/sogmm.py
    # https://github.com/gira3d/gmm_d2d_registration_examples/blob/082377d71d49201955b253e796ac6ca94fbfa904/python/create_and_save_gmm_example.py
    # Sklean GMM Fitting: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
    # ------------------------------------------------------------------

    def _fit_sogmm(self, stamp, pts: np.ndarray) -> str:
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
        rospy.logdebug(f"[gmmslam] _fit_sogmm called, HAS_SOGMM={HAS_SOGMM}")

        if not HAS_SOGMM:
            return None

        pcld_4d = make_pcld_4d(pts)  # (N, 4)  [x, y, z, range]

        # Subsample to cap fitting time (MeanShift + EM scale badly with N).
        # sogmm_max_points=0 disables the cap.
        if self.sogmm_max_points > 0 and pcld_4d.shape[0] > self.sogmm_max_points:
            idx = np.random.choice(
                pcld_4d.shape[0], self.sogmm_max_points, replace=False
            )
            pcld_4d = pcld_4d[idx]

        try:
            sg_local = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            if self.sogmm_n_components > 0:
                local_model = sg_local.gmm_fit(pcld_4d, self.sogmm_n_components)
            else:
                local_model = sg_local.fit(pcld_4d)
        except Exception as e:
            rospy.logerr(
                f"[gmmslam] SOGMM fit failed on frame {self._frame_count}: {e}"
            )
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

        if local_model is None:
            rospy.logwarn_throttle(
                5.0, f"[gmmslam] SOGMM fit returned None on frame {self._frame_count}"
            )
            return None

        self.local_gmms.append((stamp, local_model))
        n_local = local_model.n_components_
        rospy.loginfo(
            f"[gmmslam] frame {self._frame_count:4d} | "
            f"local GMM: {n_local:3d} components (fresh per-frame fit)"
        )

        if self.plot_first_frame and self._frame_count == 0:
            rospy.loginfo("[gmmslam] plot_first_frame=True — opening GMM visualisation")
            plot_gmm_3d(
                local_model,
                sigma=1.0,
                title=f"First frame local GMM (frame 0, K={n_local})",
            )

        gmm_path = os.path.join(self.gmm_dir, f"frame_{self._frame_count:06d}.gmm")
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

    def _register_gmm(self, source_path: str, target_path: str) -> np.ndarray:
        """Register source GMM to target GMM using D2D registration.

        Performs two-stage registration:
          1. isoplanar_registration - fast alignment
          2. anisotropic_registration - refinement

        Parameters
        ----------
        source_path : str
            Path to current frame's .gmm file
        target_path : str
            Path to previous frame's .gmm file

        Returns
        -------
        T : (4,4) np.ndarray or None
            returns the source in the target frame so T_target -> source
            Returns None if registration failed.
        """
        if not HAS_GMM_REGISTRATION:
            return None

        # Validate input files exist
        if not os.path.exists(source_path):
            rospy.logerr(f"[gmmslam] Source GMM file not found: {source_path}")
            return None
        if not os.path.exists(target_path):
            rospy.logerr(f"[gmmslam] Target GMM file not found: {target_path}")
            return None

        try:
            T_init = np.zeros((4, 4), dtype=np.float64)

            # Stage 1: isoplanar registration: fast initial alignment
            result_iso = gmm_d2d_registration_py.isoplanar_registration(
                T_init, source_path, target_path
            )
            T_iso = result_iso[0]
            score_iso = result_iso[1]

            # Check for NaN scores or invalid transform from isoplanar stage.
            # When isoplanar fails (no planar components in the scan) fall back
            # to anisotropic-only, initialised from identity.
            if (
                np.isnan(score_iso)
                or np.isinf(score_iso)
                or np.any(np.isnan(T_iso))
                or np.any(np.isinf(T_iso))
            ):
                rospy.logwarn(
                    "[gmmslam] isoplanar registration failed (score: nan/inf or invalid T) "
                    "– falling back to anisotropic-only from identity"
                )
                T_iso = T_init  # reset to identity so anisotropic starts clean

            # Stage 2: anisotropic refinement
            result_aniso = gmm_d2d_registration_py.anisotropic_registration(
                T_iso, source_path, target_path
            )
            T_final = result_aniso[0]
            score_final = result_aniso[1]

            # Check for NaN scores (indicates ill-conditioned matrices)
            if np.isnan(score_final) or np.isinf(score_final):
                rospy.logwarn(f"[gmmslam] D2D aniso failed - aniso score: nan")
                return None

            # Check for NaN/Inf in final transformation
            if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
                rospy.logwarn(f"[gmmslam] D2D aniso failed - ill-conditioned matrices")
                return None

            rospy.loginfo(
                f"[gmmslam] D2D registration | iso score: {score_iso:.4f} | aniso score: {score_final:.4f}"
            )

            # Log transformation for debugging
            rospy.logdebug(
                f"[gmmslam] T_final translation: [{T_final[0,3]:.3f}, {T_final[1,3]:.3f}, {T_final[2,3]:.3f}]"
            )

            return T_final
        except Exception as e:
            rospy.logerr(f"[gmmslam] D2D registration failed: {e}")
            import traceback

            rospy.logerr(traceback.format_exc())
            return None

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

    def _fixed_lag_initialize(self, stamp):
        """Insert X0 prior and perform first fixed-lag update."""
        if self._fixed_lag_initialized:
            return True
        try:
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

    def _fixed_lag_update_from_odom(
        self, prev_idx: int, curr_idx: int, T_prev_to_curr, stamp
    ):
        """Update fixed-lag smoother using one odometry edge.

        If T_prev_to_curr is None (registration failed), inserts the key
        with identity motion and heavily inflated noise so the factor
        graph chain stays connected.
        """
        if prev_idx < 0:
            return False
        if not self._fixed_lag_initialize(stamp):
            return False
        try:
            t_sec = self._stamp_to_sec(stamp)
            key_prev = X(prev_idx)
            key_curr = X(curr_idx)

            if T_prev_to_curr is not None:
                rel_pose = self._pose3_from_matrix(T_prev_to_curr)
                noise = self._odom_noise
                predicted_T = self.pose @ T_prev_to_curr
            else:
                rel_pose = gtsam.Pose3.Identity()
                noise = self._odom_noise_lost
                predicted_T = self.pose

            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(key_prev, key_curr, rel_pose, noise)
            )
            self._new_values.insert(key_curr, self._pose3_from_matrix(predicted_T))
            self._timestamps_insert(key_curr, t_sec)

            self._fixed_lag.update(
                self._new_factors, self._new_values, self._new_timestamps
            )
            self._reset_new_data()

            estimate = self._fixed_lag.calculateEstimate()
            self.pose = self._matrix_from_pose3(estimate.atPose3(key_curr))
            if T_prev_to_curr is not None:
                rospy.logdebug(f"[gmmslam] fixed-lag updated at X({curr_idx})")
            else:
                rospy.logdebug(
                    f"[gmmslam] fixed-lag updated at X({curr_idx}) (identity, lost)"
                )
            return True
        except Exception as e:
            rospy.logwarn(f"[gmmslam] fixed-lag update failed at X({curr_idx}): {e}")
            self._reset_new_data()
            return False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, stamp, pts: np.ndarray):
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
        print(f"current odom position: {T[0,3]}, {T[1,3]}, {T[2,3]}")
        self.odom_pub.publish(odom)

        # --- Path ---
        ps = pose_to_pose_stamped(T, stamp, self.odom_frame)
        self.path.header = ps.header
        self.path.poses.append(ps)
        self.path_pub.publish(self.path)

        # --- Accumulated map cloud (throttled) ---
        if self._frame_count % self.map_decimate == 0:
            # Transform current scan into world frame and append
            ones = np.ones((pts.shape[0], 1), dtype=np.float64)
            pts_h = np.hstack([pts.astype(np.float64), ones])  # (N,4)
            pts_w = (T @ pts_h.T).T[:, :3].astype(np.float32)
            self.map_pts.append(pts_w)

            if len(self.map_pts) > 0:  # and self.cloud_pub.get_num_connections() > 0:
                all_pts = np.vstack(self.map_pts)
                self.cloud_pub.publish(numpy_to_pc2(all_pts, stamp, self.odom_frame))


def main():
    node = GMMSLAMNode()
    rospy.spin()


if __name__ == "__main__":
    main()
