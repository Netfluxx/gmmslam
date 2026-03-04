#!/usr/bin/env python3
"""
GMM-SLAM ROS1 node
------------------
Subscribes to the depth point cloud published by the m500 drone
(/m500_1/mpa/depth/points) and runs a lightweight iterative-closest-point
(open3d) odometry pipeline as a stand-in for the full C++ GMM D2D backend.

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


# ===========================================================================
# Helpers
# ===========================================================================

def pose_to_transform_stamped(T: np.ndarray, stamp, parent: str, child: str) -> TransformStamped:
    """Convert a 4x4 homogeneous transform to a TransformStamped message."""
    ts = TransformStamped()
    ts.header.stamp    = stamp
    ts.header.frame_id = parent
    ts.child_frame_id  = child
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
    ps.header.stamp    = stamp
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
    rows = [[p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(rows, dtype=np.float32)  # shape (N, 3)


def numpy_to_pc2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Convert an (N, 3) float32 array to a PointCloud2 message."""
    header = rospy.Header()
    header.stamp    = stamp
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
    gmm_3d = GaussianMixture(n_components=K, covariance_type='full')
    
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

    # Regularize covariances to ensure positive definiteness
    # Taking a 3x3 sub-block from a 4x4 PD matrix doesn't guarantee the sub-block is PD
    reg = 1e-6
    max_condition_number = 1e3  # Maximum acceptable ratio between largest and smallest eigenvalue
    min_det = 1e-5  # Minimum acceptable determinant
    
    reg_count = 0
    bad_components = []
    ill_conditioned_components = []
    singular_components = []
    
    for k in range(K):
        # First check for NaN/Inf after projection
        if np.any(np.isnan(gmm_3d.covariances_[k])) or np.any(np.isinf(gmm_3d.covariances_[k])):
            rospy.logerr(f"[project_gmm] Component {k} has NaN/Inf after projection!")
            bad_components.append(k)
            continue
        
        # Ensure symmetry (numerical errors can break it)
        gmm_3d.covariances_[k] = 0.5 * (gmm_3d.covariances_[k] + gmm_3d.covariances_[k].T)
        
        # Check if covariance is positive definite via eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(gmm_3d.covariances_[k])
            min_eigval = eigvals.min()
            max_eigval = eigvals.max()
            
            # Check determinant (product of eigenvalues)
            det = np.prod(eigvals)
            if det <= min_det:
                singular_components.append(k)
            
            # Check condition number (ratio of largest to smallest eigenvalue)
            if max_eigval > 0 and min_eigval > 0:
                condition_number = max_eigval / min_eigval
                if condition_number > max_condition_number:
                    ill_conditioned_components.append(k)
            
            # Only add minimal regularization for numerical stability
            if min_eigval <= reg:
                gmm_3d.covariances_[k] += (abs(min_eigval) + reg) * np.eye(3)
                reg_count += 1
            else:
                gmm_3d.covariances_[k] += reg * np.eye(3)
        except np.linalg.LinAlgError as e:
            rospy.logerr(f"[project_gmm] Component {k} eigenvalue decomposition failed: {e}")
            bad_components.append(k)
            continue
    
    # Report issues with covariance matrices
    if singular_components:
        rospy.logwarn(f"[project_gmm] {len(singular_components)}/{K} components have det <= {min_det} (nearly singular)")
        rospy.logwarn(f"[project_gmm] Singular components: {singular_components[:10]}")
    
    if ill_conditioned_components:
        rospy.logwarn(f"[project_gmm] {len(ill_conditioned_components)}/{K} components are ill-conditioned (cond > {max_condition_number})")
        rospy.logwarn(f"[project_gmm] Ill-conditioned components: {ill_conditioned_components[:10]}")
    
    if bad_components:
        rospy.logerr(f"[project_gmm] {len(bad_components)} bad components found: {bad_components[:10]}")
        raise ValueError(f"GMM has {len(bad_components)} components with NaN/Inf values")
    
    # Warn if D2D registration is likely to fail
    if len(singular_components) + len(ill_conditioned_components) > K * 0.1:
        rospy.logwarn(f"[project_gmm] D2D registration likely to fail due to ill-conditioned matrices")
    
    if reg_count > 0:
        rospy.loginfo(f"[project_gmm] Regularized {reg_count}/{K} components")
    
    # Copy weights
    gmm_3d.weights_ = gmm_4d.weights_.copy()
    
    # Set other required attributes for sklearn GaussianMixture
    gmm_3d.n_components_ = K  # This is set after fit() normally
    gmm_3d.converged_ = True
    gmm_3d.n_iter_ = 0
    gmm_3d.lower_bound_ = -np.inf  # Log likelihood lower bound (not critical for saving)
    
    # Compute precision matrices
    try:
        gmm_3d.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_3d.covariances_))
    except np.linalg.LinAlgError as e:
        rospy.logerr(f"[project_gmm] Failed to compute precision matrices: {e}")
        raise
    
    return gmm_3d


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
        gmm_3d = gmm
    
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
        rospy.logerr(f"[save_gmm] GMM has {nan_count} NaN and {inf_count} Inf components!")
    
    # Use official gira3d save function
    save_gmm_official(filepath, gmm_3d)
    
    # Log file save with stats
    rospy.loginfo(f"[save_gmm] Saved {gmm_3d.n_components_} components to {os.path.basename(filepath)}")


# ===========================================================================
# Pre-processing
# ===========================================================================

def preprocess(pts: np.ndarray,
               min_range: float,
               max_range: float,
               voxel_size: float) -> np.ndarray:
    """Range filter + voxel-grid downsampling."""
    ranges = np.linalg.norm(pts, axis=1)
    mask   = (ranges >= min_range) & (ranges <= max_range)
    pts    = pts[mask]

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
        self.lidar_topic   = rospy.get_param("~lidar_topic",    "/m500_1/mpa/depth/points")
        self.sensor_frame  = rospy.get_param("~sensor_frame",   "depth_camera_1")
        self.odom_frame    = rospy.get_param("~odom_frame",     "world")
        self.base_frame    = rospy.get_param("~base_frame",     "m500_1_base_link")

        self.min_range     = rospy.get_param("~min_range",      0.1)
        self.max_range     = rospy.get_param("~max_range",      10.0)
        self.voxel_size    = rospy.get_param("~voxel_leaf_size", 0.05)
        self.min_points    = rospy.get_param("~min_points",     50)

        # SOGMM parameters
        self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.1)
        self.sogmm_compute   = rospy.get_param("~sogmm_compute",   "CPU")  # "CPU" or "GPU"

        rospy.loginfo(f"[gmmslam] lidar_topic  : {self.lidar_topic}")
        rospy.loginfo(f"[gmmslam] sensor_frame : {self.sensor_frame}")
        rospy.loginfo(f"[gmmslam] odom_frame   : {self.odom_frame}")
        rospy.loginfo(f"[gmmslam] base_frame   : {self.base_frame}")
        rospy.loginfo(f"[gmmslam] range        : [{self.min_range}, {self.max_range}] m")
        rospy.loginfo(f"[gmmslam] voxel_size   : {self.voxel_size} m")

        if HAS_SOGMM:
            rospy.loginfo(f"[gmmslam] SOGMM bandwidth : {self.sogmm_bandwidth}")
            rospy.loginfo(f"[gmmslam] SOGMM compute   : {self.sogmm_compute}")
        else:
            rospy.logwarn("[gmmslam] SOGMM unavailable – fitting will be skipped")

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.pose          = np.eye(4, dtype=np.float64)   # T_world_base (cumulative)
        self.prev_pcd      = None                          # open3d PointCloud (previous scan)
        self.path          = Path()
        self.map_pts: list = []                            # accumulated map points (list of np arrays)
        self.map_decimate  = 5                             # add 1 out of N frames to map

        self._frame_count  = 0

        # SOGMM state
        # sg         – incremental global SOGMM (builds up over all scans)
        # local_gmms – list of per-scan local GMMf4 models (for D2D registration)
        if HAS_SOGMM:
            self.sg         = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            self.local_gmms = []   # list[(stamp, GMMf4)]
        else:
            self.sg         = None
            self.local_gmms = []
        
        # D2D registration state
        self.prev_gmm_path = None   # path to previous frame's .gmm file
        self.gmm_dir = rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms")
        os.makedirs(self.gmm_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # TF broadcaster
        # ------------------------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.path_pub  = rospy.Publisher("~path",       Path,         queue_size=1)
        self.odom_pub  = rospy.Publisher("~odom",       Odometry,     queue_size=1)
        self.cloud_pub = rospy.Publisher("~map_cloud",  PointCloud2,  queue_size=1)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        rospy.Subscriber(self.lidar_topic, PointCloud2, self._pcl_callback, queue_size=1)

        # Depth image and camera info (for future use / logging)
        depth_ns = "/".join(self.lidar_topic.split("/")[:-1])   # strip "points"
        rospy.Subscriber(depth_ns + "/image_raw",  Image,      self._depth_callback,  queue_size=1)
        rospy.Subscriber(depth_ns + "/camera_info", CameraInfo, self._cam_info_callback, queue_size=1)

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
            import traceback; rospy.logerr(traceback.format_exc())

    def _pcl_callback_inner(self, msg: PointCloud2):
        if self._frame_count == 0:
            rospy.loginfo("[gmmslam] first point cloud received, processing started")

        stamp = msg.header.stamp

        # 1. Convert to numpy
        pts = pc2_to_numpy(msg)
        rospy.logdebug(f"[gmmslam] raw pts: {pts.shape[0]}")
        if pts.shape[0] == 0:
            rospy.logwarn_throttle(5.0, "[gmmslam] received empty point cloud, skipping")
            return

        # 2. Pre-process
        pts = preprocess(pts, self.min_range, self.max_range, self.voxel_size)
        if pts.shape[0] < self.min_points:
            rospy.logwarn_throttle(5.0,
                f"[gmmslam] only {pts.shape[0]} points after filtering (< {self.min_points}), skipping")
            return


        # --- SOGMM fitting ---
        curr_gmm_path = self._fit_sogmm(stamp, pts)
        
        # --- D2D registration ---
        if curr_gmm_path is not None and self.prev_gmm_path is not None:
            delta_T = self._register_gmm(curr_gmm_path, self.prev_gmm_path)
            if delta_T is not None:
                # Accumulate pose: T_world_new = T_world_prev @ T_prev_curr
                # delta_T is T_curr_prev, so invert it
                self.pose = self.pose @ np.linalg.inv(delta_T)
        
        # Store current GMM path for next iteration
        if curr_gmm_path is not None:
            self.prev_gmm_path = curr_gmm_path

        # Publish
        self._publish(stamp, pts)
        self._frame_count += 1

    # ------------------------------------------------------------------
    # SOGMM fitting
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

        pcld_4d = make_pcld_4d(pts)   # (N, 4)  [x, y, z, range]

        try:
            local_model = self.sg.fit(pcld_4d)
        except Exception as e:
            rospy.logerr(f"[gmmslam] SOGMM fit failed on frame {self._frame_count}: {e}")
            import traceback; rospy.logerr(traceback.format_exc())
            return None

        if local_model is None:
            rospy.logwarn_throttle(5.0,
                f"[gmmslam] SOGMM fit returned None on frame {self._frame_count}")
            return None

        self.local_gmms.append((stamp, local_model))
        n_local  = local_model.n_components_
        n_global = self.sg.model.n_components_ if self.sg.model is not None else 0
        rospy.loginfo(
            f"[gmmslam] frame {self._frame_count:4d} | "
            f"local GMM: {n_local:3d} components | "
            f"global GMM: {n_global:4d} components")
        
        gmm_path = os.path.join(self.gmm_dir, f"frame_{self._frame_count:06d}.gmm")
        try:
            save_gmm_to_file(local_model, gmm_path)
            
            if os.path.exists(gmm_path):
                file_size = os.path.getsize(gmm_path)
                rospy.loginfo(f"[gmmslam] GMM file size: {file_size} bytes")
                if file_size == 0:
                    rospy.logerr(f"[gmmslam] GMM file is empty!")
                    return None
                # Read first line to verify CSV format (should have 13 columns)
                with open(gmm_path, 'r') as f:
                    first_line = f.readline().strip()
                    values = first_line.split(',')
                    rospy.loginfo(f"[gmmslam] GMM CSV has {len(values)} columns")
                    if len(values) != 13:
                        rospy.logerr(f"[gmmslam] Expected 13 columns (3 mean + 9 cov + 1 weight)!")
                        return None
            else:
                rospy.logerr(f"[gmmslam] GMM file was not created!")
                return None
            
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
            Transformation T_source_target (from target to source frame)
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
            T_init = np.eye(4, dtype=np.float64)

            # Stage 1: isoplanar registration
            result_iso = gmm_d2d_registration_py.isoplanar_registration(
                T_init, source_path, target_path
            )
            T_iso = result_iso[0]
            score_iso = result_iso[1]
            
            # Check for NaN scores (indicates ill-conditioned matrices)
            if np.isnan(score_iso) or np.isinf(score_iso):
                rospy.logwarn(f"[gmmslam] D2D registration failed - ill-conditioned covariance matrices (iso score: nan)")
                return None
            
            # Check for NaN/Inf in transformation
            if np.any(np.isnan(T_iso)) or np.any(np.isinf(T_iso)):
                rospy.logwarn(f"[gmmslam] D2D registration failed - invalid transformation (ill-conditioned matrices)")
                return None
            
            # Stage 2: anisotropic refinement
            result_aniso = gmm_d2d_registration_py.anisotropic_registration(
                T_iso, source_path, target_path
            )
            T_final = result_aniso[0]
            score_final = result_aniso[1]
            
            # Check for NaN scores (indicates ill-conditioned matrices)
            if np.isnan(score_final) or np.isinf(score_final):
                rospy.logwarn(f"[gmmslam] D2D registration failed - ill-conditioned covariance matrices (aniso score: nan)")
                return None
            
            # Check for NaN/Inf in final transformation
            if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
                rospy.logwarn(f"[gmmslam] D2D registration failed - invalid transformation (ill-conditioned matrices)")
                return None
            
            rospy.loginfo(
                f"[gmmslam] D2D registration | iso score: {score_iso:.4f} | aniso score: {score_final:.4f}")
            
            # Log transformation for debugging
            rospy.logdebug(f"[gmmslam] T_final translation: [{T_final[0,3]:.3f}, {T_final[1,3]:.3f}, {T_final[2,3]:.3f}]")
            
            return T_final
        except Exception as e:
            rospy.logerr(f"[gmmslam] D2D registration failed: {e}")
            import traceback; rospy.logerr(traceback.format_exc())
            return None

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
        odom.header.stamp    = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame
        odom.pose.pose       = pose_to_pose_stamped(T, stamp, self.odom_frame).pose
        self.odom_pub.publish(odom)

        # --- Path ---
        ps = pose_to_pose_stamped(T, stamp, self.odom_frame)
        self.path.header = ps.header
        self.path.poses.append(ps)
        self.path_pub.publish(self.path)

        # --- Accumulated map cloud (throttled) ---
        if self._frame_count % self.map_decimate == 0:
            # Transform current scan into world frame and append
            ones  = np.ones((pts.shape[0], 1), dtype=np.float64)
            pts_h = np.hstack([pts.astype(np.float64), ones])   # (N,4)
            pts_w = (T @ pts_h.T).T[:, :3].astype(np.float32)
            self.map_pts.append(pts_w)

            if len(self.map_pts) > 0 : #and self.cloud_pub.get_num_connections() > 0:
                all_pts = np.vstack(self.map_pts)
                self.cloud_pub.publish(numpy_to_pc2(all_pts, stamp, self.odom_frame))


def main():
    node = GMMSLAMNode()
    rospy.spin()


if __name__ == "__main__":
    main()

