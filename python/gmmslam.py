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
    """Add site-packages from a venv into sys.path (idempotent)."""
    pattern = os.path.join(venv_root, "lib", "python3*", "site-packages")
    for sp in glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)

_GIRA_WS = os.environ.get("GIRA_WS", "/root/gira_ws")
_RECONSTRUCTION_VENV = os.path.join(_GIRA_WS, "gira3d-reconstruction", ".venv")
_inject_venv(_RECONSTRUCTION_VENV)

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

        # ICP / registration parameters (open3d)
        self.icp_max_dist  = rospy.get_param("~icp_max_dist",   0.3)   # [m]
        self.icp_max_iter  = rospy.get_param("~icp_max_iter",   50)

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
        rospy.Subscriber(self.lidar_topic, PointCloud2, self._cloud_cb, queue_size=1)

        # Depth image and camera info (for future use / logging)
        depth_ns = "/".join(self.lidar_topic.split("/")[:-1])   # strip "points"
        rospy.Subscriber(depth_ns + "/image_raw",  Image,      self._depth_image_cb,  queue_size=1)
        rospy.Subscriber(depth_ns + "/camera_info", CameraInfo, self._camera_info_cb, queue_size=1)

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

    def _depth_image_cb(self, msg: Image):
        """Store the latest 32FC1 depth image (reserved for future processing)."""
        self._latest_depth_image = msg

    def _camera_info_cb(self, msg: CameraInfo):
        """Store camera intrinsics (reserved for future use)."""
        self._latest_camera_info = msg

    def _cloud_cb(self, msg: PointCloud2):
        try:
            self._cloud_cb_inner(msg)
        except Exception as e:
            rospy.logerr(f"[gmmslam] exception in cloud callback: {e}")
            import traceback; rospy.logerr(traceback.format_exc())

    def _cloud_cb_inner(self, msg: PointCloud2):
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


        ############## GMM D2D REGISTRATION HERE ##############
        # Register against previous scan with GMM D2D
        # delta_T = np.eye(4, dtype=np.float64)

        # Accumulate pose
        # self.pose = self.pose @ np.linalg.inv(delta_T)

        # Store current scan as previous

        #     self.prev_pcd = pcd

        ############## GMM D2D REGISTRATION END ##############

        # --- SOGMM fitting ---
        self._fit_sogmm(stamp, pts)

        # Publish
        self._publish(stamp, pts)
        self._frame_count += 1

    # ------------------------------------------------------------------
    # SOGMM fitting
    # ------------------------------------------------------------------

    def _fit_sogmm(self, stamp, pts: np.ndarray):
        """Fit a SOGMM to the current preprocessed scan.

        Builds the 4D point cloud [x, y, z, range], runs SOGMM.fit() which:
          1. Uses MeanShift (bandwidth) to estimate the number of components.
          2. Initialises responsibilities with K-Means++.
          3. Runs EM to obtain the local GMMf4 model.
          4. Merges the local model into the global incremental model (sg.model).

        The local model is stored in self.local_gmms for later D2D registration.
        """
        rospy.logdebug(f"[gmmslam] _fit_sogmm called, HAS_SOGMM={HAS_SOGMM}")

        if not HAS_SOGMM:
            return

        pcld_4d = make_pcld_4d(pts)   # (N, 4)  [x, y, z, range]

        try:
            local_model = self.sg.fit(pcld_4d)
        except Exception as e:
            rospy.logerr(f"[gmmslam] SOGMM fit failed on frame {self._frame_count}: {e}")
            import traceback; rospy.logerr(traceback.format_exc())
            return

        if local_model is None:
            rospy.logwarn_throttle(5.0,
                f"[gmmslam] SOGMM fit returned None on frame {self._frame_count}")
            return

        self.local_gmms.append((stamp, local_model))
        n_local  = local_model.n_components_
        n_global = self.sg.model.n_components_ if self.sg.model is not None else 0
        rospy.loginfo(
            f"[gmmslam] frame {self._frame_count:4d} | "
            f"local GMM: {n_local:3d} components | "
            f"global GMM: {n_global:4d} components")

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

