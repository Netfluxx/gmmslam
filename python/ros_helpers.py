"""ROS message conversion helpers and point-cloud preprocessing."""

import struct

import numpy as np

import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation

if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm


def stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + 1e-9 * float(stamp.nsecs)


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
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
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
    """Extract XYZ points from a PointCloud2 message as an (N, 3) float32 array."""
    rows = [
        [p[0], p[1], p[2]]
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    ]
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def numpy_to_pc2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Convert an (N, 3) float32 array to a PointCloud2 message."""
    header = rospy.Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return pc2.create_cloud_xyz32(header, pts.tolist())


def numpy_to_pc2_rgb(
    pts: np.ndarray, stamp, frame_id: str, r: int, g: int, b: int
) -> PointCloud2:
    """Convert an (N,3) array to PointCloud2 with a constant RGB color."""
    header = rospy.Header()
    header.stamp = stamp
    header.frame_id = frame_id
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
    rgb_float = struct.unpack("f", struct.pack("I", rgb_uint))[0]
    points = [
        (float(p[0]), float(p[1]), float(p[2]), rgb_float)
        for p in pts.astype(np.float32)
    ]
    return pc2.create_cloud(header, fields, points)


def make_pcld_4d(pts: np.ndarray) -> np.ndarray:
    """Append L2-norm range as 4th feature column expected by SOGMM."""
    ranges = np.linalg.norm(pts, axis=1, keepdims=True)
    return np.hstack([pts, ranges]).astype(np.float64)


def preprocess(
    pts: np.ndarray, min_range: float, max_range: float, voxel_size: float
) -> np.ndarray:
    """Range filter + voxel-grid downsampling."""
    ranges = np.linalg.norm(pts, axis=1)
    mask = (ranges >= min_range) & (ranges <= max_range)
    pts = pts[mask]
    if pts.shape[0] == 0:
        return pts
    return pts


def pose_msg_to_matrix(pose_msg) -> np.ndarray:
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
