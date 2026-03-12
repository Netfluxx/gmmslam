#!/usr/bin/env python3
"""Standalone noisy ground-truth publisher node.

Publishes noisy odometry-style GT pose/path on dedicated topics, separated
from gmmslam_node to avoid heavy-node coupling.
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation


def pose_to_pose_stamped(T: np.ndarray, stamp, frame_id: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    ps.pose.orientation.x = float(q[0])
    ps.pose.orientation.y = float(q[1])
    ps.pose.orientation.z = float(q[2])
    ps.pose.orientation.w = float(q[3])
    return ps


class NoisyGTPublisherNode:
    def __init__(self):
        rospy.init_node("noisy_gt_publisher_node", anonymous=False)

        self.gt_topic = rospy.get_param("~gt_topic", "/m500_1/mavros/local_position/pose")
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.pose_topic = rospy.get_param("~pose_topic", "/gmmslam_node/noisy_gt_pose")
        self.path_topic = rospy.get_param("~path_topic", "/gmmslam_node/noisy_gt_path")
        self.gt_init_wait_s = float(rospy.get_param("~gt_init_wait_s", 3.0))
        self.gt_noise_sigma_t = float(rospy.get_param("~gt_noise_sigma_t", 0.0))
        self.gt_noise_sigma_r = float(rospy.get_param("~gt_noise_sigma_r", 0.0))
        self.gt_noise_seed = int(rospy.get_param("~gt_noise_seed", -1))

        self._rng = (
            np.random.default_rng(self.gt_noise_seed)
            if self.gt_noise_seed >= 0
            else np.random.default_rng()
        )
        self._latest_gt_pose_raw = None
        self._gt_origin_inv = None
        self._gt_init_start_time = rospy.Time.now()
        self._last_gt_odom = None
        self._noisy_pose = np.eye(4, dtype=np.float64)
        self._noisy_path = Path()
        self._noisy_path.header.frame_id = self.odom_frame

        self.noisy_pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.noisy_path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1)
        rospy.Subscriber(self.gt_topic, PoseStamped, self._gt_callback, queue_size=1)

        rospy.loginfo(
            f"[noisy_gt_pub] ready | gt={self.gt_topic} | pose={self.pose_topic} | path={self.path_topic}"
        )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.secs) + 1e-9 * float(stamp.nsecs)

    @staticmethod
    def _pose_msg_to_matrix(pose_msg) -> np.ndarray:
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
            raise ValueError("invalid quaternion norm")
        q /= q_norm
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[0, 3] = float(pose_msg.position.x)
        T[1, 3] = float(pose_msg.position.y)
        T[2, 3] = float(pose_msg.position.z)
        return T

    def _ensure_origin_initialized(self, stamp) -> bool:
        if self._gt_origin_inv is not None:
            return True
        if self._latest_gt_pose_raw is None:
            return False
        now_t = self._stamp_to_sec(stamp)
        t0 = self._stamp_to_sec(self._gt_init_start_time)
        if (now_t - t0) < self.gt_init_wait_s:
            return False
        try:
            T0 = self._pose_msg_to_matrix(self._latest_gt_pose_raw.pose)
            self._gt_origin_inv = np.linalg.inv(T0)
            self._last_gt_odom = None
            self._noisy_pose = np.eye(4, dtype=np.float64)
            self._noisy_path = Path()
            self._noisy_path.header.frame_id = self.odom_frame
            rospy.loginfo("[noisy_gt_pub] GT origin initialized and noisy odometry reset")
            return True
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[noisy_gt_pub] failed to init GT origin: {e}")
            return False

    def _gt_callback(self, msg: PoseStamped):
        self._latest_gt_pose_raw = msg
        if not self._ensure_origin_initialized(msg.header.stamp):
            return
        try:
            T_gt_webots = self._pose_msg_to_matrix(msg.pose)
            T_gt_odom = self._gt_origin_inv @ T_gt_webots
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[noisy_gt_pub] invalid GT pose: {e}")
            return

        if self._last_gt_odom is None:
            self._last_gt_odom = T_gt_odom
            ps0 = pose_to_pose_stamped(self._noisy_pose, msg.header.stamp, self.odom_frame)
            self._noisy_path.header.stamp = msg.header.stamp
            self._noisy_path.poses.append(ps0)
            self.noisy_pose_pub.publish(ps0)
            self.noisy_path_pub.publish(self._noisy_path)
            return

        T_rel = np.linalg.inv(self._last_gt_odom) @ T_gt_odom
        self._last_gt_odom = T_gt_odom

        rot_noise_vec = self._rng.normal(0.0, self.gt_noise_sigma_r, size=3)
        trans_noise = self._rng.normal(0.0, self.gt_noise_sigma_t, size=3)
        T_noisy_rel = np.eye(4, dtype=np.float64)
        T_noisy_rel[:3, :3] = Rotation.from_rotvec(rot_noise_vec).as_matrix() @ T_rel[:3, :3]
        T_noisy_rel[:3, 3] = T_rel[:3, 3] + trans_noise

        self._noisy_pose = self._noisy_pose @ T_noisy_rel
        ps = pose_to_pose_stamped(self._noisy_pose, msg.header.stamp, self.odom_frame)
        self._noisy_path.header.stamp = msg.header.stamp
        self._noisy_path.poses.append(ps)
        self.noisy_pose_pub.publish(ps)
        self.noisy_path_pub.publish(self._noisy_path)


def main():
    NoisyGTPublisherNode()
    rospy.spin()


if __name__ == "__main__":
    main()
