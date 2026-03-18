"""Visualization: RViz publishing (path, clouds, GMM markers, graph nodes)."""

import queue

import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation

from ros_helpers import (
    pose_to_transform_stamped,
    pose_to_pose_stamped,
    numpy_to_pc2_rgb,
    stamp_to_sec,
)
from gmm_utils import precompute_gmm_local_data

if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm


class Visualizer:
    """Handles all RViz-facing publishing on a dedicated thread."""

    def __init__(
        self,
        smoother,
        registration,
        global_graph,
        odom_frame: str,
        base_frame: str,
        gmm_marker_sigma: float,
        map_decimate: int,
        global_map_publish_period_s: float,
        global_gmm_publish_period_s: float,
        publishers: dict,
    ):
        self.smoother = smoother
        self.registration = registration
        self.global_graph = global_graph
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.gmm_marker_sigma = gmm_marker_sigma
        self.map_decimate = map_decimate
        self.global_map_publish_period_s = global_map_publish_period_s
        self.global_gmm_publish_period_s = global_gmm_publish_period_s

        # Publishers (created by the node)
        self.path_pub = publishers["path"]
        self.odom_pub = publishers["odom"]
        self.cloud_pub = publishers["cloud"]
        self.latest_frame_cloud_pub = publishers["latest_frame_cloud"]
        self.gmm_markers_pub = publishers["gmm_markers"]
        self.gmm_global_markers_pub = publishers["gmm_global_markers"]
        self.global_graph_markers_pub = publishers["global_graph_markers"]
        self.graph_nodes_pub = publishers["graph_nodes"]
        self.tf_broadcaster = publishers["tf_broadcaster"]

        # State
        self.path = Path()
        self.map_pts: list = []
        self._map_cloud_last_pub_t = 0.0
        self._global_gmm_markers_last_pub_t = 0.0
        self._last_global_gmm_processed_idx = -1
        self._global_gmm_cache = []
        self._graph_node_markers = MarkerArray()
        self._graph_nodes_last_pub_t = 0.0
        self._global_graph_markers_last_pub_t = 0.0

        # Drop-queue for the vis thread (maxsize=1)
        self._vis_queue: queue.Queue = queue.Queue(maxsize=1)

    # ------------------------------------------------------------------
    # Pose-only publish
    # ------------------------------------------------------------------

    def publish_pose_only(self, T, stamp):
        ts = pose_to_transform_stamped(T, stamp, self.odom_frame, self.base_frame)
        self.tf_broadcaster.sendTransform(ts)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose_to_pose_stamped(T, stamp, self.odom_frame).pose
        self.odom_pub.publish(odom)

    # ------------------------------------------------------------------
    # Vis thread
    # ------------------------------------------------------------------

    def enqueue_frame(self, stamp, pts, frame_count, capture_pose):
        try:
            self._vis_queue.put_nowait(
                (stamp, pts.copy(), frame_count, capture_pose.copy())
            )
        except queue.Full:
            pass

    def vis_loop(self):
        """Dedicated visualization thread target."""
        while not rospy.is_shutdown():
            try:
                item = self._vis_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            stamp, pts, frame_count, capture_pose = item
            try:
                self._publish_scan_products(stamp, pts, frame_count, capture_pose)
            except Exception as e:
                rospy.logerr_throttle(2.0, f"[vis] thread error: {e}")

    def _publish_scan_products(self, stamp, pts, frame_count, capture_pose):
        T = capture_pose

        # Path
        ps = pose_to_pose_stamped(T, stamp, self.odom_frame)
        self.path.header = ps.header
        self.path.poses.append(ps)
        self.path_pub.publish(self.path)

        # Transform current scan into world frame
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts.astype(np.float64), ones])
        pts_w = (T @ pts_h.T).T[:, :3].astype(np.float32)

        # Latest frame cloud (red)
        self.latest_frame_cloud_pub.publish(
            numpy_to_pc2_rgb(pts_w, stamp, self.odom_frame, r=255, g=0, b=0)
        )

        # Accumulated global map cloud (gray, slow refresh)
        if frame_count % self.map_decimate == 0:
            self.map_pts.append(pts_w)

        now_t = stamp_to_sec(stamp)
        if (
            len(self.map_pts) > 0
            and (now_t - self._map_cloud_last_pub_t) >= self.global_map_publish_period_s
        ):

            all_pts = np.vstack(self.map_pts)
            self.cloud_pub.publish(
                numpy_to_pc2_rgb(all_pts, stamp, self.odom_frame, r=140, g=140, b=140)
            )
            self._map_cloud_last_pub_t = now_t

        # Graph node marker
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
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color.r = 1.0
        m.color.a = 1.0
        self._graph_node_markers.markers.append(m)
        if (now_t - self._graph_nodes_last_pub_t) >= 0.5:
            self.graph_nodes_pub.publish(self._graph_node_markers)
            self._graph_nodes_last_pub_t = now_t

        # Global submap graph markers
        self._publish_global_graph_markers(stamp, now_t)

        # GMM component markers
        self._publish_gmm_markers(stamp, T)

    # ------------------------------------------------------------------
    # Global graph markers
    # ------------------------------------------------------------------

    def _publish_global_graph_markers(self, stamp, now_t: float):
        gg = self.global_graph
        if gg is None or not gg.enable:
            return
        if (now_t - self._global_graph_markers_last_pub_t) < 0.5:
            return

        with gg.lock:
            submap_ids = list(gg.submap_ids)
            submap_poses = {sid: gg.submap_pose_by_idx.get(sid) for sid in submap_ids}
            loop_edges = list(gg.loop_edges_added)

        ma = MarkerArray()
        for sid in submap_ids:
            T_sid = submap_poses.get(sid)
            if T_sid is None:
                continue
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_frame
            m.ns = "global_submaps"
            m.id = int(sid)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(T_sid[0, 3])
            m.pose.position.y = float(T_sid[1, 3])
            m.pose.position.z = float(T_sid[2, 3])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color.g = 1.0
            m.color.a = 0.95
            ma.markers.append(m)

        loop_marker = Marker()
        loop_marker.header.stamp = stamp
        loop_marker.header.frame_id = self.odom_frame
        loop_marker.ns = "global_loops"
        loop_marker.id = 0
        loop_marker.type = Marker.LINE_LIST
        loop_marker.action = Marker.ADD
        loop_marker.pose.orientation.w = 1.0
        loop_marker.scale.x = 0.05
        loop_marker.color.r = 1.0
        loop_marker.color.b = 1.0
        loop_marker.color.a = 0.95
        for sid_a, sid_b in loop_edges:
            Ta = submap_poses.get(sid_a)
            Tb = submap_poses.get(sid_b)
            if Ta is None or Tb is None:
                continue
            p0, p1 = Point(), Point()
            p0.x = float(Ta[0, 3])
            p0.y = float(Ta[1, 3])
            p0.z = float(Ta[2, 3])
            p1.x = float(Tb[0, 3])
            p1.y = float(Tb[1, 3])
            p1.z = float(Tb[2, 3])
            loop_marker.points.extend([p0, p1])
        if loop_marker.points:
            ma.markers.append(loop_marker)

        self.global_graph_markers_pub.publish(ma)
        self._global_graph_markers_last_pub_t = now_t

    # ------------------------------------------------------------------
    # GMM component markers
    # ------------------------------------------------------------------

    def _make_markers_from_cache(
        self, components, T_world, stamp, ns, color_rgb, alpha,
        id_start=0, lifetime_s=0.0,
    ):
        """Build RViz markers from pre-computed local data + a world pose."""
        ma = MarkerArray()
        R_world = T_world[:3, :3]
        s_factor = 2.0 * self.gmm_marker_sigma
        dur = rospy.Duration(lifetime_s) if lifetime_s > 0.0 else rospy.Duration(0.0)

        for k, (scales, R_local, mu_local) in enumerate(components):
            q = Rotation.from_matrix(R_world @ R_local).as_quat()
            mu_h = np.array([mu_local[0], mu_local[1], mu_local[2], 1.0],
                            dtype=np.float64)
            mu_w = (T_world @ mu_h)[:3]

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.odom_frame
            m.ns = ns
            m.id = int(id_start + k)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(mu_w[0])
            m.pose.position.y = float(mu_w[1])
            m.pose.position.z = float(mu_w[2])
            m.pose.orientation.x = float(q[0])
            m.pose.orientation.y = float(q[1])
            m.pose.orientation.z = float(q[2])
            m.pose.orientation.w = float(q[3])
            m.scale.x = max(s_factor * scales[0], 0.02)
            m.scale.y = max(s_factor * scales[1], 0.02)
            m.scale.z = max(s_factor * scales[2], 0.02)
            m.color.r = float(color_rgb[0])
            m.color.g = float(color_rgb[1])
            m.color.b = float(color_rgb[2])
            m.color.a = float(alpha)
            m.lifetime = dur
            ma.markers.append(m)
        return ma

    def _publish_gmm_markers(self, stamp, T: np.ndarray):
        reg = self.registration
        gg = self.global_graph

        # --- Latest-frame ellipsoids (white, use smoother's best pose) ---
        with reg.lock:
            latest_idx = int(reg.latest_gmm_idx)
            latest_gmm = reg.latest_gmm_model
        if latest_gmm is not None:
            with self.smoother.graph_lock:
                T_latest = self.smoother.pose_by_idx.get(latest_idx, T).copy()
            local_data = precompute_gmm_local_data(latest_gmm)
            latest_ma = self._make_markers_from_cache(
                local_data, T_latest, stamp,
                ns="gmm_latest", color_rgb=(1.0, 1.0, 1.0),
                alpha=0.9, id_start=0, lifetime_s=0.6,
            )
            self.gmm_markers_pub.publish(latest_ma)

        # --- Global GMM map: render per-submap with distinct colors ---
        now_t = stamp_to_sec(stamp)
        if (
            now_t - self._global_gmm_markers_last_pub_t
        ) < self.global_gmm_publish_period_s:
            return

        global_ma = MarkerArray()
        id_counter = 0
        n_submaps_rendered = 0

        if gg is not None and gg.enable:
            with gg.lock:
                submap_ids = list(gg.submap_ids)
                submap_poses = {
                    sid: gg.submap_pose_by_idx.get(sid)
                    for sid in submap_ids
                }
                submap_components = dict(gg.submap_gmm_components)
                frozen_submap_poses = dict(gg.submap_frozen_pose_by_idx)

            for sid in submap_ids:
                components = submap_components.get(sid)
                if components is None:
                    continue
                # Finalized submaps are rendered at a frozen pose captured
                # at finalization time to avoid post-hoc skew from later graph updates.
                T_sid = frozen_submap_poses.get(sid, submap_poses.get(sid))
                if T_sid is None:
                    continue
                color = gg.submap_color(sid)
                ma_i = self._make_markers_from_cache(
                    components, T_sid, stamp,
                    ns="gmm_global", color_rgb=color,
                    alpha=0.45, id_start=id_counter,
                )
                global_ma.markers.extend(ma_i.markers)
                id_counter += len(ma_i.markers)
                n_submaps_rendered += 1

        # Keyframes not yet assigned to a finalized submap: render in gray
        # using per-keyframe cache
        with reg.lock:
            new_idx = sorted(
                k for k in reg.local_gmms_by_idx
                if k > self._last_global_gmm_processed_idx
            )
            for idx in new_idx:
                entry = reg.local_gmms_by_idx[idx]
                gmm_i = entry[1]
                components = precompute_gmm_local_data(gmm_i)
                self._global_gmm_cache.append((idx, components))
                self._last_global_gmm_processed_idx = max(
                    self._last_global_gmm_processed_idx, idx
                )

        # Determine which keyframe indices belong to finalized submaps
        finalized_keys = set()
        if gg is not None and gg.enable:
            with gg.lock:
                for sid in gg.submap_ids:
                    if sid in gg.submap_gmm_components:
                        for ki in gg.submap_keyframes.get(sid, []):
                            finalized_keys.add(ki)

        with self.smoother.graph_lock:
            pose_snap = {k: v.copy() for k, v in self.smoother.pose_by_idx.items()}

        for idx, components in self._global_gmm_cache:
            if idx in finalized_keys:
                continue
            T_i = pose_snap.get(idx)
            if T_i is None:
                continue
            # Use the current open submap's color if available
            open_sid = (gg.key_to_submap.get(idx) if gg is not None else None)
            if open_sid is not None:
                color = gg.submap_color(open_sid)
            else:
                color = (0.5, 0.5, 0.5)
            ma_i = self._make_markers_from_cache(
                components, T_i, stamp,
                ns="gmm_global", color_rgb=color,
                alpha=0.35, id_start=id_counter,
            )
            global_ma.markers.extend(ma_i.markers)
            id_counter += len(ma_i.markers)

        rospy.loginfo_throttle(
            10.0,
            f"[vis] global GMM map: {len(global_ma.markers)} markers, "
            f"{n_submaps_rendered} finalized submaps",
        )
        self.gmm_global_markers_pub.publish(global_ma)
        self._global_gmm_markers_last_pub_t = now_t
