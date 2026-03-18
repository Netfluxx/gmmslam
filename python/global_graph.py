"""Long-term global submap pose graph backed by GTSAM iSAM2.

Each submap aggregates N keyframe GMMs via concatenation + weight
renormalization.  Overlapping submaps trigger D2D registration whose
results are added as BetweenFactors between submap nodes.
"""

import json
import os
import threading

import numpy as np
import rospy
from nav_msgs.msg import Path

from imports_setup import gtsam, X
from ros_helpers import pose_to_pose_stamped, stamp_to_sec
from gmm_utils import (
    merge_gmms_concatenate,
    precompute_gmm_local_data,
    save_gmm_to_file,
    SUBMAP_COLORS,
)


class GlobalPoseGraph:
    """Manages the global submap graph for long-term consistency."""

    def __init__(
        self,
        odom_frame: str,
        submap_between_sigma_t: float,
        submap_between_sigma_r: float,
        submap_prior_sigma_t: float,
        submap_prior_sigma_r: float,
        loop_super_sigma_t: float,
        loop_super_sigma_r: float,
        submap_keyframes_per_submap: int,
        enable: bool,
        path_pub,
        get_pose_fn,
        get_gmm_fn=None,
        get_submap_imu_delta_fn=None,
        get_submap_traj_delta_fn=None,
        reg_request_pub=None,
        gmm_dir: str = "/tmp/gmmslam_gmms",
        overlap_radius_m: float = 5.0,
        submap_reg_score_threshold: float = 0.5,
        submap_imu_sigma_t: float = 0.3,
        submap_imu_sigma_r: float = 0.3,
        submap_traj_sigma_t: float = 0.15,
        submap_traj_sigma_r: float = 0.15,
        score_sigma_low: float = 1.0,
        score_sigma_high: float = 2.0,
        loop_sigma_t_min: float = 0.03,
        loop_sigma_t_max: float = 0.40,
        loop_sigma_r_min: float = 0.02,
        loop_sigma_r_max: float = 0.25,
        submap_loop_sigma_t_min: float = 0.05,
        submap_loop_sigma_t_max: float = 0.50,
        submap_loop_sigma_r_min: float = 0.03,
        submap_loop_sigma_r_max: float = 0.30,
        submap_aux_gate_abs_trans_m: float = 10.0,
        submap_aux_gate_abs_rot_deg: float = 90.0,
        submap_aux_gate_consistency_trans_m: float = 2.0,
        submap_aux_gate_consistency_rot_deg: float = 35.0,
    ):
        self.odom_frame = odom_frame
        self.enable = enable
        self.submap_keyframes_per_submap = submap_keyframes_per_submap
        self._get_pose = get_pose_fn
        self._get_gmm = get_gmm_fn
        self._get_submap_imu_delta = get_submap_imu_delta_fn
        self._get_submap_traj_delta = get_submap_traj_delta_fn
        self._reg_pub = reg_request_pub
        self._path_pub = path_pub
        self._gmm_dir = gmm_dir
        self._overlap_radius_m = overlap_radius_m
        self.submap_reg_score_threshold = submap_reg_score_threshold
        self.score_sigma_low = float(score_sigma_low)
        self.score_sigma_high = float(score_sigma_high)
        self.loop_sigma_t_min = float(loop_sigma_t_min)
        self.loop_sigma_t_max = float(loop_sigma_t_max)
        self.loop_sigma_r_min = float(loop_sigma_r_min)
        self.loop_sigma_r_max = float(loop_sigma_r_max)
        self.submap_loop_sigma_t_min = float(submap_loop_sigma_t_min)
        self.submap_loop_sigma_t_max = float(submap_loop_sigma_t_max)
        self.submap_loop_sigma_r_min = float(submap_loop_sigma_r_min)
        self.submap_loop_sigma_r_max = float(submap_loop_sigma_r_max)
        self.submap_aux_gate_abs_trans_m = float(submap_aux_gate_abs_trans_m)
        self.submap_aux_gate_abs_rot_deg = float(submap_aux_gate_abs_rot_deg)
        self.submap_aux_gate_consistency_trans_m = float(
            submap_aux_gate_consistency_trans_m
        )
        self.submap_aux_gate_consistency_rot_deg = float(
            submap_aux_gate_consistency_rot_deg
        )

        os.makedirs(self._gmm_dir, exist_ok=True)

        # Noise models
        sub_sigmas = np.array(
            [submap_between_sigma_r] * 3 + [submap_between_sigma_t] * 3,
            dtype=np.float64,
        )
        self._between_noise = gtsam.noiseModel.Diagonal.Sigmas(sub_sigmas)
        imu_sigmas = np.array(
            [submap_imu_sigma_r] * 3 + [submap_imu_sigma_t] * 3,
            dtype=np.float64,
        )
        self._submap_imu_noise = gtsam.noiseModel.Diagonal.Sigmas(imu_sigmas)
        traj_sigmas = np.array(
            [submap_traj_sigma_r] * 3 + [submap_traj_sigma_t] * 3,
            dtype=np.float64,
        )
        self._submap_traj_noise = gtsam.noiseModel.Diagonal.Sigmas(traj_sigmas)
        prior_sigmas = np.array(
            [submap_prior_sigma_r] * 3 + [submap_prior_sigma_t] * 3,
            dtype=np.float64,
        )
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        loop_sigmas = np.array(
            [loop_super_sigma_r] * 3 + [loop_super_sigma_t] * 3, dtype=np.float64
        )
        self._loop_super_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

        # iSAM2 for global graph
        self._isam = gtsam.ISAM2(gtsam.ISAM2Params())
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
        self.lock = threading.Lock()

        # Submap state
        self.submap_ids: list = []
        self.submap_pose_by_idx: dict = {}
        self.submap_anchor_key: dict = {}
        self.submap_anchor_time_sec: dict = {}
        self.key_to_submap: dict = {}
        self.submap_keyframes: dict = {}
        self._last_submap_idx = -1
        self._last_submap_anchor_key_idx = None
        self._last_submap_pose = None
        self._last_submap_t_sec = None

        # Submap GMM data
        self.submap_gmm: dict = {}
        self.submap_gmm_path: dict = {}
        self.submap_gmm_components: dict = {}

        # Loop-edge tracking at the global (submap) level
        self.loop_edges_added: set = set()
        self._pending_submap_registrations: set = set()

        self._path = Path()
        self._path.header.frame_id = odom_frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _p3(T: np.ndarray):
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        return gtsam.Pose3(R, t)

    @staticmethod
    def _m3(pose) -> np.ndarray:
        return np.array(pose.matrix(), dtype=np.float64)

    @staticmethod
    def submap_color(sid: int):
        return SUBMAP_COLORS[sid % len(SUBMAP_COLORS)]

    def _noise_from_score(
        self,
        score: float,
        sigma_t_min: float,
        sigma_t_max: float,
        sigma_r_min: float,
        sigma_r_max: float,
    ):
        s_lo = float(self.score_sigma_low)
        s_hi = float(self.score_sigma_high)
        if s_hi <= s_lo:
            s_hi = s_lo + 1e-6
        alpha = (float(score) - s_lo) / (s_hi - s_lo)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        sigma_t = float(sigma_t_max - alpha * (sigma_t_max - sigma_t_min))
        sigma_r = float(sigma_r_max - alpha * (sigma_r_max - sigma_r_min))
        sigmas = np.array([sigma_r] * 3 + [sigma_t] * 3, dtype=np.float64)
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas), sigma_t, sigma_r

    @staticmethod
    def _rot_angle_deg(R: np.ndarray) -> float:
        c = float((np.trace(R) - 1.0) * 0.5)
        c = float(np.clip(c, -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _passes_aux_gate(self, name: str, T_aux: np.ndarray, T_ref: np.ndarray):
        """Gate auxiliary submap delta with absolute and consistency limits."""
        if T_aux is None or not np.all(np.isfinite(T_aux)):
            return False
        d_abs = float(np.linalg.norm(T_aux[:3, 3]))
        r_abs = self._rot_angle_deg(T_aux[:3, :3])
        if d_abs > self.submap_aux_gate_abs_trans_m:
            rospy.logwarn(
                f"[global_graph] rejected {name} submap factor (abs trans {d_abs:.3f}m > "
                f"{self.submap_aux_gate_abs_trans_m:.3f}m)"
            )
            return False
        if r_abs > self.submap_aux_gate_abs_rot_deg:
            rospy.logwarn(
                f"[global_graph] rejected {name} submap factor (abs rot {r_abs:.2f}deg > "
                f"{self.submap_aux_gate_abs_rot_deg:.2f}deg)"
            )
            return False
        if T_ref is not None and np.all(np.isfinite(T_ref)):
            T_err = np.linalg.inv(T_ref) @ T_aux
            d_err = float(np.linalg.norm(T_err[:3, 3]))
            r_err = self._rot_angle_deg(T_err[:3, :3])
            if d_err > self.submap_aux_gate_consistency_trans_m:
                rospy.logwarn(
                    f"[global_graph] rejected {name} submap factor (consistency trans "
                    f"{d_err:.3f}m > {self.submap_aux_gate_consistency_trans_m:.3f}m)"
                )
                return False
            if r_err > self.submap_aux_gate_consistency_rot_deg:
                rospy.logwarn(
                    f"[global_graph] rejected {name} submap factor (consistency rot "
                    f"{r_err:.2f}deg > {self.submap_aux_gate_consistency_rot_deg:.2f}deg)"
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Keyframe / submap lifecycle
    # ------------------------------------------------------------------

    def should_create_submap(self, key_idx: int) -> bool:
        if self._last_submap_anchor_key_idx is None:
            return True
        return (
            key_idx - self._last_submap_anchor_key_idx
        ) >= self.submap_keyframes_per_submap

    def update_with_keyframe(
        self, key_idx: int, stamp, T_curr: np.ndarray, t_sec: float
    ):
        if not self.enable:
            return

        # Assign keyframe to current open submap
        if self._last_submap_idx >= 0:
            self.submap_keyframes.setdefault(self._last_submap_idx, []).append(
                key_idx
            )
            self.key_to_submap[key_idx] = self._last_submap_idx

        if not self.should_create_submap(key_idx):
            return

        # Finalize previous submap (merge its GMMs)
        if self._last_submap_idx >= 0:
            self._finalize_submap(self._last_submap_idx, stamp)

        # --- Create new submap node ---
        sid = len(self.submap_ids)
        key_sid = X(sid)
        with self.lock:
            self._new_values.insert(key_sid, self._p3(T_curr))
            if sid == 0:
                self._new_factors.push_back(
                    gtsam.PriorFactorPose3(
                        key_sid, self._p3(T_curr), self._prior_noise
                    )
                )
            else:
                rel = np.linalg.inv(self._last_submap_pose) @ T_curr
                self._new_factors.push_back(
                    gtsam.BetweenFactorPose3(
                        X(self._last_submap_idx),
                        key_sid,
                        self._p3(rel),
                        self._between_noise,
                    )
                )
                self._add_transition_aux_factors(
                    prev_sid=self._last_submap_idx,
                    curr_sid=sid,
                    T_ref_rel=rel,
                    prev_anchor_key=self.submap_anchor_key.get(self._last_submap_idx),
                    curr_anchor_key=key_idx,
                    prev_anchor_t=self.submap_anchor_time_sec.get(self._last_submap_idx),
                    curr_anchor_t=t_sec,
                )

        self.submap_ids.append(sid)
        self.submap_pose_by_idx[sid] = T_curr.copy()
        self.submap_anchor_key[sid] = key_idx
        self.submap_anchor_time_sec[sid] = t_sec
        self._last_submap_anchor_key_idx = key_idx
        self._last_submap_idx = sid
        self._last_submap_pose = T_curr.copy()
        self._last_submap_t_sec = t_sec
        self.submap_keyframes[sid] = [key_idx]
        self.key_to_submap[key_idx] = sid

        rospy.loginfo(
            f"[global_graph] new submap S({sid}) anchored at X({key_idx}) "
            f"color={self.submap_color(sid)}"
        )
        self.commit(stamp)

    def _add_transition_aux_factors(
        self,
        prev_sid,
        curr_sid,
        T_ref_rel,
        prev_anchor_key,
        curr_anchor_key,
        prev_anchor_t,
        curr_anchor_t,
    ):
        """Add optional transition factors from IMU and fixed-lag trajectory."""
        if prev_anchor_key is None or curr_anchor_key is None:
            return

        if self._get_submap_imu_delta is not None:
            try:
                T_imu = self._get_submap_imu_delta(
                    prev_anchor_key, curr_anchor_key, prev_anchor_t, curr_anchor_t
                )
                if self._passes_aux_gate("imu", T_imu, T_ref_rel):
                    self._new_factors.push_back(
                        gtsam.BetweenFactorPose3(
                            X(prev_sid),
                            X(curr_sid),
                            self._p3(T_imu),
                            self._submap_imu_noise,
                        )
                    )
                    pos = T_imu[:3, 3]
                    rospy.loginfo(
                        f"[global_graph] ADDED BetweenFactor (submap imu) "
                        f"S({prev_sid})->S({curr_sid}) "
                        f"t=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                    )
            except Exception as e:
                rospy.logwarn_throttle(
                    2.0, f"[global_graph] failed IMU submap factor S({prev_sid})->S({curr_sid}): {e}"
                )

        if self._get_submap_traj_delta is not None:
            try:
                T_traj = self._get_submap_traj_delta(
                    prev_anchor_key, curr_anchor_key, prev_anchor_t, curr_anchor_t
                )
                if self._passes_aux_gate("traj", T_traj, T_ref_rel):
                    self._new_factors.push_back(
                        gtsam.BetweenFactorPose3(
                            X(prev_sid),
                            X(curr_sid),
                            self._p3(T_traj),
                            self._submap_traj_noise,
                        )
                    )
                    pos = T_traj[:3, 3]
                    rospy.loginfo(
                        f"[global_graph] ADDED BetweenFactor (submap traj) "
                        f"S({prev_sid})->S({curr_sid}) "
                        f"t=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                    )
            except Exception as e:
                rospy.logwarn_throttle(
                    2.0, f"[global_graph] failed traj submap factor S({prev_sid})->S({curr_sid}): {e}"
                )

    # ------------------------------------------------------------------
    # Submap finalization
    # ------------------------------------------------------------------

    def _finalize_submap(self, sid, stamp):
        """Merge constituent keyframe GMMs into a single submap GMM.

        Uses each keyframe's capture-time pose (the pose at the moment the
        point cloud was acquired) rather than the latest smoother estimate,
        so the GMM components are consistent with the actual observations.
        """
        if self._get_gmm is None:
            return

        key_indices = self.submap_keyframes.get(sid, [])
        T_ref = self.submap_pose_by_idx.get(sid)
        if T_ref is None or not key_indices:
            return

        gmms_with_poses = []
        for ki in key_indices:
            result = self._get_gmm(ki)
            if result is None:
                continue
            gmm, capture_pose = result
            T_kf = capture_pose if capture_pose is not None else self._get_pose(ki)
            if gmm is not None and T_kf is not None:
                gmms_with_poses.append((gmm, T_kf))

        if not gmms_with_poses:
            rospy.logwarn(
                f"[global_graph] S({sid}): no keyframe GMMs available "
                f"({len(key_indices)} keyframes, 0 GMMs)"
            )
            return

        merged = merge_gmms_concatenate(gmms_with_poses, T_ref)
        if merged is None:
            return

        self.submap_gmm[sid] = merged
        self.submap_gmm_components[sid] = precompute_gmm_local_data(merged)

        n_comp = merged.n_components_ if hasattr(merged, "n_components_") else merged.n_components
        rospy.loginfo(
            f"[global_graph] S({sid}) finalized: {n_comp} components "
            f"from {len(gmms_with_poses)}/{len(key_indices)} keyframes"
        )

        gmm_path = os.path.join(self._gmm_dir, f"submap_{sid:04d}.gmm")
        try:
            save_gmm_to_file(merged, gmm_path)
            self.submap_gmm_path[sid] = gmm_path
        except Exception as e:
            rospy.logwarn(f"[global_graph] failed to save S({sid}) GMM: {e}")
            return

        self._request_overlap_registrations(sid, stamp)

    # ------------------------------------------------------------------
    # Overlap-based submap registration
    # ------------------------------------------------------------------

    def _request_overlap_registrations(self, sid_new, stamp):
        """Check spatial overlap with older submaps and request D2D registration."""
        if self._reg_pub is None:
            return
        T_new = self.submap_pose_by_idx.get(sid_new)
        new_path = self.submap_gmm_path.get(sid_new)
        if T_new is None or new_path is None:
            return
        pos_new = T_new[:3, 3]

        for sid_old in self.submap_ids:
            if sid_old >= sid_new:
                continue
            if sid_old == sid_new - 1:
                continue
            edge = (min(sid_old, sid_new), max(sid_old, sid_new))
            if edge in self.loop_edges_added or edge in self._pending_submap_registrations:
                continue
            T_old = self.submap_pose_by_idx.get(sid_old)
            old_path = self.submap_gmm_path.get(sid_old)
            if T_old is None or old_path is None:
                continue
            d = float(np.linalg.norm(T_old[:3, 3] - pos_new))
            if d > self._overlap_radius_m:
                continue

            payload = {
                "prev_idx": int(sid_old),
                "curr_idx": int(sid_new),
                "stamp": float(stamp_to_sec(stamp)),
                "source_path": new_path,
                "target_path": old_path,
                "is_loop_closure": False,
                "is_submap_registration": True,
            }
            from std_msgs.msg import String

            self._reg_pub.publish(String(data=json.dumps(payload)))
            self._pending_submap_registrations.add(edge)
            rospy.loginfo(
                f"[global_graph] requested registration "
                f"S({sid_old})<->S({sid_new}) (d={d:.2f}m)"
            )

    def handle_submap_registration_result(
        self, sid_prev, sid_curr, T_rel, score, stamp
    ):
        """Process a completed submap D2D registration."""
        edge = (min(sid_prev, sid_curr), max(sid_prev, sid_curr))
        self._pending_submap_registrations.discard(edge)

        if score < self.submap_reg_score_threshold:
            rospy.loginfo(
                f"[global_graph] submap reg S({sid_prev})->S({sid_curr}) "
                f"rejected (score={score:.4f} < {self.submap_reg_score_threshold})"
            )
            return

        if edge in self.loop_edges_added:
            rospy.logdebug(
                f"[global_graph] submap edge S({sid_prev})<->S({sid_curr}) "
                f"already exists, skipping"
            )
            return

        pos = T_rel[:3, 3]
        noise, sigma_t, sigma_r = self._noise_from_score(
            score=score,
            sigma_t_min=self.submap_loop_sigma_t_min,
            sigma_t_max=self.submap_loop_sigma_t_max,
            sigma_r_min=self.submap_loop_sigma_r_min,
            sigma_r_max=self.submap_loop_sigma_r_max,
        )
        with self.lock:
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    X(sid_prev),
                    X(sid_curr),
                    self._p3(T_rel),
                    noise,
                )
            )
        self.loop_edges_added.add(edge)
        rospy.loginfo(
            f"[global_graph] ADDED BetweenFactor "
            f"S({sid_prev})->S({sid_curr}) score={score:.4f} "
            f"t=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
            f"sigma_t={sigma_t:.4f} sigma_r={sigma_r:.4f}"
        )
        self.commit(stamp)

    # ------------------------------------------------------------------
    # Legacy keyframe-level loop factor (from registration.py)
    # ------------------------------------------------------------------

    def add_loop_factor(
        self,
        prev_key_idx: int,
        curr_key_idx: int,
        T_prev_to_curr: np.ndarray,
        stamp,
        pose_by_idx: dict,
        score: float = None,
    ):
        if not self.enable:
            return
        sid_prev = self.key_to_submap.get(prev_key_idx)
        sid_curr = self.key_to_submap.get(curr_key_idx)
        if sid_prev is None or sid_curr is None or sid_prev == sid_curr:
            return
        edge = (min(sid_prev, sid_curr), max(sid_prev, sid_curr))
        if edge in self.loop_edges_added:
            return

        T_rel_sub = T_prev_to_curr.copy()
        T_w_kp = pose_by_idx.get(prev_key_idx)
        T_w_kc = pose_by_idx.get(curr_key_idx)
        T_w_sp = self.submap_pose_by_idx.get(sid_prev)
        T_w_sc = self.submap_pose_by_idx.get(sid_curr)
        if all(t is not None for t in [T_w_kp, T_w_kc, T_w_sp, T_w_sc]):
            T_sp_kp = np.linalg.inv(T_w_sp) @ T_w_kp
            T_kc_sc = np.linalg.inv(T_w_kc) @ T_w_sc
            T_rel_sub = T_sp_kp @ T_prev_to_curr @ T_kc_sc

        use_score = self.submap_reg_score_threshold if score is None else float(score)
        noise, sigma_t, sigma_r = self._noise_from_score(
            score=use_score,
            sigma_t_min=self.loop_sigma_t_min,
            sigma_t_max=self.loop_sigma_t_max,
            sigma_r_min=self.loop_sigma_r_min,
            sigma_r_max=self.loop_sigma_r_max,
        )
        with self.lock:
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    X(sid_prev),
                    X(sid_curr),
                    self._p3(T_rel_sub),
                    noise,
                )
            )
        self.loop_edges_added.add(edge)
        pos = T_rel_sub[:3, 3]
        rospy.loginfo(
            f"[global_graph] ADDED BetweenFactor (keyframe loop) "
            f"S({sid_prev})->S({sid_curr}) via X({prev_key_idx})->X({curr_key_idx}) "
            f"t=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
            f"score={use_score:.4f} sigma_t={sigma_t:.4f} sigma_r={sigma_r:.4f}"
        )
        self.commit(stamp)

    # ------------------------------------------------------------------
    # iSAM2 update
    # ------------------------------------------------------------------

    def commit(self, stamp):
        if not self.enable:
            return
        with self.lock:
            try:
                self._isam.update(self._new_factors, self._new_values)
                self._new_factors.resize(0)
                self._new_values.clear()
                est = self._isam.calculateEstimate()
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[global_graph] update failed: {e}")
                return

            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = self.odom_frame
            for sid in self.submap_ids:
                try:
                    T_sid = self._m3(est.atPose3(X(sid)))
                except Exception:
                    continue
                self.submap_pose_by_idx[sid] = T_sid
                path.poses.append(
                    pose_to_pose_stamped(T_sid, stamp, self.odom_frame)
                )
            self._path = path
        self._path_pub.publish(self._path)
