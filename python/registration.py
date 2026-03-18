"""Registration manager: SOGMM fitting, D2D registration requests, and factor staging."""

import json
import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np
import rospy
from std_msgs.msg import String
from scipy.spatial.transform import Rotation

from imports_setup import HAS_SOGMM, SOGMM, gtsam, X
from ros_helpers import make_pcld_4d, stamp_to_sec
from gmm_utils import save_gmm_to_file, plot_gmm_3d

if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm


def _sogmm_fit_in_process(args):
    """Run SOGMM fit in a subprocess so the main process GIL stays free.

    Returns (means, covariances, weights, n_components) or None on failure.
    """
    bandwidth, compute, n_components, pcld_4d = args
    try:
        import sys as _sys
        import os as _os
        import glob as _glob

        gira_ws = _os.environ.get("GIRA_WS", "/root/gira_ws")
        for sub in ("gira3d-reconstruction", "gira3d-registration"):
            pattern = _os.path.join(
                gira_ws, sub, ".venv", "lib", "python3*", "site-packages"
            )
            for sp in _glob.glob(pattern):
                if sp not in _sys.path:
                    _sys.path.insert(0, sp)

        from sogmm_py.sogmm import SOGMM as _SOGMM

        sg = _SOGMM(bandwidth, compute=compute)
        if n_components > 0:
            model = sg.gmm_fit(pcld_4d, n_components)
        else:
            model = sg.fit(pcld_4d)
        if model is None:
            return None
        means = np.array(model.means_, dtype=np.float64)
        covs = np.array(model.covariances_, dtype=np.float64)
        D = means.shape[1]
        if covs.ndim == 2:
            covs = covs.reshape(-1, D, D)
        return (
            means,
            covs,
            np.array(model.weights_, dtype=np.float64),
            int(model.n_components_),
        )
    except Exception as e:
        import traceback as _tb

        _tb.print_exc()
        return None


class RegistrationManager:
    """Async GMM fitting, D2D registration request dispatch, and factor staging."""

    def __init__(self, smoother, global_graph, reg_request_pub, **cfg):
        self.smoother = smoother
        self.global_graph = global_graph
        self._pub = reg_request_pub

        # Configuration
        self.sogmm_bandwidth = cfg.get("sogmm_bandwidth", 0.1)
        self.sogmm_compute = cfg.get("sogmm_compute", "CPU")
        self.sogmm_max_points = cfg.get("sogmm_max_points", 2000)
        self.sogmm_n_components = cfg.get("sogmm_n_components", 0)
        self.gmm_dir = cfg.get("gmm_dir", "/tmp/gmmslam_gmms")
        self.plot_first_frame = cfg.get("plot_first_frame", False)
        self.compensate_fit_latency_in_map = cfg.get(
            "compensate_fit_latency_in_map", True
        )
        self.registration_score_threshold = cfg.get(
            "registration_score_threshold", -1e9
        )
        self.registration_factor_every_n_frames = cfg.get(
            "registration_factor_every_n_frames", 1
        )
        self.loop_closure_score_threshold = cfg.get("loop_closure_score_threshold", 2.0)
        self.loop_closure_detect_score_threshold = cfg.get(
            "loop_closure_detect_score_threshold", 0.8
        )
        self.enable_loop_closure_detection = cfg.get(
            "enable_loop_closure_detection", True
        )
        self.loop_closure_min_keyframe_gap = cfg.get(
            "loop_closure_min_keyframe_gap", 10
        )
        self.loop_closure_max_candidates = cfg.get("loop_closure_max_candidates", 100)
        self.loop_closure_search_radius_m = cfg.get("loop_closure_search_radius_m", 3.0)
        self.loop_closure_search_cooldown_keyframes = cfg.get(
            "loop_closure_search_cooldown_keyframes", 2
        )
        self.loop_closure_request_every_n_keyframes = cfg.get(
            "loop_closure_request_every_n_keyframes", 1
        )
        self.loop_closure_min_separation_m = cfg.get(
            "loop_closure_min_separation_m", 0.8
        )
        self.loop_closure_min_separation_deg = cfg.get(
            "loop_closure_min_separation_deg", 20.0
        )
        self.loop_closure_max_age_s = cfg.get("loop_closure_max_age_s", 520.0)
        self.loop_closure_gmm_keep_keyframes = cfg.get(
            "loop_closure_gmm_keep_keyframes", 2000
        )
        self.score_sigma_low = float(
            cfg.get("score_sigma_low", self.loop_closure_detect_score_threshold)
        )
        self.score_sigma_high = float(
            cfg.get("score_sigma_high", self.loop_closure_score_threshold)
        )
        self.seq_sigma_t_min = float(cfg.get("seq_sigma_t_min", 0.02))
        self.seq_sigma_t_max = float(cfg.get("seq_sigma_t_max", 0.20))
        self.seq_sigma_r_min = float(cfg.get("seq_sigma_r_min", 0.01))
        self.seq_sigma_r_max = float(cfg.get("seq_sigma_r_max", 0.15))
        self.loop_sigma_t_min = float(cfg.get("loop_sigma_t_min", 0.03))
        self.loop_sigma_t_max = float(cfg.get("loop_sigma_t_max", 0.40))
        self.loop_sigma_r_min = float(cfg.get("loop_sigma_r_min", 0.02))
        self.loop_sigma_r_max = float(cfg.get("loop_sigma_r_max", 0.25))
        self.map_decimate = cfg.get("map_decimate", 5)
        os.makedirs(self.gmm_dir, exist_ok=True)

        # State
        self.lock = threading.Lock()
        self.gmm_paths_by_idx: dict = {}
        self.local_gmms_by_idx: dict = {}
        self.latest_gmm_idx: int = -1
        self.latest_gmm_model = None
        self._pending_loop_requests: set = set()
        self.loop_edges_added: set = set()
        self._last_loop_search_idx = -1000000
        self._dropped_fit_frames = 0
        self._dropped_result_msgs = 0

        # SOGMM global model (for map accumulation)
        if HAS_SOGMM and SOGMM is not None:
            self.sg = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            self.local_gmms: list = []
        else:
            self.sg = None
            self.local_gmms = []

        # Subprocess pool for SOGMM fitting (avoids GIL contention).
        # 'spawn' is safer than 'fork' after rospy.init_node() because fork
        # in a multi-threaded process can deadlock on C++ mutex state in SOGMM.
        self._fit_pool = None
        self._fit_pool_ok = False
        self._fit_pool_workers = max(1, int(cfg.get("fit_pool_workers", 2)))
        self._fit_pool_timeout_s = 60.0
        self._fit_pool_consecutive_timeouts = 0
        if HAS_SOGMM:
            for method in ("spawn", "forkserver", "fork"):
                try:
                    ctx = mp.get_context(method)
                    self._fit_pool = ctx.Pool(processes=self._fit_pool_workers)
                    self._fit_pool_ok = True
                    rospy.loginfo(
                        f"[registration] subprocess pool: "
                        f"{self._fit_pool_workers} workers ({method})"
                    )
                    break
                except Exception as e:
                    rospy.logwarn(f"[registration] Pool({method}) failed: {e}")
            if not self._fit_pool_ok:
                rospy.logwarn(
                    "[registration] all Pool methods failed, "
                    "using in-thread SOGMM fitting (may cause lag)"
                )

        # Queues
        reg_q_size = cfg.get("registration_queue_size", 8)
        res_q_size = cfg.get("registration_result_queue_size", 64)
        self.fit_queue: queue.Queue = queue.Queue(maxsize=reg_q_size)
        self.result_queue: queue.Queue = queue.Queue(maxsize=res_q_size)

    # ------------------------------------------------------------------
    # SOGMM fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gmm_from_params(means, covariances, weights, n_comp):
        """Reconstruct a sklearn GaussianMixture from raw arrays."""
        from sklearn.mixture import GaussianMixture

        D = means.shape[1]
        if covariances.ndim == 2:
            covariances = covariances.reshape(-1, D, D)

        gmm = GaussianMixture(n_components=n_comp, covariance_type="full")
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.weights_ = weights
        gmm.n_components_ = n_comp
        gmm.converged_ = True
        gmm.n_iter_ = 0
        gmm.lower_bound_ = -np.inf
        try:
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
        except np.linalg.LinAlgError:
            covs_safe = covariances + 1e-3 * np.eye(D)
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs_safe))
        return gmm

    def _fit_sogmm_sync(self, pcld_4d, frame_idx):
        """In-thread fallback when the subprocess pool is unavailable."""
        try:
            sg_local = SOGMM(self.sogmm_bandwidth, compute=self.sogmm_compute)
            if self.sogmm_n_components > 0:
                return sg_local.gmm_fit(pcld_4d, self.sogmm_n_components)
            else:
                return sg_local.fit(pcld_4d)
        except Exception as e:
            rospy.logerr(
                f"[registration] in-thread SOGMM fit failed on frame {frame_idx}: {e}"
            )
            return None

    def _finish_fit(self, local_model, frame_idx, stamp, capture_t_sec, capture_pose):
        """Post-fit bookkeeping: save .gmm file, emit registration & loop requests."""
        self.local_gmms.append((stamp, local_model))
        with self.lock:
            self.latest_gmm_idx = frame_idx
            self.latest_gmm_model = local_model
            self.local_gmms_by_idx[frame_idx] = (
                stamp,
                local_model,
                None,
                None,
                None,
                capture_pose,
            )
            stale = [k for k in self.local_gmms_by_idx if k < (frame_idx - 400)]
            for k in stale:
                del self.local_gmms_by_idx[k]

        n_local = getattr(local_model, "n_components_", local_model.n_components)
        rospy.loginfo(
            f"[registration] frame {frame_idx:4d} | "
            f"local GMM: {n_local:3d} components"
        )

        if self.plot_first_frame and frame_idx == 0:
            rospy.loginfo(
                "[registration] plot_first_frame=True — opening GMM visualisation"
            )
            plot_gmm_3d(
                local_model,
                sigma=1.0,
                title=f"First frame local GMM (frame 0, K={n_local})",
            )

        gmm_path = os.path.join(self.gmm_dir, f"frame_{frame_idx:06d}.gmm")
        try:
            save_gmm_to_file(local_model, gmm_path)
        except Exception as e:
            rospy.logerr(f"[registration] failed to save GMM: {e}")
            return

        fit_t_sec = stamp_to_sec(rospy.Time.now())
        if capture_pose is None:
            with self.smoother.graph_lock:
                capture_pose = self.smoother.pose_by_idx.get(
                    frame_idx, self.smoother.pose
                ).copy()

        if self.compensate_fit_latency_in_map:
            with self.smoother.graph_lock:
                map_pose = self.smoother.pose_by_idx.get(frame_idx)
                if map_pose is not None:
                    map_pose = map_pose.copy()
                else:
                    T_now = self.smoother.pose.copy()
                    map_pose = T_now @ np.linalg.inv(capture_pose) @ capture_pose
        else:
            with self.smoother.graph_lock:
                map_pose = self.smoother.pose_by_idx.get(frame_idx, capture_pose).copy()

        with self.lock:
            self.gmm_paths_by_idx[frame_idx] = gmm_path
            if frame_idx in self.local_gmms_by_idx:
                entry = self.local_gmms_by_idx[frame_idx]
                self.local_gmms_by_idx[frame_idx] = (
                    entry[0],
                    entry[1],
                    map_pose,
                    capture_t_sec,
                    fit_t_sec,
                    capture_pose,
                )
            prev_idx = max(
                (k for k in self.gmm_paths_by_idx if k < frame_idx),
                default=None,
            )
            prev_path = (
                self.gmm_paths_by_idx.get(prev_idx) if prev_idx is not None else None
            )

        if prev_path is not None:
            payload = {
                "prev_idx": int(prev_idx),
                "curr_idx": int(frame_idx),
                "stamp": float(stamp_to_sec(stamp)),
                "source_path": gmm_path,
                "target_path": prev_path,
                "is_loop_closure": False,
            }
            self._pub.publish(String(data=json.dumps(payload)))

            self._enqueue_loop_closure_requests(
                frame_idx, stamp, gmm_path, sequential_prev_idx=prev_idx
            )

        with self.lock:
            min_keep = max(0, frame_idx - self.loop_closure_gmm_keep_keyframes)
            stale_paths = [k for k in self.gmm_paths_by_idx if k < min_keep]
            for k in stale_paths:
                del self.gmm_paths_by_idx[k]

    # ------------------------------------------------------------------
    # Fit worker (background thread)
    # ------------------------------------------------------------------

    def fit_worker_loop(self):
        """Background thread: submit GMM fits to subprocess pool, process results.

        Uses apply_async so multiple fits can run in parallel across pool
        workers.  When the subprocess pool is unavailable, falls back to
        synchronous in-thread fitting.
        """
        pending = []

        while not rospy.is_shutdown():
            # --- 1. Drain fit queue and submit new items ---
            drained = 0
            while drained < 8:
                try:
                    item = self.fit_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    for ar, *_ in pending:
                        try:
                            ar.wait(timeout=5)
                        except Exception:
                            pass
                    return

                drained += 1
                if len(item) >= 5:
                    frame_idx, stamp, pts, capture_t_sec, capture_pose = item
                else:
                    frame_idx, stamp, pts = item
                    capture_t_sec = stamp_to_sec(stamp)
                    capture_pose = None

                pcld_4d = make_pcld_4d(pts)
                if (
                    self.sogmm_max_points > 0
                    and pcld_4d.shape[0] > self.sogmm_max_points
                ):
                    sample_idx = np.random.choice(
                        pcld_4d.shape[0], self.sogmm_max_points, replace=False
                    )
                    pcld_4d = pcld_4d[sample_idx]

                if self._fit_pool_ok:
                    try:
                        ar = self._fit_pool.apply_async(
                            _sogmm_fit_in_process,
                            (
                                (
                                    self.sogmm_bandwidth,
                                    self.sogmm_compute,
                                    self.sogmm_n_components,
                                    pcld_4d,
                                ),
                            ),
                        )
                        pending.append(
                            (
                                ar,
                                frame_idx,
                                stamp,
                                capture_t_sec,
                                capture_pose,
                                time.monotonic(),
                            )
                        )
                        continue
                    except Exception as e:
                        rospy.logwarn(
                            f"[registration] async submit failed ({e}), "
                            "falling back to in-thread"
                        )
                        self._fit_pool_ok = False

                model = self._fit_sogmm_sync(pcld_4d, frame_idx)
                if model is not None:
                    try:
                        self._finish_fit(
                            model, frame_idx, stamp, capture_t_sec, capture_pose
                        )
                    except Exception as e:
                        rospy.logerr_throttle(
                            2.0, f"[registration] finish_fit error: {e}"
                        )

            # --- 2. Harvest completed async results ---
            now_mono = time.monotonic()
            still_pending = []
            for ar, fi, st, ct, cp, submit_t in pending:
                if ar.ready():
                    self._fit_pool_consecutive_timeouts = 0
                    try:
                        result = ar.get(timeout=0)
                        if result is not None:
                            means, covs, weights, n_comp = result
                            model = self._build_gmm_from_params(
                                means, covs, weights, n_comp
                            )
                            self._finish_fit(model, fi, st, ct, cp)
                        else:
                            rospy.logwarn_throttle(
                                5.0,
                                f"[registration] subprocess fit returned None "
                                f"for frame {fi}",
                            )
                    except Exception as e:
                        rospy.logwarn_throttle(
                            2.0,
                            f"[registration] subprocess fit error for "
                            f"frame {fi}: {e}",
                        )
                elif (now_mono - submit_t) > self._fit_pool_timeout_s:
                    self._fit_pool_consecutive_timeouts += 1
                    rospy.logwarn(
                        f"[registration] subprocess fit timed out for "
                        f"frame {fi} after {self._fit_pool_timeout_s:.0f}s "
                        f"(consecutive={self._fit_pool_consecutive_timeouts})"
                    )
                    if self._fit_pool_consecutive_timeouts >= 3:
                        rospy.logerr(
                            "[registration] subprocess pool appears dead, "
                            "switching to in-thread fitting"
                        )
                        self._fit_pool_ok = False
                        try:
                            self._fit_pool.terminate()
                        except Exception:
                            pass
                        still_pending.clear()
                        break
                else:
                    still_pending.append((ar, fi, st, ct, cp, submit_t))
            pending = still_pending

            # --- 3. Adaptive sleep ---
            time.sleep(0.02 if pending else 0.05)

    # ------------------------------------------------------------------
    # Loop closure
    # ------------------------------------------------------------------

    def _enqueue_loop_closure_requests(
        self, curr_idx, stamp, source_path, sequential_prev_idx=None
    ):
        if not self.enable_loop_closure_detection:
            return
        if curr_idx < self.loop_closure_min_keyframe_gap:
            return
        if (curr_idx % self.loop_closure_request_every_n_keyframes) != 0:
            return
        if (
            curr_idx - self._last_loop_search_idx
        ) < self.loop_closure_search_cooldown_keyframes:
            return
        self._last_loop_search_idx = curr_idx

        with self.smoother.graph_lock:
            T_curr = self.smoother.pose_by_idx.get(curr_idx)
            if T_curr is None:
                rospy.logwarn_throttle(
                    5.0,
                    f"[registration] loop search: no pose for X({curr_idx})",
                )
                return
            curr_pos = T_curr[:3, 3].copy()
            pose_snapshot = {k: v.copy() for k, v in self.smoother.pose_by_idx.items()}

        t_curr = self.smoother.key_t_sec.get(curr_idx)
        if t_curr is None:
            return

        with self.lock:
            gmm_snapshot = dict(self.gmm_paths_by_idx)
            pending_snapshot = set(self._pending_loop_requests)
            added_snapshot = set(self.loop_edges_added)

        rospy.logdebug(
            f"[registration] loop search @key {curr_idx}: "
            f"{len(gmm_snapshot)} GMMs available, "
            f"pos=[{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}]"
        )

        near = []
        for idx, target_path in gmm_snapshot.items():
            if idx >= curr_idx:
                continue
            if curr_idx - idx < self.loop_closure_min_keyframe_gap:
                continue
            if sequential_prev_idx is not None and idx == sequential_prev_idx:
                continue
            t_idx = self.smoother.key_t_sec.get(idx)
            if t_idx is None:
                continue
            if (t_curr - t_idx) > self.loop_closure_max_age_s:
                continue
            T_idx = pose_snapshot.get(idx)
            if T_idx is None:
                continue
            d = float(np.linalg.norm(T_idx[:3, 3] - curr_pos))
            if d > self.loop_closure_search_radius_m:
                continue
            edge = (min(idx, curr_idx), max(idx, curr_idx))
            if edge in pending_snapshot or edge in added_snapshot:
                continue
            near.append((d, idx, target_path))

        if not near:
            rospy.loginfo_throttle(
                10.0,
                f"[registration] loop search @key {curr_idx}: "
                f"0 candidates within {self.loop_closure_search_radius_m:.1f}m "
                f"(GMMs={len(gmm_snapshot)}, "
                f"gap>={self.loop_closure_min_keyframe_gap})",
            )
            return
        near.sort(key=lambda x: x[0])
        selected = near[: self.loop_closure_max_candidates]
        rospy.loginfo(
            f"[registration] loop search @key {curr_idx}: "
            f"{len(near)} candidates, dispatching {len(selected)}"
        )

        for _, idx, target_path in selected:
            payload = {
                "prev_idx": int(idx),
                "curr_idx": int(curr_idx),
                "stamp": float(stamp_to_sec(stamp)),
                "source_path": source_path,
                "target_path": target_path,
                "is_loop_closure": True,
            }
            self._pub.publish(String(data=json.dumps(payload)))
            with self.lock:
                self._pending_loop_requests.add(
                    (min(int(idx), int(curr_idx)), max(int(idx), int(curr_idx)))
                )

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------

    def _noise_from_score(
        self,
        score: float,
        sigma_t_min: float,
        sigma_t_max: float,
        sigma_r_min: float,
        sigma_r_max: float,
    ):
        """Map registration score to a bounded diagonal Pose3 noise model."""
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

    def result_callback(self, msg):
        """Parse async registration result and enqueue for staging."""
        try:
            data = json.loads(msg.data)
            result_stamp_sec = float(data.get("stamp", stamp_to_sec(rospy.Time.now())))
            if not np.isfinite(result_stamp_sec):
                result_stamp_sec = float(stamp_to_sec(rospy.Time.now()))

            # --- Submap-level registration results ---
            if bool(data.get("is_submap_registration", False)):
                self._handle_submap_result(data, result_stamp_sec)
                return

            is_loop = bool(data.get("is_loop_closure", False))
            prev_idx = int(data["prev_idx"])
            curr_idx = int(data["curr_idx"])
            edge = (min(prev_idx, curr_idx), max(prev_idx, curr_idx))
            if is_loop:
                with self.lock:
                    self._pending_loop_requests.discard(edge)
            if not bool(data.get("success", False)):
                return
            score = float(data.get("score", float("-inf")))
            if is_loop:
                if (curr_idx - prev_idx) < self.loop_closure_min_keyframe_gap:
                    return
                if score < self.loop_closure_detect_score_threshold:
                    return
            else:
                if score < self.registration_score_threshold:
                    return
                if (curr_idx % self.registration_factor_every_n_frames) != 0:
                    return
            if prev_idx >= curr_idx:
                return
            T = np.array(data["transform"], dtype=np.float64).reshape(4, 4)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[registration] bad result: {e}")
            return

        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            return

        force_loop = is_loop or (score >= self.loop_closure_score_threshold)
        use_super = is_loop and (score >= self.loop_closure_detect_score_threshold)
        if is_loop:
            rospy.loginfo(
                f"[registration] loop detected: X({prev_idx})->X({curr_idx}) "
                f"score={score:.4f} (super_noise={use_super})"
            )
            if use_super and self.global_graph is not None:
                with self.smoother.graph_lock:
                    pose_snap = dict(self.smoother.pose_by_idx)
                    stamp_msg = rospy.Time.from_sec(result_stamp_sec)
                self.global_graph.add_loop_factor(
                        prev_idx, curr_idx, T, stamp_msg, pose_snap, score=score
                )

        try:
            self.result_queue.put_nowait(
                (
                    prev_idx,
                    curr_idx,
                    T,
                    force_loop,
                    use_super,
                    is_loop,
                    score,
                    result_stamp_sec,
                )
            )
        except queue.Full:
            self._dropped_result_msgs += 1

    def _handle_submap_result(self, data, result_stamp_sec):
        """Route a submap D2D registration result to the global graph."""
        sid_prev = int(data["prev_idx"])
        sid_curr = int(data["curr_idx"])
        success = bool(data.get("success", False))
        score = float(data.get("score", float("-inf")))
        rospy.loginfo(
            f"[registration] submap result S({sid_prev})<->S({sid_curr}) "
            f"success={success} score={score:.4f}"
        )
        if not success:
            return
        if self.global_graph is None:
            return
        T = np.array(data["transform"], dtype=np.float64).reshape(4, 4)
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            rospy.logwarn(
                f"[registration] submap result S({sid_prev})<->S({sid_curr}) "
                f"has NaN/Inf transform, discarding"
            )
            return
        stamp_msg = rospy.Time.from_sec(float(result_stamp_sec))
        self.global_graph.handle_submap_registration_result(
            sid_prev, sid_curr, T, score, stamp_msg
        )

    def drain_results(self, stamp):
        """Drain result queue and stage factors in the smoother."""
        while True:
            try:
                item = self.result_queue.get_nowait()
                (
                    r_prev,
                    r_curr,
                    r_T,
                    r_loop,
                    r_super,
                    r_is_loop,
                    r_score,
                    r_stamp_sec,
                ) = item
                factor_stamp = (
                    rospy.Time.from_sec(float(r_stamp_sec))
                    if np.isfinite(float(r_stamp_sec))
                    else stamp
                )
                self._stage_registration_factor(
                    r_prev,
                    r_curr,
                    r_T,
                    score=r_score,
                    force_loop=r_loop,
                    use_super_loop_noise=r_super,
                    is_loop_candidate=r_is_loop,
                    stamp=factor_stamp,
                )
            except queue.Empty:
                break
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[registration] drain error: {e}")
                break

    def _stage_registration_factor(
        self,
        prev_idx,
        curr_idx,
        T_prev_to_curr,
        score,
        force_loop=False,
        use_super_loop_noise=False,
        is_loop_candidate=False,
        stamp=None,
    ):
        """Push a registration factor into the smoother's pending batch.

        Sequential registration results add a BetweenFactor (both keys must
        be in the lag window).  Loop closures add a **PriorFactor** on
        curr_idx, computed from the stored world pose of prev_idx and the
        relative transform.  A prior is more robust for fixed-lag smoothing
        because it works even when prev_idx has already been marginalized.
        """
        sm = self.smoother
        if prev_idx < 0 or curr_idx > sm.latest_key_idx:
            return

        t_curr = sm.key_t_sec.get(curr_idx)
        if t_curr is None:
            return
        t_latest = sm.key_t_sec.get(sm.latest_key_idx, 0.0)
        if (t_latest - t_curr) > sm.fixed_lag_s * 1.1:
            return

        edge = (min(prev_idx, curr_idx), max(prev_idx, curr_idx))
        with self.lock:
            if is_loop_candidate and edge in self.loop_edges_added:
                return

        if force_loop:
            # --- Loop closure → PriorFactor on curr_idx ---
            # Compute the absolute world pose implied by the loop:
            #   T_w_curr = T_w_prev @ T_prev_to_curr
            T_w_prev = sm.pose_by_idx.get(prev_idx)
            if T_w_prev is None:
                rospy.logwarn(
                    f"[registration] no stored pose for X({prev_idx}), "
                    f"cannot stage loop prior on X({curr_idx})"
                )
                return
            T_w_curr_from_loop = T_w_prev @ T_prev_to_curr
            noise, sigma_t, sigma_r = self._noise_from_score(
                score=score,
                sigma_t_min=self.loop_sigma_t_min,
                sigma_t_max=self.loop_sigma_t_max,
                sigma_r_min=self.loop_sigma_r_min,
                sigma_r_max=self.loop_sigma_r_max,
            )
            sm.stage_factor(
                gtsam.PriorFactorPose3(
                    X(curr_idx),
                    sm.pose3_from_matrix(T_w_curr_from_loop),
                    noise,
                )
            )
            if is_loop_candidate:
                with self.lock:
                    self.loop_edges_added.add(edge)
                if (
                    stamp is not None
                    and use_super_loop_noise
                    and self.global_graph is not None
                ):
                    with sm.graph_lock:
                        pose_snap = dict(sm.pose_by_idx)
                    self.global_graph.add_loop_factor(
                        prev_idx,
                        curr_idx,
                        T_prev_to_curr,
                        stamp,
                        pose_snap,
                        score=score,
                    )
            rospy.loginfo(
                f"[registration] staged loop prior on X({curr_idx}) "
                f"from X({prev_idx})->X({curr_idx}) "
                f"score={score:.4f} sigma_t={sigma_t:.4f} sigma_r={sigma_r:.4f}"
            )
        else:
            # --- Sequential registration → BetweenFactor ---
            t_prev = sm.key_t_sec.get(prev_idx)
            if t_prev is None:
                return
            if (t_latest - t_prev) > sm.fixed_lag_s * 1.1:
                return
            rel_pose = sm.pose3_from_matrix(T_prev_to_curr)
            noise, sigma_t, sigma_r = self._noise_from_score(
                score=score,
                sigma_t_min=self.seq_sigma_t_min,
                sigma_t_max=self.seq_sigma_t_max,
                sigma_r_min=self.seq_sigma_r_min,
                sigma_r_max=self.seq_sigma_r_max,
            )
            sm.stage_factor(
                gtsam.BetweenFactorPose3(
                    X(prev_idx), X(curr_idx), rel_pose, noise
                )
            )
            rospy.loginfo(
                f"[registration] staged between factor X({prev_idx})->X({curr_idx}) "
                f"score={score:.4f} sigma_t={sigma_t:.4f} sigma_r={sigma_r:.4f}"
            )

    # ------------------------------------------------------------------
    # Enqueue fit
    # ------------------------------------------------------------------

    def enqueue_fit(self, frame_idx, stamp, pts, capture_t_sec, capture_pose):
        """Enqueue a GMM fit. Returns True if enqueued, False if dropped."""
        try:
            self.fit_queue.put_nowait(
                (frame_idx, stamp, pts.copy(), capture_t_sec, capture_pose)
            )
            return True
        except queue.Full:
            self._dropped_fit_frames += 1
            try:
                _ = self.fit_queue.get_nowait()
                self.fit_queue.put_nowait(
                    (frame_idx, stamp, pts.copy(), capture_t_sec, capture_pose)
                )
                return True
            except (queue.Empty, queue.Full):
                return False
