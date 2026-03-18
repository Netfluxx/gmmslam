"""Fixed-lag smoothing backend wrapping GTSAM IncrementalFixedLagSmoother."""

import queue
import threading

import numpy as np
import rospy

from imports_setup import gtsam, gtsam_unstable, X, HAS_GTSAM, HAS_GTSAM_UNSTABLE
from ros_helpers import stamp_to_sec


class FixedLagBackend:
    """Manages the GTSAM fixed-lag smoother and the dedicated solve thread."""

    def __init__(
        self,
        lag_s: float,
        odom_sigma_t: float,
        odom_sigma_r: float,
        gt_factor_sigma_t: float,
        gt_factor_sigma_r: float,
        loop_sigma_t: float,
        loop_sigma_r: float,
        loop_super_sigma_t: float,
        loop_super_sigma_r: float,
        lost_scale: float = 10.0,
        prior_scale: float = 0.1,
        pose_history_keep: int = 5000,
        enable_imu_preintegration: bool = False,
        imu_gravity_mps2: float = 9.81,
        imu_acc_noise_sigma: float = 0.2,
        imu_gyro_noise_sigma: float = 0.05,
        imu_integration_sigma: float = 1e-4,
        imu_bias_acc_rw_sigma: float = 1e-3,
        imu_bias_gyro_rw_sigma: float = 1e-4,
        imu_velocity_prior_sigma: float = 1.0,
        imu_bias_prior_sigma: float = 0.1,
    ):
        if not HAS_GTSAM:
            raise RuntimeError("GTSAM not available")
        if not (
            HAS_GTSAM_UNSTABLE
            and hasattr(gtsam_unstable, "IncrementalFixedLagSmoother")
        ):
            raise RuntimeError(
                "gtsam_unstable.IncrementalFixedLagSmoother not available"
            )

        self.fixed_lag_s = lag_s
        self.pose_history_keep = pose_history_keep

        # Noise models (GTSAM ordering: 3 rotation, 3 translation)
        sigmas = np.array(
            [odom_sigma_r] * 3 + [odom_sigma_t] * 3, dtype=np.float64
        )
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self.odom_noise_lost = gtsam.noiseModel.Diagonal.Sigmas(sigmas * lost_scale)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas * prior_scale)

        gt_sigmas = np.array(
            [gt_factor_sigma_r] * 3 + [gt_factor_sigma_t] * 3, dtype=np.float64
        )
        self.gt_factor_noise = gtsam.noiseModel.Diagonal.Sigmas(gt_sigmas)

        loop_sigmas = np.array(
            [loop_sigma_r] * 3 + [loop_sigma_t] * 3, dtype=np.float64
        )
        self.loop_closure_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

        loop_super_sigmas = np.array(
            [loop_super_sigma_r] * 3 + [loop_super_sigma_t] * 3, dtype=np.float64
        )
        self.loop_closure_super_noise = gtsam.noiseModel.Diagonal.Sigmas(
            loop_super_sigmas
        )

        # IMU preintegration capability
        self._has_combined_imu = all(
            hasattr(gtsam, name)
            for name in (
                "CombinedImuFactor",
                "PreintegratedCombinedMeasurements",
                "PreintegrationCombinedParams",
            )
        )
        self._has_basic_imu = all(
            hasattr(gtsam, name)
            for name in ("ImuFactor", "PreintegratedImuMeasurements", "PreintegrationParams")
        )
        self.enable_imu_preintegration = bool(enable_imu_preintegration) and (
            self._has_combined_imu or self._has_basic_imu
        )
        if enable_imu_preintegration and not self.enable_imu_preintegration:
            rospy.logwarn(
                "[smoother] IMU preintegration requested but required GTSAM classes "
                "are unavailable; continuing without IMU factors"
            )

        self.imu_gravity_mps2 = float(imu_gravity_mps2)
        self.imu_acc_noise_sigma = float(imu_acc_noise_sigma)
        self.imu_gyro_noise_sigma = float(imu_gyro_noise_sigma)
        self.imu_integration_sigma = float(imu_integration_sigma)
        self.imu_bias_acc_rw_sigma = float(imu_bias_acc_rw_sigma)
        self.imu_bias_gyro_rw_sigma = float(imu_bias_gyro_rw_sigma)

        self.vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(
            3, float(imu_velocity_prior_sigma)
        )
        self.bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, float(imu_bias_prior_sigma)
        )
        self.bias_rw_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, max(1e-9, float(imu_bias_prior_sigma))
        )

        # GTSAM fixed-lag smoother
        self._fixed_lag = gtsam_unstable.IncrementalFixedLagSmoother(lag_s)
        map_ctor = getattr(gtsam_unstable, "FixedLagSmootherKeyTimestampMap", None)
        if map_ctor is None:
            raise RuntimeError("FixedLagSmootherKeyTimestampMap not available")
        self._map_ctor = map_ctor
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()
        self._new_timestamps = map_ctor()
        self._initialized = False

        # Shared state (protected by graph_lock)
        self.pose = np.eye(4, dtype=np.float64)
        self.pose_by_idx: dict = {0: self.pose.copy()}
        self.velocity_by_idx: dict = {}
        self.bias_by_idx: dict = {}
        self.key_t_sec: dict = {}
        self.latest_key_idx: int = 0
        self.graph_lock = threading.Lock()

        # GTSAM solve thread queue (maxsize=2 keeps latency low)
        self._gtsam_queue: queue.Queue = queue.Queue(maxsize=2)
        self._deferred_batches = 0

        rospy.loginfo(
            f"[smoother] initialized: IncrementalFixedLagSmoother (lag={lag_s:.2f}s, "
            f"imu_preintegration={'on' if self.enable_imu_preintegration else 'off'})"
        )

    # ------------------------------------------------------------------
    # GTSAM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pose3_from_matrix(T: np.ndarray):
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        return gtsam.Pose3(R, t)

    @staticmethod
    def matrix_from_pose3(pose) -> np.ndarray:
        return np.array(pose.matrix(), dtype=np.float64)

    @staticmethod
    def _key(prefix: str, idx: int):
        try:
            return gtsam.symbol(prefix, int(idx))
        except Exception:
            return gtsam.Symbol(prefix, int(idx)).key()

    @staticmethod
    def _key_v(idx: int):
        return FixedLagBackend._key("v", idx)

    @staticmethod
    def _key_b(idx: int):
        return FixedLagBackend._key("b", idx)

    @staticmethod
    def _zero_bias():
        return gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    @staticmethod
    def _safe_call(obj, method_name, *args):
        fn = getattr(obj, method_name, None)
        if callable(fn):
            fn(*args)

    def _build_preintegrator(self, bias):
        """Build a GTSAM preintegrator from configured IMU noise params."""
        acc_var = float(self.imu_acc_noise_sigma ** 2)
        gyro_var = float(self.imu_gyro_noise_sigma ** 2)
        integ_var = float(self.imu_integration_sigma ** 2)
        bias_acc_var = float(self.imu_bias_acc_rw_sigma ** 2)
        bias_gyro_var = float(self.imu_bias_gyro_rw_sigma ** 2)

        if self._has_combined_imu:
            params = gtsam.PreintegrationCombinedParams.MakeSharedU(
                self.imu_gravity_mps2
            )
            self._safe_call(params, "setAccelerometerCovariance", acc_var * np.eye(3))
            self._safe_call(params, "setGyroscopeCovariance", gyro_var * np.eye(3))
            self._safe_call(params, "setIntegrationCovariance", integ_var * np.eye(3))
            self._safe_call(params, "setBiasAccCovariance", bias_acc_var * np.eye(3))
            self._safe_call(params, "setBiasOmegaCovariance", bias_gyro_var * np.eye(3))
            self._safe_call(
                params, "setBiasAccOmegaInit", (1e-5 ** 2) * np.eye(6)
            )
            return gtsam.PreintegratedCombinedMeasurements(params, bias)

        params = gtsam.PreintegrationParams.MakeSharedU(self.imu_gravity_mps2)
        self._safe_call(params, "setAccelerometerCovariance", acc_var * np.eye(3))
        self._safe_call(params, "setGyroscopeCovariance", gyro_var * np.eye(3))
        self._safe_call(params, "setIntegrationCovariance", integ_var * np.eye(3))
        return gtsam.PreintegratedImuMeasurements(params, bias)

    def compute_imu_delta_pose(self, imu_measurements, bias_vec=None):
        """Compute a relative pose from IMU samples using GTSAM preintegration."""
        if not self.enable_imu_preintegration:
            return None
        if imu_measurements is None or len(imu_measurements) == 0:
            return None

        if bias_vec is None:
            b = self._zero_bias()
        else:
            bv = np.asarray(bias_vec, dtype=np.float64).reshape(-1)
            if bv.size >= 6:
                b = gtsam.imuBias.ConstantBias(bv[:3], bv[3:6])
            else:
                b = self._zero_bias()

        preint = self._build_preintegrator(b)
        n_imu = 0
        for dt, acc, gyro in imu_measurements:
            dt_f = float(dt)
            if dt_f <= 0.0:
                continue
            preint.integrateMeasurement(
                np.asarray(acc, dtype=np.float64),
                np.asarray(gyro, dtype=np.float64),
                dt_f,
            )
            n_imu += 1
        if n_imu == 0:
            return None

        # Preferred: use deltaXij() NavState increment when available.
        try:
            if hasattr(preint, "deltaXij"):
                dx = preint.deltaXij()
                if hasattr(dx, "pose"):
                    return self.matrix_from_pose3(dx.pose())
        except Exception:
            pass

        # Fallback: compose from deltaRij()/deltaPij().
        try:
            T = np.eye(4, dtype=np.float64)
            dR = preint.deltaRij()
            dp = preint.deltaPij()
            T[:3, :3] = np.array(dR.matrix(), dtype=np.float64)
            dp_arr = np.array(dp, dtype=np.float64).reshape(-1)
            if dp_arr.size >= 3:
                T[:3, 3] = dp_arr[:3]
            else:
                T[:3, 3] = np.array(
                    [float(dp.x()), float(dp.y()), float(dp.z())], dtype=np.float64
                )
            return T
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, f"[smoother] failed to build IMU delta pose from preintegration: {e}"
            )
            return None

    def _reset_new_data(self):
        self._new_factors.resize(0)
        self._new_values.clear()
        self._new_timestamps.clear()

    def _timestamps_insert(self, key, t_sec: float):
        try:
            self._new_timestamps.insert((key, float(t_sec)))
        except TypeError:
            self._new_timestamps.insert(key, float(t_sec))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stage_factor(self, factor):
        """Add a factor to the pending batch (committed on next add_frame)."""
        self._new_factors.push_back(factor)

    def initialize(self, pose: np.ndarray, stamp) -> bool:
        """Insert X(0) prior and perform first synchronous update."""
        if self._initialized:
            return True
        try:
            with self.graph_lock:
                self.pose = pose.copy()
                key0 = X(0)
                t0 = stamp_to_sec(stamp)
                pose0 = self.pose3_from_matrix(pose)
                self._new_factors.push_back(
                    gtsam.PriorFactorPose3(key0, pose0, self.prior_noise)
                )
                self._new_values.insert(key0, pose0)
                self._timestamps_insert(key0, t0)

                if self.enable_imu_preintegration:
                    k_v0 = self._key_v(0)
                    k_b0 = self._key_b(0)
                    v0 = np.zeros(3, dtype=np.float64)
                    b0 = self._zero_bias()
                    self._new_values.insert(k_v0, v0)
                    self._new_values.insert(k_b0, b0)
                    self._timestamps_insert(k_v0, t0)
                    self._timestamps_insert(k_b0, t0)
                    self._new_factors.push_back(
                        gtsam.PriorFactorVector(k_v0, v0, self.vel_prior_noise)
                    )
                    if hasattr(gtsam, "PriorFactorConstantBias"):
                        self._new_factors.push_back(
                            gtsam.PriorFactorConstantBias(
                                k_b0, b0, self.bias_prior_noise
                            )
                        )

                self._fixed_lag.update(
                    self._new_factors, self._new_values, self._new_timestamps
                )
                self._reset_new_data()
                self.pose_by_idx[0] = pose.copy()
                self.key_t_sec[0] = t0
                if self.enable_imu_preintegration:
                    self.velocity_by_idx[0] = np.zeros(3, dtype=np.float64)
                    self.bias_by_idx[0] = np.zeros(6, dtype=np.float64)
            self._initialized = True
            rospy.loginfo("[smoother] fixed-lag smoother initialized with prior X(0)")
            return True
        except Exception as e:
            rospy.logwarn(f"[smoother] init failed: {e}")
            return False

    def add_frame(
        self,
        prev_idx: int,
        curr_idx: int,
        stamp,
        predicted_pose: np.ndarray,
        gt_rel_mat: np.ndarray = None,
        imu_measurements=None,
    ) -> bool:
        """Add one frame to the smoother with odometry and optional IMU factor."""
        if not self.initialize(predicted_pose, stamp):
            return False
        if prev_idx < 0:
            self.key_t_sec[curr_idx] = stamp_to_sec(stamp)
            return True

        t_sec = stamp_to_sec(stamp)
        key_prev = X(prev_idx)
        key_curr = X(curr_idx)

        if gt_rel_mat is not None:
            # Accurate between-factor from odometry / GT measurement
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    key_prev,
                    key_curr,
                    self.pose3_from_matrix(gt_rel_mat),
                    self.gt_factor_noise,
                )
            )
        else:
            # No odometry available: identity with inflated noise
            self._new_factors.push_back(
                gtsam.BetweenFactorPose3(
                    key_prev,
                    key_curr,
                    gtsam.Pose3.Identity(),
                    self.odom_noise_lost,
                )
            )

        self._new_values.insert(key_curr, self.pose3_from_matrix(predicted_pose))
        self._timestamps_insert(key_curr, t_sec)
        self.key_t_sec[curr_idx] = t_sec

        # Optional IMU preintegration factor
        if self.enable_imu_preintegration:
            key_v_prev = self._key_v(prev_idx)
            key_v_curr = self._key_v(curr_idx)
            key_b_prev = self._key_b(prev_idx)
            key_b_curr = self._key_b(curr_idx)

            prev_bias_vec = self.bias_by_idx.get(prev_idx, np.zeros(6, dtype=np.float64))
            prev_bias = gtsam.imuBias.ConstantBias(
                prev_bias_vec[:3], prev_bias_vec[3:]
            )

            preint = self._build_preintegrator(prev_bias)
            n_imu = 0
            if imu_measurements is not None:
                for dt, acc, gyro in imu_measurements:
                    dt_f = float(dt)
                    if dt_f <= 0.0:
                        continue
                    preint.integrateMeasurement(
                        np.asarray(acc, dtype=np.float64),
                        np.asarray(gyro, dtype=np.float64),
                        dt_f,
                    )
                    n_imu += 1

            if n_imu > 0:
                if self._has_combined_imu:
                    self._new_factors.push_back(
                        gtsam.CombinedImuFactor(
                            key_prev,
                            key_v_prev,
                            key_curr,
                            key_v_curr,
                            key_b_prev,
                            key_b_curr,
                            preint,
                        )
                    )
                else:
                    self._new_factors.push_back(
                        gtsam.ImuFactor(
                            key_prev,
                            key_v_prev,
                            key_curr,
                            key_v_curr,
                            key_b_prev,
                            preint,
                        )
                    )
                    if hasattr(gtsam, "BetweenFactorConstantBias"):
                        self._new_factors.push_back(
                            gtsam.BetweenFactorConstantBias(
                                key_b_prev,
                                key_b_curr,
                                self._zero_bias(),
                                self.bias_rw_noise,
                            )
                        )

            # Initial guesses for V/B
            t_prev = self.key_t_sec.get(prev_idx, t_sec - 1e-3)
            dt_pose = max(1e-3, float(t_sec - t_prev))
            T_prev = self.pose_by_idx.get(prev_idx)
            if T_prev is not None:
                vel_guess = (predicted_pose[:3, 3] - T_prev[:3, 3]) / dt_pose
            else:
                vel_guess = self.velocity_by_idx.get(prev_idx, np.zeros(3, dtype=np.float64))
            self._new_values.insert(key_v_curr, np.asarray(vel_guess, dtype=np.float64))
            self._new_values.insert(key_b_curr, prev_bias)
            self._timestamps_insert(key_v_curr, t_sec)
            self._timestamps_insert(key_b_curr, t_sec)
            self.velocity_by_idx[curr_idx] = np.asarray(vel_guess, dtype=np.float64)
            self.bias_by_idx[curr_idx] = prev_bias_vec.copy()

            rospy.logdebug(
                f"[smoother] IMU preintegration at X({curr_idx}): samples={n_imu}"
            )

        # Snapshot containers and hand off to GTSAM thread
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
            # Do not drop: keep staged so it batches with the next frame.
            self._new_factors = factors_snap
            self._new_values = values_snap
            self._new_timestamps = timestamps_snap
            self._deferred_batches += 1

        with self.graph_lock:
            self.pose = predicted_pose.copy()
            self.latest_key_idx = curr_idx
            self.pose_by_idx[curr_idx] = predicted_pose.copy()
        return True

    def is_in_lag_window(self, idx: int) -> bool:
        t_idx = self.key_t_sec.get(idx)
        if t_idx is None:
            return False
        t_latest = self.key_t_sec.get(self.latest_key_idx, 0.0)
        return (t_latest - t_idx) <= self.fixed_lag_s * 1.1

    # ------------------------------------------------------------------
    # GTSAM solve thread
    # ------------------------------------------------------------------

    def backend_loop(self):
        """Dedicated GTSAM solve thread target."""
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
                new_pose = self.matrix_from_pose3(estimate.atPose3(X(curr_idx)))
                with self.graph_lock:
                    latest_t = self.key_t_sec.get(curr_idx, None)
                    for k_idx, k_t in list(self.key_t_sec.items()):
                        if (
                            latest_t is not None
                            and (latest_t - k_t) > self.fixed_lag_s * 1.2
                        ):
                            continue
                        try:
                            self.pose_by_idx[k_idx] = self.matrix_from_pose3(
                                estimate.atPose3(X(k_idx))
                            )
                        except Exception:
                            pass
                        if self.enable_imu_preintegration:
                            try:
                                self.velocity_by_idx[k_idx] = np.array(
                                    estimate.atVector(self._key_v(k_idx)),
                                    dtype=np.float64,
                                )
                            except Exception:
                                pass
                            try:
                                b = estimate.atConstantBias(self._key_b(k_idx))
                                self.bias_by_idx[k_idx] = np.hstack(
                                    [
                                        np.array(b.accelerometer(), dtype=np.float64),
                                        np.array(b.gyroscope(), dtype=np.float64),
                                    ]
                                )
                            except Exception:
                                pass
                    self.pose = new_pose
                    self.latest_key_idx = curr_idx
                    self.pose_by_idx[curr_idx] = new_pose.copy()
                    stale = [
                        k
                        for k in self.pose_by_idx
                        if k < (curr_idx - self.pose_history_keep)
                    ]
                    for k in stale:
                        del self.pose_by_idx[k]
                        if k in self.velocity_by_idx:
                            del self.velocity_by_idx[k]
                        if k in self.bias_by_idx:
                            del self.bias_by_idx[k]
                rospy.logdebug(f"[smoother] GTSAM backend solved at X({curr_idx})")
            except Exception as e:
                rospy.logwarn(
                    f"[smoother] GTSAM backend failed at X({curr_idx}): {e}"
                )
