#!/usr/bin/env python3
"""Async D2D registration worker node.

Consumes registration requests and publishes results as JSON over std_msgs/String.
Designed to run in a separate process/core from gmmslam.py.
"""

import json
import os
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rospy
from std_msgs.msg import String

import sys
import glob


def _inject_venv(venv_root: str):
    pattern = os.path.join(venv_root, "lib", "python3*", "site-packages")
    for sp in glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)


_GIRA_WS = os.environ.get("GIRA_WS", "/root/gira_ws")
_REGISTRATION_VENV = os.path.join(_GIRA_WS, "gira3d-registration", ".venv")
_inject_venv(_REGISTRATION_VENV)

try:
    import gmm_d2d_registration_py

    HAS_GMM_REGISTRATION = True
except ImportError:
    HAS_GMM_REGISTRATION = False
    gmm_d2d_registration_py = None


@contextmanager
def _silence_process_output(enabled: bool):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(null_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


class D2DRegistrationNode:
    def __init__(self):
        rospy.init_node("d2d_registration_node", anonymous=False)
        self.request_topic = rospy.get_param(
            "~request_topic", "/gmmslam_node/registration/request"
        )
        self.result_topic = rospy.get_param(
            "~result_topic", "/gmmslam_node/registration/result"
        )
        self.num_workers = max(1, int(rospy.get_param("~num_workers", 2)))
        self.score_threshold = float(rospy.get_param("~score_threshold", -1.0e9))
        self.suppress_backend_output = bool(
            rospy.get_param("~suppress_backend_output", True)
        )
        self.backend_io_lock = threading.Lock()

        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=50)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        rospy.Subscriber(
            self.request_topic, String, self._request_callback, queue_size=200
        )
        rospy.loginfo(
            f"[d2d_reg] ready | workers={self.num_workers} | "
            f"{self.request_topic} -> {self.result_topic} | "
            f"suppress_backend_output={self.suppress_backend_output}"
        )

    def _request_callback(self, msg: String):
        self.executor.submit(self._process_request, msg.data)

    def _process_request(self, payload_str: str):
        try:
            req = json.loads(payload_str)
            prev_idx = int(req["prev_idx"])
            curr_idx = int(req["curr_idx"])
            stamp = float(req.get("stamp", 0.0))
            source_path = str(req["source_path"])
            target_path = str(req["target_path"])
            is_loop_closure = bool(req.get("is_loop_closure", False))
            is_submap_registration = bool(req.get("is_submap_registration", False))
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[d2d_reg] malformed request: {e}")
            return

        result = {
            "prev_idx": prev_idx,
            "curr_idx": curr_idx,
            "stamp": stamp,
            "success": False,
            "score": float("-inf"),
            "transform": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
            "is_loop_closure": is_loop_closure,
            "is_submap_registration": is_submap_registration,
        }

        if not HAS_GMM_REGISTRATION:
            self.result_pub.publish(String(data=json.dumps(result)))
            return

        if not os.path.exists(source_path) or not os.path.exists(target_path):
            self.result_pub.publish(String(data=json.dumps(result)))
            return

        try:
            T_init = np.eye(4, dtype=np.float64)
            if self.suppress_backend_output:
                # The linked GIRA3D backend prints covariance inversion diagnostics
                # directly to stdout/stderr. Redirection is process-wide, so guard it.
                with self.backend_io_lock, _silence_process_output(True):
                    result_iso = gmm_d2d_registration_py.isoplanar_registration(
                        T_init, source_path, target_path
                    )
            else:
                result_iso = gmm_d2d_registration_py.isoplanar_registration(
                    T_init, source_path, target_path
                )
            T_iso = result_iso[0]
            score_iso = result_iso[1]
            if (
                np.isnan(score_iso)
                or np.isinf(score_iso)
                or np.any(np.isnan(T_iso))
                or np.any(np.isinf(T_iso))
            ):
                T_iso = T_init

            if self.suppress_backend_output:
                with self.backend_io_lock, _silence_process_output(True):
                    result_aniso = gmm_d2d_registration_py.anisotropic_registration(
                        T_iso, source_path, target_path
                    )
            else:
                result_aniso = gmm_d2d_registration_py.anisotropic_registration(
                    T_iso, source_path, target_path
                )
            T_final = np.array(result_aniso[0], dtype=np.float64)
            score_final = float(result_aniso[1])
            if (
                np.isnan(score_final)
                or np.isinf(score_final)
                or np.any(np.isnan(T_final))
                or np.any(np.isinf(T_final))
            ):
                self.result_pub.publish(String(data=json.dumps(result)))
                return

            if score_final < self.score_threshold:
                self.result_pub.publish(String(data=json.dumps(result)))
                return

            result["success"] = True
            result["score"] = score_final
            result["transform"] = T_final.reshape(-1).tolist()
            # Echo SOLiD / loop metadata from the request for downstream viz & logging.
            for key in (
                "solid_cos_sim",
                "solid_rescue",
                "solid_yaw_deg",
                "solid_yaw_overlap",
            ):
                if key in req:
                    result[key] = req[key]

            tag = "submap" if is_submap_registration else "keyframe"
            if is_submap_registration:
                rospy.loginfo(
                    f"[d2d_reg] {tag} registration success | "
                    f"prev={prev_idx} curr={curr_idx} score={score_final:.4f}"
                )
            else:
                rospy.loginfo_throttle(
                    5.0,
                    f"[d2d_reg] {tag} registration success | "
                    f"prev={prev_idx} curr={curr_idx} score={score_final:.4f}",
                )
            self.result_pub.publish(String(data=json.dumps(result)))
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[d2d_reg] registration failed: {e}")
            self.result_pub.publish(String(data=json.dumps(result)))


def main():
    node = D2DRegistrationNode()
    rospy.spin()


if __name__ == "__main__":
    main()
