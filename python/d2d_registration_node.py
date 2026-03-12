#!/usr/bin/env python3
"""Async D2D registration worker node.

Consumes registration requests and publishes results as JSON over std_msgs/String.
Designed to run in a separate process/core from gmmslam.py.
"""

import json
import os
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

        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=50)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        rospy.Subscriber(
            self.request_topic, String, self._request_callback, queue_size=200
        )
        rospy.loginfo(
            f"[d2d_reg] ready | workers={self.num_workers} | {self.request_topic} -> {self.result_topic}"
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
        }

        if not HAS_GMM_REGISTRATION:
            self.result_pub.publish(String(data=json.dumps(result)))
            return

        if not os.path.exists(source_path) or not os.path.exists(target_path):
            self.result_pub.publish(String(data=json.dumps(result)))
            return

        try:
            T_init = np.eye(4, dtype=np.float64)
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

            rospy.loginfo_throttle(
                5.0,
                f"[d2d_reg] registration success | prev={prev_idx} curr={curr_idx} score={score_final:.4f}",
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
