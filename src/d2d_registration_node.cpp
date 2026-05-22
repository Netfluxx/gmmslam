#include "gmmslam/config.hpp"
#include "gmmslam/d2d_registration.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/String.h>
#include <nlohmann/json.hpp>

#include <Eigen/Core>
#include <Eigen/LU>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

void applyDebugPrints(bool enable) {
    if (!enable) {
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                       ros::console::levels::Warn);
        ros::console::notifyLoggerLevelsChanged();
    }
}

} // namespace

class D2DRegistrationNode {
public:
    D2DRegistrationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        std::string request_topic, result_topic;
        pnh.param<std::string>("request_topic", request_topic,
                               "/gmmslam_node/registration/request");
        pnh.param<std::string>("result_topic", result_topic,
                               "/gmmslam_node/registration/result");
        pnh.param<int>("num_workers", num_workers_, 2);
        num_workers_ = std::max(1, num_workers_);
        pnh.param<double>("score_threshold", score_threshold_, -1.0e9);
        pnh.param<bool>("suppress_backend_output", suppress_backend_output_, true);
        pnh.param<double>("drop_stale_keyframe_age_s",
                          drop_stale_keyframe_age_s_, 2.0);
        nh.param("/gmmslam/DEBUG_PRINTS", debug_prints_, debug_prints_);
        pnh.param<bool>("debug_prints", debug_prints_, debug_prints_);
        applyDebugPrints(debug_prints_);

        result_pub_ = nh.advertise<std_msgs::String>(result_topic, 50);

        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&D2DRegistrationNode::workerLoop, this);
        }

        request_sub_ = nh.subscribe(request_topic, 200,
                                    &D2DRegistrationNode::requestCallback, this);

        ROS_INFO("[d2d_reg] ready | workers=%d | %s -> %s | "
                 "suppress_backend_output=%s | drop_stale_keyframe_age_s=%.2f | "
                 "debug_prints=%s",
                 num_workers_, request_topic.c_str(), result_topic.c_str(),
                 suppress_backend_output_ ? "true" : "false",
                 drop_stale_keyframe_age_s_,
                 debug_prints_ ? "true" : "false");
    }

    ~D2DRegistrationNode() {
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
            running_ = false;
        }
        queue_cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

private:
    static constexpr std::size_t kMaxQueueSize = 64;

    class ScopedBackendOutputSilencer {
    public:
        explicit ScopedBackendOutputSilencer(bool enable) {
            if (!enable) return;
            std::fflush(stdout);
            std::fflush(stderr);
            saved_stdout_ = ::dup(STDOUT_FILENO);
            saved_stderr_ = ::dup(STDERR_FILENO);
            null_fd_ = ::open("/dev/null", O_WRONLY);
            if (saved_stdout_ < 0 || saved_stderr_ < 0 || null_fd_ < 0) {
                cleanup();
                return;
            }
            ::dup2(null_fd_, STDOUT_FILENO);
            ::dup2(null_fd_, STDERR_FILENO);
            active_ = true;
        }

        ~ScopedBackendOutputSilencer() {
            if (active_) {
                std::fflush(stdout);
                std::fflush(stderr);
                ::dup2(saved_stdout_, STDOUT_FILENO);
                ::dup2(saved_stderr_, STDERR_FILENO);
            }
            cleanup();
        }

        ScopedBackendOutputSilencer(const ScopedBackendOutputSilencer&) = delete;
        ScopedBackendOutputSilencer& operator=(const ScopedBackendOutputSilencer&) = delete;

    private:
        void cleanup() {
            if (null_fd_ >= 0) ::close(null_fd_);
            if (saved_stdout_ >= 0) ::close(saved_stdout_);
            if (saved_stderr_ >= 0) ::close(saved_stderr_);
            null_fd_ = saved_stdout_ = saved_stderr_ = -1;
            active_ = false;
        }

        int saved_stdout_ = -1;
        int saved_stderr_ = -1;
        int null_fd_ = -1;
        bool active_ = false;
    };

    std::size_t queuedTaskCount() const {
        return loop_queue_.size() + submap_queue_.size() + sequential_queue_.size();
    }

    static bool requestKind(const std::string& payload,
                            bool& is_loop,
                            bool& is_submap,
                            int& prev_idx,
                            int& curr_idx,
                            double& stamp) {
        try {
            const auto req = json::parse(payload);
            is_loop = req.value("is_loop_closure", false);
            is_submap = req.value("is_submap_registration", false);
            prev_idx = req.value("prev_idx", -1);
            curr_idx = req.value("curr_idx", -1);
            stamp = req.value("stamp", 0.0);
            return true;
        } catch (const std::exception&) {
            is_loop = false;
            is_submap = false;
            prev_idx = -1;
            curr_idx = -1;
            stamp = 0.0;
            return false;
        }
    }

    void requestCallback(const std_msgs::String::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        bool is_loop = false;
        bool is_submap = false;
        int prev_idx = -1;
        int curr_idx = -1;
        double stamp = 0.0;
        requestKind(msg->data, is_loop, is_submap, prev_idx, curr_idx, stamp);

        if (!is_loop && !is_submap && drop_stale_keyframe_age_s_ > 0.0 &&
            std::isfinite(stamp) && stamp > 0.0) {
            const double age_s = ros::Time::now().toSec() - stamp;
            if (std::isfinite(age_s) && age_s > drop_stale_keyframe_age_s_) {
                ROS_INFO_THROTTLE(
                    2.0,
                    "[d2d_reg] dropping stale keyframe registration at enqueue "
                    "prev=%d curr=%d age=%.2fs > %.2fs",
                    prev_idx, curr_idx, age_s, drop_stale_keyframe_age_s_);
                return;
            }
        }

        if (queuedTaskCount() >= kMaxQueueSize) {
            if (!sequential_queue_.empty()) {
                sequential_queue_.pop_back();
                ROS_WARN_THROTTLE(
                    5.0,
                    "[d2d_reg] task queue full; dropped oldest sequential job "
                    "to admit %s request",
                    requestTag(is_loop, is_submap).c_str());
            } else if (is_loop && !submap_queue_.empty()) {
                submap_queue_.pop_back();
                ROS_WARN_THROTTLE(
                    5.0,
                    "[d2d_reg] task queue full; dropped oldest submap job "
                    "to admit loop request");
            } else {
                ROS_WARN_THROTTLE(
                    5.0,
                    "[d2d_reg] task queue full (%zu), dropping %s request",
                    queuedTaskCount(), requestTag(is_loop, is_submap).c_str());
                return;
            }
        }

        if (is_loop) {
            loop_queue_.push_back(msg->data);
        } else if (is_submap) {
            submap_queue_.push_back(msg->data);
        } else {
            sequential_queue_.push_back(msg->data);
        }
        queue_cv_.notify_one();
    }

    bool popNextTask(std::string& payload) {
        if (!loop_queue_.empty()) {
            payload = std::move(loop_queue_.front());
            loop_queue_.pop_front();
            return true;
        }
        if (!submap_queue_.empty()) {
            payload = std::move(submap_queue_.front());
            submap_queue_.pop_front();
            return true;
        }
        if (!sequential_queue_.empty()) {
            payload = std::move(sequential_queue_.front());
            sequential_queue_.pop_front();
            return true;
        }
        return false;
    }

    bool hasQueuedTask() const {
        return !loop_queue_.empty() || !submap_queue_.empty() ||
               !sequential_queue_.empty();
    }

    void workerLoop() {
        while (true) {
            std::string payload;
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] { return hasQueuedTask() || !running_; });
                if (!running_ && !hasQueuedTask()) return;
                if (!popNextTask(payload)) {
                    continue;
                }
            }
            processRequest(payload);
        }
    }

    static std::string requestTag(bool is_loop, bool is_submap) {
        if (is_submap) return "submap";
        if (is_loop) return "loop_keyframe";
        return "sequential_keyframe";
    }

    static double rotationAngleDeg(const Eigen::Matrix3f& R) {
        const float c = std::max(-1.0f, std::min(1.0f, (R.trace() - 1.0f) * 0.5f));
        return std::acos(static_cast<double>(c)) * 180.0 / std::acos(-1.0);
    }

    void processRequest(const std::string& payload_str) {
        const auto request_t0 = std::chrono::steady_clock::now();
        json result;
        try {
            const auto req = json::parse(payload_str);
            const int prev_idx = req["prev_idx"].get<int>();
            const int curr_idx = req["curr_idx"].get<int>();
            const double stamp = req.value("stamp", 0.0);
            const std::string source_path = req["source_path"].get<std::string>();
            const std::string target_path = req["target_path"].get<std::string>();
            const bool is_loop = req.value("is_loop_closure", false);
            const bool is_submap = req.value("is_submap_registration", false);
            ROS_INFO("[d2d_reg] %s request start | prev=%d curr=%d",
                     requestTag(is_loop, is_submap).c_str(), prev_idx, curr_idx);

            if (!is_loop && !is_submap && drop_stale_keyframe_age_s_ > 0.0 &&
                std::isfinite(stamp) && stamp > 0.0) {
                const double age_s = ros::Time::now().toSec() - stamp;
                if (std::isfinite(age_s) && age_s > drop_stale_keyframe_age_s_) {
                    ROS_INFO_THROTTLE(
                        2.0,
                        "[d2d_reg] dropping stale keyframe registration "
                        "prev=%d curr=%d age=%.2fs > %.2fs",
                        prev_idx, curr_idx, age_s, drop_stale_keyframe_age_s_);
                    return;
                }
            }

            // Row-major identity flattened: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> I_row(
                Eigen::Matrix4d::Identity());
            const std::vector<double> flat_identity(I_row.data(), I_row.data() + 16);

            // NOTE: finite sentinel — nlohmann::json serializes non-finite
            // doubles (NaN, ±Inf) as JSON null, which then throws
            // type_error.302 on the receiver when read as a number.
            result = {
                {"prev_idx", prev_idx},
                {"curr_idx", curr_idx},
                {"stamp", stamp},
                {"success", false},
                {"score", -1.0e30},
                {"failure_reason", "not_evaluated"},
                {"transform", flat_identity},
                {"is_loop_closure", is_loop},
                {"is_submap_registration", is_submap}
            };

            auto fail = [&](const std::string& reason) {
                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_t0).count();
                result["failure_reason"] = reason;
                result["elapsed_ms"] = static_cast<long>(elapsed_ms);
                ROS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=%s elapsed_ms=%ld",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, reason.c_str(),
                         static_cast<long>(elapsed_ms));
                publishResult(result);
            };

            if (!std::filesystem::exists(source_path) ||
                !std::filesystem::exists(target_path)) {
                ROS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=missing_gmm_file source_exists=%d target_exists=%d",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx,
                         std::filesystem::exists(source_path) ? 1 : 0,
                         std::filesystem::exists(target_path) ? 1 : 0);
                fail("missing_gmm_file");
                return;
            }

            // Step 1: isoplanar registration. Default T_init is identity;
            // loop-closure requests may attach an `initial_transform` (e.g.
            // a SOLiD-derived yaw prior) to improve the convergence basin.
            Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
            bool has_initial_transform = false;
            if (req.contains("initial_transform")) {
                const auto& arr = req.at("initial_transform");
                if (arr.is_array() && arr.size() == 16) {
                    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_row;
                    bool ok = true;
                    for (int k = 0; k < 16 && ok; ++k) {
                        if (!arr[k].is_number()) { ok = false; break; }
                        T_row.data()[k] = arr[k].get<float>();
                    }
                    if (ok && T_row.allFinite()) {
                        T_init = Eigen::Matrix4f(T_row);
                        has_initial_transform = true;
                        result["initial_transform"] = arr;
                    }
                }
            }

            gmmslam::RegistrationResult iso_result;
            gmmslam::RegistrationResult aniso_result;
            Eigen::Matrix4f T_iso = T_init;
            if (has_initial_transform) {
                // The isoplanar pre-pass is fragile for turn-in-place / planar
                // views and can produce NaNs even with a good pose prior. When
                // the caller already provides a full SE(3) initialization, use
                // it directly for anisotropic D2D.
                ROS_INFO_THROTTLE(
                    2.0,
                    "[d2d_reg] %s skipping isoplanar pre-pass; using provided initial_transform",
                    requestTag(is_loop, is_submap).c_str());
            } else if (suppress_backend_output_) {
                // The linked GIRA3D registration library prints covariance inversion
                // diagnostics directly to stdout/stderr. File-descriptor redirection
                // is process-wide, so serialize backend calls while silencing is on.
                std::lock_guard<std::mutex> lk(backend_call_mutex_);
                ScopedBackendOutputSilencer silence(true);
                iso_result =
                    gmmslam::isoplanarRegistration(T_init, source_path, target_path);
            } else {
                iso_result =
                    gmmslam::isoplanarRegistration(T_init, source_path, target_path);
            }
            if (!has_initial_transform) {
                T_iso = iso_result.transform;
            }
            if (!has_initial_transform &&
                (!std::isfinite(iso_result.score) || T_iso.hasNaN())) {
                ROS_INFO("[d2d_reg] %s isoplanar stage invalid | prev=%d curr=%d "
                         "score=%.4f; falling back to initial transform",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx,
                         static_cast<double>(iso_result.score));
                T_iso = T_init;
            }

            // Step 2: anisotropic registration
            if (suppress_backend_output_) {
                std::lock_guard<std::mutex> lk(backend_call_mutex_);
                ScopedBackendOutputSilencer silence(true);
                aniso_result =
                    gmmslam::anisotropicRegistration(T_iso, source_path, target_path);
            } else {
                aniso_result =
                    gmmslam::anisotropicRegistration(T_iso, source_path, target_path);
            }
            const Eigen::Matrix4f T_final = aniso_result.transform;
            const float score_final = aniso_result.score;

            if (!std::isfinite(score_final) || T_final.hasNaN()) {
                ROS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=invalid_anisotropic_result score=%.4f has_nan=%d",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, static_cast<double>(score_final),
                         T_final.hasNaN() ? 1 : 0);
                fail("invalid_anisotropic_result");
                return;
            }
            if (has_initial_transform && !is_submap) {
                const Eigen::Matrix4f T_err = T_init.inverse() * T_final;
                const double trans_err =
                    T_err.block<3, 1>(0, 3).cast<double>().norm();
                const double rot_err_deg =
                    rotationAngleDeg(T_err.block<3, 3>(0, 0));
                const double trans_limit_m = is_loop ? 3.0 : 0.50;
                const double rot_limit_deg = is_loop ? 45.0 : 15.0;
                if (trans_err > trans_limit_m || rot_err_deg > rot_limit_deg) {
                    ROS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                             "reason=prior_consistency_gate trans=%.3fm rot=%.2fdeg "
                             "limits=%.3fm/%.2fdeg score=%.4f",
                             requestTag(is_loop, is_submap).c_str(),
                             prev_idx, curr_idx, trans_err, rot_err_deg,
                             trans_limit_m, rot_limit_deg,
                             static_cast<double>(score_final));
                    fail("prior_consistency_gate");
                    return;
                }
            }
            if (score_final < score_threshold_) {
                ROS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=worker_score_threshold score=%.4f < %.4f",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, static_cast<double>(score_final),
                         score_threshold_);
                fail("worker_score_threshold");
                return;
            }

            result["success"] = true;
            result.erase("failure_reason");
            result["score"] = static_cast<double>(score_final);

            // Flatten in row-major order to match Python numpy convention
            const Eigen::Matrix4d T_final_d = T_final.cast<double>();
            const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row(T_final_d);
            const std::vector<double> flat_T(T_row.data(), T_row.data() + 16);
            result["transform"] = flat_T;

            const std::string tag = requestTag(is_loop, is_submap);
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - request_t0).count();
            if (is_submap) {
                ROS_INFO("[d2d_reg] %s registration success | prev=%d curr=%d "
                         "score=%.4f elapsed_ms=%ld",
                         tag.c_str(), prev_idx, curr_idx,
                         static_cast<double>(score_final),
                         static_cast<long>(elapsed_ms));
            } else {
                ROS_INFO_THROTTLE(
                    5.0,
                    "[d2d_reg] %s registration success | prev=%d curr=%d "
                    "score=%.4f elapsed_ms=%ld",
                    tag.c_str(), prev_idx, curr_idx,
                    static_cast<double>(score_final),
                    static_cast<long>(elapsed_ms));
            }
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[d2d_reg] registration failed: %s", e.what());
            if (!result.empty()) {
                result["failure_reason"] = "exception";
                result["failure_detail"] = e.what();
            }
            if (result.empty()) return;
        }
        publishResult(result);
    }

    void publishResult(const json& result) {
        std_msgs::String msg;
        msg.data = result.dump();
        result_pub_.publish(msg);
    }

    ros::Subscriber request_sub_;
    ros::Publisher result_pub_;
    int num_workers_ = 2;
    double score_threshold_ = -1.0e9;
    bool suppress_backend_output_ = true;
    double drop_stale_keyframe_age_s_ = 2.0;
    bool debug_prints_ = true;

    bool running_ = true;
    std::mutex queue_mutex_;
    std::mutex backend_call_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::string> loop_queue_;
    std::deque<std::string> submap_queue_;
    std::deque<std::string> sequential_queue_;
    std::vector<std::thread> workers_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "d2d_registration_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    D2DRegistrationNode node(nh, pnh);
    ros::spin();
    return 0;
}
