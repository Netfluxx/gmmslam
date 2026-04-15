#include "gmmslam/config.hpp"
#include "gmmslam/d2d_registration.hpp"

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nlohmann/json.hpp>

#include <Eigen/Core>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

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

        result_pub_ = nh.advertise<std_msgs::String>(result_topic, 50);

        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&D2DRegistrationNode::workerLoop, this);
        }

        request_sub_ = nh.subscribe(request_topic, 200,
                                    &D2DRegistrationNode::requestCallback, this);

        ROS_INFO("[d2d_reg] ready | workers=%d | %s -> %s",
                 num_workers_, request_topic.c_str(), result_topic.c_str());
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

    void requestCallback(const std_msgs::String::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if (task_queue_.size() >= kMaxQueueSize) {
            ROS_WARN_THROTTLE(5.0, "[d2d_reg] task queue full (%zu), dropping request",
                              task_queue_.size());
            return;
        }
        task_queue_.push(msg->data);
        queue_cv_.notify_one();
    }

    void workerLoop() {
        while (true) {
            std::string payload;
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] { return !task_queue_.empty() || !running_; });
                if (!running_ && task_queue_.empty()) return;
                payload = std::move(task_queue_.front());
                task_queue_.pop();
            }
            processRequest(payload);
        }
    }

    void processRequest(const std::string& payload_str) {
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

            // Row-major identity flattened: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> I_row(
                Eigen::Matrix4d::Identity());
            const std::vector<double> flat_identity(I_row.data(), I_row.data() + 16);

            result = {
                {"prev_idx", prev_idx},
                {"curr_idx", curr_idx},
                {"stamp", stamp},
                {"success", false},
                {"score", -std::numeric_limits<double>::infinity()},
                {"transform", flat_identity},
                {"is_loop_closure", is_loop},
                {"is_submap_registration", is_submap}
            };

            if (!std::filesystem::exists(source_path) ||
                !std::filesystem::exists(target_path)) {
                publishResult(result);
                return;
            }

            // Step 1: isoplanar registration
            const Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
            auto iso_result =
                gmmslam::isoplanarRegistration(T_init, source_path, target_path);
            Eigen::Matrix4f T_iso = iso_result.transform;
            if (!std::isfinite(iso_result.score) || T_iso.hasNaN()) {
                T_iso = T_init;
            }

            // Step 2: anisotropic registration
            const auto aniso_result =
                gmmslam::anisotropicRegistration(T_iso, source_path, target_path);
            const Eigen::Matrix4f T_final = aniso_result.transform;
            const float score_final = aniso_result.score;

            if (!std::isfinite(score_final) || T_final.hasNaN()) {
                publishResult(result);
                return;
            }
            if (score_final < score_threshold_) {
                publishResult(result);
                return;
            }

            result["success"] = true;
            result["score"] = static_cast<double>(score_final);

            // Flatten in row-major order to match Python numpy convention
            const Eigen::Matrix4d T_final_d = T_final.cast<double>();
            const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row(T_final_d);
            const std::vector<double> flat_T(T_row.data(), T_row.data() + 16);
            result["transform"] = flat_T;

            const std::string tag = is_submap ? "submap" : "keyframe";
            if (is_submap) {
                ROS_INFO("[d2d_reg] %s registration success | prev=%d curr=%d "
                         "score=%.4f",
                         tag.c_str(), prev_idx, curr_idx,
                         static_cast<double>(score_final));
            } else {
                ROS_INFO_THROTTLE(
                    5.0,
                    "[d2d_reg] %s registration success | prev=%d curr=%d "
                    "score=%.4f",
                    tag.c_str(), prev_idx, curr_idx,
                    static_cast<double>(score_final));
            }
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[d2d_reg] registration failed: %s", e.what());
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

    bool running_ = true;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::string> task_queue_;
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
