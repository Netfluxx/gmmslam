#include "gmmslam/config.hpp"
#include "gmmslam/rclcpp_logging.hpp"
#include "gmmslam/d2d_registration.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <rcutils/logging.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

void applyDebugPrints(bool enable) {
    if (!enable) {
        rcutils_logging_set_logger_level(
            "gmmslam", RCUTILS_LOG_SEVERITY_WARN);
    }
}

const char* d2dSuccessColor(bool is_loop, bool is_submap) {
    if (is_loop) return "\033[31m";      // red
    if (is_submap) return "\033[38;5;208m";  // orange
    return "\033[37m";                   // white
}

constexpr const char* kAnsiReset = "\033[0m";

class NodeParams {
public:
    explicit NodeParams(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {}

    template <typename T>
    void param(const std::string& key, T& value, const T& default_value) const {
        value = get<T>(key, default_value);
    }

    template <typename T>
    bool getParam(const std::string& key, T& value) const {
        if (!node_) return false;
        if (node_->has_parameter(key)) {
            return node_->get_parameter(key, value);
        }
        const std::string dotted = dottedName(key);
        if (dotted != key && node_->has_parameter(dotted)) {
            return node_->get_parameter(dotted, value);
        }
        return false;
    }

private:
    static std::string dottedName(std::string key) {
        while (!key.empty() && key.front() == '/') {
            key.erase(key.begin());
        }
        for (char& ch : key) {
            if (ch == '/') ch = '.';
        }
        return key;
    }

    template <typename T>
    T get(const std::string& key, const T& fallback) const {
        T value = fallback;
        if (getParam(key, value)) {
            return value;
        }
        if (node_ && !node_->has_parameter(key)) {
            node_->declare_parameter<T>(key, fallback);
        }
        return value;
    }

    rclcpp::Node::SharedPtr node_;
};

rclcpp::QoS reliableQos(std::size_t depth) {
    return rclcpp::QoS(rclcpp::KeepLast(depth)).reliable();
}

template <typename T>
void readYamlValue(const YAML::Node& root,
                   std::initializer_list<const char*> path,
                   T& value) {
    YAML::Node node = root;
    for (const char* key : path) {
        if (!node || !node[key]) {
            return;
        }
        node = node[key];
    }
    try {
        value = node.as<T>();
    } catch (const YAML::Exception&) {
    }
}

} // namespace

class D2DRegistrationNode {
public:
    explicit D2DRegistrationNode(rclcpp::Node::SharedPtr node)
        : node_(std::move(node)) {
        NodeParams pnh(node_);
        std::string request_topic, result_topic;
        std::string config_path;
        pnh.param<std::string>("config_file", config_path, "");
        gmmslam::Config cfg;
        YAML::Node yaml_root;
        if (!config_path.empty()) {
            cfg = gmmslam::loadConfig(config_path);
            try {
                yaml_root = YAML::LoadFile(config_path);
            } catch (const YAML::Exception& e) {
                GMS_WARN("[d2d_reg] failed to parse config_file for D2D-only "
                         "options: %s", e.what());
            }
        }

        pnh.param<std::string>("request_topic", request_topic,
                               cfg.ros.registration_request_topic);
        pnh.param<std::string>("result_topic", result_topic,
                               cfg.ros.registration_result_topic);
        pnh.param<int>("num_workers", num_workers_, cfg.registration.workers);
        num_workers_ = std::max(1, num_workers_);
        score_threshold_ = cfg.registration.score_threshold;
        loop_score_threshold_ = cfg.loop_closure.detect_score_threshold;
        pnh.param<bool>("suppress_backend_output", suppress_backend_output_, true);
        pnh.param<double>("drop_stale_keyframe_age_s",
                          drop_stale_keyframe_age_s_, 2.0);
        readYamlValue(yaml_root,
                      {"registration", "sequential_prior_gate_trans_m"},
                      sequential_prior_gate_trans_m_);
        readYamlValue(yaml_root,
                      {"registration", "sequential_prior_gate_rot_deg"},
                      sequential_prior_gate_rot_deg_);
        readYamlValue(yaml_root,
                      {"registration", "loop_prior_gate_trans_m"},
                      loop_prior_gate_trans_m_);
        readYamlValue(yaml_root,
                      {"registration", "loop_prior_gate_rot_deg"},
                      loop_prior_gate_rot_deg_);
        pnh.param<double>("sequential_prior_gate_trans_m",
                          sequential_prior_gate_trans_m_,
                          sequential_prior_gate_trans_m_);
        pnh.param<double>("sequential_prior_gate_rot_deg",
                          sequential_prior_gate_rot_deg_,
                          sequential_prior_gate_rot_deg_);
        pnh.param<double>("loop_prior_gate_trans_m",
                          loop_prior_gate_trans_m_,
                          loop_prior_gate_trans_m_);
        pnh.param<double>("loop_prior_gate_rot_deg",
                          loop_prior_gate_rot_deg_,
                          loop_prior_gate_rot_deg_);
        debug_prints_ = cfg.debug_prints;
        pnh.param<bool>("debug_prints", debug_prints_, debug_prints_);
        applyDebugPrints(debug_prints_);
        loadD2DOptions(yaml_root, pnh);

        result_pub_ =
            node_->create_publisher<std_msgs::msg::String>(result_topic,
                                                           reliableQos(50));

        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&D2DRegistrationNode::workerLoop, this);
        }

        request_callback_group_ = node_->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions request_options;
        request_options.callback_group = request_callback_group_;
        request_sub_ = node_->create_subscription<std_msgs::msg::String>(
            request_topic, reliableQos(200),
            [this](std_msgs::msg::String::ConstSharedPtr msg) {
                requestCallback(msg);
            },
            request_options);

        GMS_INFO("[d2d_reg] ready | workers=%d | %s -> %s | "
                 "suppress_backend_output=%s | drop_stale_keyframe_age_s=%.2f | "
                 "score_thresholds(seq=%.3f loop=%.3f) | "
                 "prior_gates(seq=%.2fm/%.1fdeg loop=%.2fm/%.1fdeg) | "
                 "debug_prints=%s",
                 num_workers_, request_topic.c_str(), result_topic.c_str(),
                 suppress_backend_output_ ? "true" : "false",
                 drop_stale_keyframe_age_s_,
                 score_threshold_,
                 loop_score_threshold_,
                 sequential_prior_gate_trans_m_,
                 sequential_prior_gate_rot_deg_,
                 loop_prior_gate_trans_m_,
                 loop_prior_gate_rot_deg_,
                 debug_prints_ ? "true" : "false");
        GMS_INFO("[d2d_reg] d2d_options | "
                 "seq(fast=%s margin=%.3f coarse_iter=%lu fine=%s) | "
                 "loop(fast=%s coarse_iter=%lu fine=%s fine_min=%.3f) | "
                 "submap(fast=%s coarse_iter=%lu fine=%s fine_min=%.3f) | "
                 "delta_stop=%.1e fine_delta_stop=%.1e trust=%.2f fine_trust=%.2f "
                 "fine_iter=%lu hessian_damping=%.1e | "
                 "openmp_raw_score=%s min_pairs=%d max_threads=%d timing_log=%s",
                 sequential_d2d_options_.initial_score_fast_path ? "true" : "false",
                 sequential_d2d_options_.initial_score_margin,
                 sequential_d2d_options_.coarse_max_iterations,
                 sequential_d2d_options_.fine_refine_enable ? "true" : "false",
                 loop_d2d_options_.initial_score_fast_path ? "true" : "false",
                 loop_d2d_options_.coarse_max_iterations,
                 loop_d2d_options_.fine_refine_enable ? "true" : "false",
                 loop_d2d_options_.fine_min_coarse_score,
                 submap_d2d_options_.initial_score_fast_path ? "true" : "false",
                 submap_d2d_options_.coarse_max_iterations,
                 submap_d2d_options_.fine_refine_enable ? "true" : "false",
                 submap_d2d_options_.fine_min_coarse_score,
                 sequential_d2d_options_.objective_delta_stop,
                 sequential_d2d_options_.fine_objective_delta_stop,
                 sequential_d2d_options_.initial_trust_radius,
                 sequential_d2d_options_.fine_initial_trust_radius,
                 sequential_d2d_options_.fine_max_iterations,
                 sequential_d2d_options_.hessian_damping,
                 sequential_d2d_options_.openmp_raw_score_enable ? "true" : "false",
                 sequential_d2d_options_.openmp_min_pairs,
                 sequential_d2d_options_.openmp_max_threads,
                 sequential_d2d_options_.openmp_timing_log_enable ? "true" : "false");
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
    static constexpr std::size_t kGmmCacheCapacity = 128;

    class StringRingQueue {
    public:
        explicit StringRingQueue(std::size_t capacity)
            : slots_(capacity), capacity_(capacity) {}

        bool empty() const { return size_ == 0; }
        std::size_t size() const { return size_; }

        std::string& front() { return slots_[head_]; }

        void push_back(std::string value) {
            if (capacity_ == 0) return;
            if (size_ == capacity_) {
                pop_back();
            }
            slots_[tail_] = std::move(value);
            tail_ = (tail_ + 1) % capacity_;
            ++size_;
        }

        void pop_front() {
            if (size_ == 0) return;
            slots_[head_].clear();
            head_ = (head_ + 1) % capacity_;
            --size_;
            if (size_ == 0) {
                head_ = tail_;
            }
        }

        void pop_back() {
            if (size_ == 0) return;
            tail_ = (tail_ + capacity_ - 1) % capacity_;
            slots_[tail_].clear();
            --size_;
            if (size_ == 0) {
                head_ = tail_;
            }
        }

        void clear() {
            for (std::size_t i = 0; i < size_; ++i) {
                slots_[(head_ + i) % capacity_].clear();
            }
            size_ = 0;
            tail_ = head_;
        }

    private:
        std::vector<std::string> slots_;
        std::size_t capacity_ = 0;
        std::size_t head_ = 0;
        std::size_t tail_ = 0;
        std::size_t size_ = 0;
    };

    class GmmLruCache {
    public:
        explicit GmmLruCache(std::size_t capacity) : capacity_(capacity) {
            entries_.reserve(capacity_);
        }

        std::shared_ptr<const gmm_utils::GMM3f> get(const std::string& path,
                                                    bool isoplanar) {
            std::lock_guard<std::mutex> lk(mutex_);

            ++clock_;
            for (auto& entry : entries_) {
                if (entry.raw_path == path) {
                    entry.last_used = clock_;
                    ++hits_;
                    return entry.model;
                }
            }

            const std::string key =
                std::filesystem::absolute(path).lexically_normal().string();
            for (auto& entry : entries_) {
                if (entry.normalized_path == key) {
                    entry.raw_path = path;
                    entry.last_used = clock_;
                    ++hits_;
                    return entry.model;
                }
            }

            auto model_ptr = std::make_shared<gmm_utils::GMM3f>();
            model_ptr->load(key);
            if (isoplanar) {
                model_ptr->makeCovsIsoplanar();
            }

            if (entries_.size() < capacity_) {
                entries_.push_back(Entry{path, key, model_ptr, clock_});
            } else if (!entries_.empty()) {
                auto victim = std::min_element(
                    entries_.begin(), entries_.end(),
                    [](const Entry& a, const Entry& b) {
                        return a.last_used < b.last_used;
                    });
                *victim = Entry{path, key, model_ptr, clock_};
            }
            ++misses_;

            GMS_DEBUG_THROTTLE(
                10.0,
                "[d2d_reg] %s GMM cache size=%zu hits=%zu misses=%zu",
                isoplanar ? "isoplanar" : "normal",
                entries_.size(), hits_, misses_);
            return model_ptr;
        }

    private:
        struct Entry {
            std::string raw_path;
            std::string normalized_path;
            std::shared_ptr<const gmm_utils::GMM3f> model;
            std::uint64_t last_used = 0;
        };

        std::size_t capacity_;
        std::mutex mutex_;
        std::vector<Entry> entries_;
        std::uint64_t clock_ = 0;
        std::size_t hits_ = 0;
        std::size_t misses_ = 0;
    };

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

    static void readDoubleParam(const NodeParams& pnh,
                                const YAML::Node& yaml_root,
                                std::initializer_list<const char*> yaml_path,
                                const std::string& private_key,
                                double& value) {
        readYamlValue(yaml_root, yaml_path, value);
        pnh.param<double>(private_key, value, value);
    }

    static void readBoolParam(const NodeParams& pnh,
                              const YAML::Node& yaml_root,
                              std::initializer_list<const char*> yaml_path,
                              const std::string& private_key,
                              bool& value) {
        readYamlValue(yaml_root, yaml_path, value);
        pnh.param<bool>(private_key, value, value);
    }

    static void readUnsignedParam(const NodeParams& pnh,
                                  const YAML::Node& yaml_root,
                                  std::initializer_list<const char*> yaml_path,
                                  const std::string& private_key,
                                  unsigned long& value) {
        int yaml_value = static_cast<int>(std::min<unsigned long>(
            value, static_cast<unsigned long>(std::numeric_limits<int>::max())));
        readYamlValue(yaml_root, yaml_path, yaml_value);
        int parsed = static_cast<int>(std::min<unsigned long>(
            static_cast<unsigned long>(std::max(1, yaml_value)),
            static_cast<unsigned long>(std::numeric_limits<int>::max())));
        pnh.param<int>(private_key, parsed, parsed);
        value = static_cast<unsigned long>(std::max(1, parsed));
    }

    static void readIntParam(const NodeParams& pnh,
                             const YAML::Node& yaml_root,
                             std::initializer_list<const char*> yaml_path,
                             const std::string& private_key,
                             int& value) {
        readYamlValue(yaml_root, yaml_path, value);
        pnh.param<int>(private_key, value, value);
    }

    static void loadModeOptions(const YAML::Node& yaml_root,
                                const NodeParams& pnh,
                                const std::string& mode,
                                gmmslam::D2DRegistrationOptions& options) {
        const std::string private_prefix = "d2d_" + mode + "_";

        readBoolParam(pnh, yaml_root,
                      {"registration", "d2d", mode.c_str(),
                       "initial_score_fast_path"},
                      private_prefix + "initial_score_fast_path",
                      options.initial_score_fast_path);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", mode.c_str(),
                         "initial_score_margin"},
                        private_prefix + "initial_score_margin",
                        options.initial_score_margin);
        readUnsignedParam(pnh, yaml_root,
                          {"registration", "d2d", mode.c_str(),
                           "coarse_max_iterations"},
                          private_prefix + "coarse_max_iterations",
                          options.coarse_max_iterations);
        readBoolParam(pnh, yaml_root,
                      {"registration", "d2d", mode.c_str(),
                       "fine_refine_enable"},
                      private_prefix + "fine_refine_enable",
                      options.fine_refine_enable);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", mode.c_str(),
                         "fine_min_coarse_score"},
                        private_prefix + "fine_min_coarse_score",
                        options.fine_min_coarse_score);
    }

    void loadD2DOptions(const YAML::Node& yaml_root, const NodeParams& pnh) {
        gmmslam::D2DRegistrationOptions base;
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", "objective_delta_stop"},
                        "d2d_objective_delta_stop",
                        base.objective_delta_stop);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", "initial_trust_radius"},
                        "d2d_initial_trust_radius",
                        base.initial_trust_radius);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", "fine_objective_delta_stop"},
                        "d2d_fine_objective_delta_stop",
                        base.fine_objective_delta_stop);
        readUnsignedParam(pnh, yaml_root,
                          {"registration", "d2d", "fine_max_iterations"},
                          "d2d_fine_max_iterations",
                          base.fine_max_iterations);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", "fine_initial_trust_radius"},
                        "d2d_fine_initial_trust_radius",
                        base.fine_initial_trust_radius);
        readDoubleParam(pnh, yaml_root,
                        {"registration", "d2d", "hessian_damping"},
                        "d2d_hessian_damping",
                        base.hessian_damping);
        readBoolParam(pnh, yaml_root,
                      {"registration", "d2d", "openmp_raw_score_enable"},
                      "d2d_openmp_raw_score_enable",
                      base.openmp_raw_score_enable);
        readIntParam(pnh, yaml_root,
                     {"registration", "d2d", "openmp_min_pairs"},
                     "d2d_openmp_min_pairs",
                     base.openmp_min_pairs);
        readIntParam(pnh, yaml_root,
                     {"registration", "d2d", "openmp_max_threads"},
                     "d2d_openmp_max_threads",
                     base.openmp_max_threads);
        readBoolParam(pnh, yaml_root,
                      {"registration", "d2d", "openmp_timing_log_enable"},
                      "d2d_openmp_timing_log_enable",
                      base.openmp_timing_log_enable);
        base.openmp_min_pairs = std::max(0, base.openmp_min_pairs);
        base.openmp_max_threads = std::max(1, base.openmp_max_threads);

        sequential_d2d_options_ = base;
        sequential_d2d_options_.initial_score_fast_path = true;
        sequential_d2d_options_.initial_score_margin = 0.05;
        sequential_d2d_options_.coarse_max_iterations = 10;
        sequential_d2d_options_.fine_refine_enable = false;

        loop_d2d_options_ = base;
        loop_d2d_options_.initial_score_fast_path = false;
        loop_d2d_options_.coarse_max_iterations = 20;
        loop_d2d_options_.fine_refine_enable = true;
        loop_d2d_options_.fine_min_coarse_score = 0.5;

        submap_d2d_options_ = loop_d2d_options_;

        loadModeOptions(yaml_root, pnh, "sequential", sequential_d2d_options_);
        loadModeOptions(yaml_root, pnh, "loop", loop_d2d_options_);
        loadModeOptions(yaml_root, pnh, "submap", submap_d2d_options_);
    }

    gmmslam::D2DRegistrationOptions optionsForRequest(bool is_loop,
                                                      bool is_submap) const {
        if (is_submap) return submap_d2d_options_;
        if (is_loop) return loop_d2d_options_;
        return sequential_d2d_options_;
    }

    void requestCallback(const std_msgs::msg::String::ConstSharedPtr& msg) {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        bool is_loop = false;
        bool is_submap = false;
        int prev_idx = -1;
        int curr_idx = -1;
        double stamp = 0.0;
        requestKind(msg->data, is_loop, is_submap, prev_idx, curr_idx, stamp);

        if (!is_loop && !is_submap && drop_stale_keyframe_age_s_ > 0.0 &&
            std::isfinite(stamp) && stamp > 0.0) {
            const double age_s = node_->now().seconds() - stamp;
            if (std::isfinite(age_s) && age_s > drop_stale_keyframe_age_s_) {
                GMS_INFO_THROTTLE(
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
                GMS_WARN_THROTTLE(
                    5.0,
                    "[d2d_reg] task queue full; dropped oldest sequential job "
                    "to admit %s request",
                    requestTag(is_loop, is_submap).c_str());
            } else if (is_loop && !submap_queue_.empty()) {
                submap_queue_.pop_back();
                GMS_WARN_THROTTLE(
                    5.0,
                    "[d2d_reg] task queue full; dropped oldest submap job "
                    "to admit loop request");
            } else {
                GMS_WARN_THROTTLE(
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
            const std::size_t dropped = sequential_queue_.size();
            sequential_queue_.clear();
            if (dropped > 0) {
                GMS_WARN_THROTTLE(
                    2.0,
                    "[d2d_reg] dropped %zu queued sequential D2D job(s); "
                    "keeping newest prev=%d curr=%d",
                    dropped, prev_idx, curr_idx);
            }
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
            const double active_score_threshold =
                is_loop ? loop_score_threshold_ : score_threshold_;
            gmmslam::D2DRegistrationOptions d2d_options =
                optionsForRequest(is_loop, is_submap);
            d2d_options.initial_score_threshold = active_score_threshold;
            long seed_score_ms = 0;
            long anisotropic_ms = 0;
            int seed_score_calls = 0;
            int k_source_for_timing = -1;
            int k_target_for_timing = -1;
            auto elapsedSinceRequestStartMs = [&]() -> long {
                return static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - request_t0).count());
            };
            auto logD2DTiming = [&](const char* outcome) {
                if (!d2d_options.openmp_timing_log_enable) {
                    return;
                }
                const std::size_t estimated_pairs =
                    (k_source_for_timing > 0 && k_target_for_timing > 0)
                        ? static_cast<std::size_t>(k_source_for_timing) *
                              static_cast<std::size_t>(k_target_for_timing)
                        : 0U;
                const bool openmp_used =
                    d2d_options.openmp_raw_score_enable &&
                    d2d_options.openmp_max_threads > 1 &&
                    estimated_pairs >=
                        static_cast<std::size_t>(
                            std::max(0, d2d_options.openmp_min_pairs));
                GMS_INFO("[d2d_reg] %s timing | prev=%d curr=%d outcome=%s "
                         "K_src=%d K_tgt=%d est_pairs=%zu openmp_used=%s "
                         "openmp_threads=%d seed_score_calls=%d "
                         "seed_score_ms=%ld anisotropic_ms=%ld total_ms=%ld",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, outcome,
                         k_source_for_timing, k_target_for_timing,
                         estimated_pairs,
                         openmp_used ? "true" : "false",
                         d2d_options.openmp_max_threads,
                         seed_score_calls,
                         seed_score_ms,
                         anisotropic_ms,
                         elapsedSinceRequestStartMs());
            };
            GMS_INFO("[d2d_reg] %s request start | prev=%d curr=%d",
                     requestTag(is_loop, is_submap).c_str(), prev_idx, curr_idx);

            if (!is_loop && !is_submap && drop_stale_keyframe_age_s_ > 0.0 &&
                std::isfinite(stamp) && stamp > 0.0) {
                const double age_s = node_->now().seconds() - stamp;
                if (std::isfinite(age_s) && age_s > drop_stale_keyframe_age_s_) {
                    GMS_INFO_THROTTLE(
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
                const auto elapsed_ms = elapsedSinceRequestStartMs();
                result["failure_reason"] = reason;
                result["elapsed_ms"] = static_cast<long>(elapsed_ms);
                logD2DTiming(reason.c_str());
                GMS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=%s elapsed_ms=%ld",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, reason.c_str(),
                         static_cast<long>(elapsed_ms));
                publishResult(result);
            };

            if (!std::filesystem::exists(source_path) ||
                !std::filesystem::exists(target_path)) {
                GMS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
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
                GMS_INFO_THROTTLE(
                    2.0,
                    "[d2d_reg] %s skipping isoplanar pre-pass; using provided initial_transform",
                    requestTag(is_loop, is_submap).c_str());
            } else if (suppress_backend_output_) {
                // The linked GIRA3D registration library prints covariance inversion
                // diagnostics directly to stdout/stderr. File-descriptor redirection
                // is process-wide, so serialize backend calls while silencing is on.
                std::lock_guard<std::mutex> lk(backend_call_mutex_);
                ScopedBackendOutputSilencer silence(true);
                const auto source_iso =
                    isoplanar_gmm_cache_.get(source_path, true);
                const auto target_iso =
                    isoplanar_gmm_cache_.get(target_path, true);
                iso_result =
                    gmmslam::isoplanarRegistration(
                        T_init, *source_iso, *target_iso, d2d_options);
            } else {
                const auto source_iso =
                    isoplanar_gmm_cache_.get(source_path, true);
                const auto target_iso =
                    isoplanar_gmm_cache_.get(target_path, true);
                iso_result =
                    gmmslam::isoplanarRegistration(
                        T_init, *source_iso, *target_iso, d2d_options);
            }
            if (!has_initial_transform) {
                T_iso = iso_result.transform;
            }
            if (!has_initial_transform &&
                (!std::isfinite(iso_result.score) || T_iso.hasNaN())) {
                GMS_INFO("[d2d_reg] %s isoplanar stage invalid | prev=%d curr=%d "
                         "score=%.4f; falling back to initial transform",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx,
                         static_cast<double>(iso_result.score));
                T_iso = T_init;
            }

            auto run_anisotropic = [&](const Eigen::Matrix4f& seed) {
                const auto t0 = std::chrono::steady_clock::now();
                gmmslam::RegistrationResult result;
                if (suppress_backend_output_) {
                    std::lock_guard<std::mutex> lk(backend_call_mutex_);
                    ScopedBackendOutputSilencer silence(true);
                    const auto source_gmm =
                        normal_gmm_cache_.get(source_path, false);
                    const auto target_gmm =
                        normal_gmm_cache_.get(target_path, false);
                    k_source_for_timing =
                        static_cast<int>(source_gmm->getNClusters());
                    k_target_for_timing =
                        static_cast<int>(target_gmm->getNClusters());
                    result = gmmslam::anisotropicRegistration(
                        seed, *source_gmm, *target_gmm, d2d_options);
                } else {
                    const auto source_gmm =
                        normal_gmm_cache_.get(source_path, false);
                    const auto target_gmm =
                        normal_gmm_cache_.get(target_path, false);
                    k_source_for_timing =
                        static_cast<int>(source_gmm->getNClusters());
                    k_target_for_timing =
                        static_cast<int>(target_gmm->getNClusters());
                    result = gmmslam::anisotropicRegistration(
                        seed, *source_gmm, *target_gmm, d2d_options);
                }
                anisotropic_ms += static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                return result;
            };

            auto score_anisotropic_seed = [&](const Eigen::Matrix4f& seed) {
                const auto t0 = std::chrono::steady_clock::now();
                const auto source_gmm =
                    normal_gmm_cache_.get(source_path, false);
                const auto target_gmm =
                    normal_gmm_cache_.get(target_path, false);
                k_source_for_timing =
                    static_cast<int>(source_gmm->getNClusters());
                k_target_for_timing =
                    static_cast<int>(target_gmm->getNClusters());
                const float score = gmmslam::anisotropicInitialScore(
                    seed, *source_gmm, *target_gmm, d2d_options);
                seed_score_ms += static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count());
                ++seed_score_calls;
                return score;
            };

            // Step 2: anisotropic registration. When SOLiD yaw is available,
            // choose the better seed by cheap initial score and run full D2D once.
            Eigen::Matrix4f selected_seed = T_iso;
            if (is_loop && has_initial_transform &&
                req.contains("solid_yaw_prior_rad") &&
                req.at("solid_yaw_prior_rad").is_number()) {
                const double yaw_rad =
                    req.at("solid_yaw_prior_rad").get<double>();
                if (std::isfinite(yaw_rad)) {
                    Eigen::Matrix4f T_yaw = T_init;
                    T_yaw.block<3, 3>(0, 0) =
                        Eigen::AngleAxisf(static_cast<float>(yaw_rad),
                                           Eigen::Vector3f::UnitZ())
                            .toRotationMatrix();

                    const float latest_seed_score =
                        score_anisotropic_seed(T_iso);
                    const float yaw_seed_score =
                        score_anisotropic_seed(T_yaw);
                    GMS_INFO("[d2d_reg] %s SOLiD yaw seed scores | "
                             "prev=%d curr=%d latest=%.4f yaw=%.4f yaw_deg=%.1f",
                             requestTag(is_loop, is_submap).c_str(),
                             prev_idx, curr_idx,
                             static_cast<double>(latest_seed_score),
                             static_cast<double>(yaw_seed_score),
                             yaw_rad * 180.0 / M_PI);

                    if (std::isfinite(yaw_seed_score) &&
                        (!std::isfinite(latest_seed_score) ||
                         yaw_seed_score > latest_seed_score)) {
                        selected_seed = T_yaw;
                        T_init = T_yaw;
                        result["solid_yaw_retry_used"] = true;

                        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_row(T_init);
                        std::vector<double> flat(16);
                        for (int k = 0; k < 16; ++k) {
                            flat[static_cast<std::size_t>(k)] =
                                static_cast<double>(T_row.data()[k]);
                        }
                        result["initial_transform"] = std::move(flat);
                    }
                }
            }

            aniso_result = run_anisotropic(selected_seed);
            Eigen::Matrix4f T_final = aniso_result.transform;
            float score_final = aniso_result.score;

            if (aniso_result.initial_score_fast_path_used) {
                const int hits = ++d2d_fast_path_hits_;
                GMS_DEBUG_THROTTLE(
                    5.0,
                    "[d2d_reg] initial-score fast-path hits=%d",
                    hits);
            }

            {
                const float init_t = has_initial_transform
                    ? T_init.block<3, 1>(0, 3).norm() : 0.0f;
                const float init_r_deg = has_initial_transform
                    ? static_cast<float>(rotationAngleDeg(T_init.block<3, 3>(0, 0))) : 0.0f;
                const Eigen::Matrix4f T_delta = T_init.inverse() * T_final;
                const float delta_t = T_delta.block<3, 1>(0, 3).norm();
                const float delta_r_deg =
                    static_cast<float>(rotationAngleDeg(T_delta.block<3, 3>(0, 0)));
                result["n_source"] = aniso_result.n_source;
                result["n_target"] = aniso_result.n_target;
                result["init_t_m"] = static_cast<double>(init_t);
                result["init_r_deg"] = static_cast<double>(init_r_deg);
                result["delta_t_m"] = static_cast<double>(delta_t);
                result["delta_r_deg"] = static_cast<double>(delta_r_deg);
                result["coarse_score"] = static_cast<double>(aniso_result.coarse_score);
                GMS_INFO("[d2d_reg] %s dbg | prev=%d curr=%d "
                         "K_src=%d K_tgt=%d "
                         "init_t=%.3fm init_r=%.1fdeg "
                         "delta_t=%.3fm delta_r=%.1fdeg "
                         "coarse=%.4f final=%.4f",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx,
                         aniso_result.n_source, aniso_result.n_target,
                         static_cast<double>(init_t),
                         static_cast<double>(init_r_deg),
                         static_cast<double>(delta_t),
                         static_cast<double>(delta_r_deg),
                         static_cast<double>(aniso_result.coarse_score),
                         static_cast<double>(score_final));
            }

            if (!std::isfinite(score_final) || T_final.hasNaN()) {
                GMS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
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
                const double trans_limit_m = is_loop
                    ? loop_prior_gate_trans_m_
                    : sequential_prior_gate_trans_m_;
                const double rot_limit_deg = is_loop
                    ? loop_prior_gate_rot_deg_
                    : sequential_prior_gate_rot_deg_;
                if (trans_err > trans_limit_m || rot_err_deg > rot_limit_deg) {
                    GMS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
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
            if (score_final < active_score_threshold) {
                GMS_INFO("[d2d_reg] %s registration failed | prev=%d curr=%d "
                         "reason=worker_score_threshold score=%.4f < %.4f",
                         requestTag(is_loop, is_submap).c_str(),
                         prev_idx, curr_idx, static_cast<double>(score_final),
                         active_score_threshold);
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
            const auto elapsed_ms = elapsedSinceRequestStartMs();
            result["elapsed_ms"] = static_cast<long>(elapsed_ms);
            logD2DTiming("success");
            if (is_submap) {
                GMS_INFO("%s[d2d_reg] %s registration success | prev=%d curr=%d "
                         "score=%.4f elapsed_ms=%ld%s",
                         d2dSuccessColor(is_loop, is_submap),
                         tag.c_str(), prev_idx, curr_idx,
                         static_cast<double>(score_final),
                         static_cast<long>(elapsed_ms),
                         kAnsiReset);
            } else {
                GMS_INFO_THROTTLE(
                    5.0,
                    "%s[d2d_reg] %s registration success | prev=%d curr=%d "
                    "score=%.4f elapsed_ms=%ld%s",
                    d2dSuccessColor(is_loop, is_submap),
                    tag.c_str(), prev_idx, curr_idx,
                    static_cast<double>(score_final),
                    static_cast<long>(elapsed_ms),
                    kAnsiReset);
            }
        } catch (const std::exception& e) {
            GMS_WARN_THROTTLE(2.0, "[d2d_reg] registration failed: %s", e.what());
            if (!result.empty()) {
                result["failure_reason"] = "exception";
                result["failure_detail"] = e.what();
            }
            if (result.empty()) return;
        }
        publishResult(result);
    }

    void publishResult(const json& result) {
        std_msgs::msg::String msg;
        msg.data = result.dump();
        result_pub_->publish(msg);
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr request_callback_group_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr request_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_pub_;
    int num_workers_ = 2;
    double score_threshold_ = -1.0e9;
    double loop_score_threshold_ = 1.1;
    bool suppress_backend_output_ = true;
    double drop_stale_keyframe_age_s_ = 2.0;
    double sequential_prior_gate_trans_m_ = 0.25;
    double sequential_prior_gate_rot_deg_ = 15.0;
    double loop_prior_gate_trans_m_ = 3.0;
    double loop_prior_gate_rot_deg_ = 45.0;
    bool debug_prints_ = true;
    gmmslam::D2DRegistrationOptions sequential_d2d_options_;
    gmmslam::D2DRegistrationOptions loop_d2d_options_;
    gmmslam::D2DRegistrationOptions submap_d2d_options_;
    std::atomic<int> d2d_fast_path_hits_{0};

    bool running_ = true;
    std::mutex queue_mutex_;
    std::mutex backend_call_mutex_;
    std::condition_variable queue_cv_;
    StringRingQueue loop_queue_{kMaxQueueSize};
    StringRingQueue submap_queue_{kMaxQueueSize};
    StringRingQueue sequential_queue_{kMaxQueueSize};
    std::vector<std::thread> workers_;
    GmmLruCache normal_gmm_cache_{kGmmCacheCapacity};
    GmmLruCache isoplanar_gmm_cache_{kGmmCacheCapacity};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared(
        "d2d_registration_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

    {
        D2DRegistrationNode app(node);
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
        executor.remove_node(node);
    }

    rclcpp::shutdown();
    return 0;
}
