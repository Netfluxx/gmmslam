#pragma once

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace ros {

class Duration {
public:
    Duration() = default;
    explicit Duration(double seconds) : seconds_(seconds) {}

    double toSec() const { return seconds_; }

    void sleep() const {
        if (seconds_ > 0.0) {
            rclcpp::sleep_for(std::chrono::nanoseconds(
                static_cast<std::int64_t>(seconds_ * 1.0e9)));
        }
    }

    operator builtin_interfaces::msg::Duration() const {
        builtin_interfaces::msg::Duration msg;
        const auto nsec_total =
            static_cast<std::int64_t>(seconds_ * 1000000000.0);
        msg.sec = static_cast<std::int32_t>(nsec_total / 1000000000LL);
        msg.nanosec = static_cast<std::uint32_t>(
            std::max<std::int64_t>(0, nsec_total % 1000000000LL));
        return msg;
    }

private:
    double seconds_ = 0.0;
};

class Time {
public:
    Time() = default;
    explicit Time(double seconds)
        : time_(static_cast<std::int64_t>(seconds * 1000000000.0), RCL_ROS_TIME) {}
    Time(const builtin_interfaces::msg::Time& stamp)
        : time_(stamp, RCL_ROS_TIME) {}
    Time(const rclcpp::Time& stamp) : time_(stamp) {}

    static Time now();

    double toSec() const { return time_.seconds(); }
    bool isZero() const { return time_.nanoseconds() == 0; }

    operator builtin_interfaces::msg::Time() const {
        return time_.to_msg();
    }

    const rclcpp::Time& rclcppTime() const { return time_; }

private:
    rclcpp::Time time_{0, 0, RCL_ROS_TIME};
};

inline Duration operator-(const Time& a, const Time& b) {
    return Duration(a.toSec() - b.toSec());
}

namespace detail {

inline rclcpp::Node::SharedPtr& globalNode() {
    static rclcpp::Node::SharedPtr node;
    return node;
}

inline rclcpp::Logger logger() {
    if (globalNode()) {
        return globalNode()->get_logger();
    }
    return rclcpp::get_logger("gmmslam");
}

inline rclcpp::Clock& clock() {
    static rclcpp::Clock fallback_clock(RCL_ROS_TIME);
    if (globalNode()) {
        return *globalNode()->get_clock();
    }
    return fallback_clock;
}

inline std::string normalizeParamName(std::string name) {
    while (!name.empty() && name.front() == '/') {
        name.erase(name.begin());
    }
    for (char& ch : name) {
        if (ch == '/') ch = '.';
    }
    return name;
}

template <typename Msg>
rclcpp::QoS makeQos(std::size_t depth, bool transient_local = false) {
    rclcpp::QoS qos(depth);
    if constexpr (std::is_same_v<Msg, sensor_msgs::msg::PointCloud2> ||
                  std::is_same_v<Msg, sensor_msgs::msg::Imu>) {
        qos = rclcpp::SensorDataQoS().keep_last(depth);
    } else {
        qos.reliable();
    }
    if (transient_local) {
        qos.transient_local();
    }
    return qos;
}

inline std::string resolveTopic(const std::string& ns,
                                bool is_private,
                                const std::string& topic) {
    if (topic.empty() || topic.front() == '/') {
        return topic;
    }
    if (is_private) {
        return "~/" + topic;
    }
    if (!ns.empty() && ns.front() == '/') {
        return ns + "/" + topic;
    }
    return topic;
}

} // namespace detail

inline Time Time::now() {
    return Time(detail::clock().now());
}

class Publisher {
public:
    Publisher() = default;

    template <typename Msg>
    explicit Publisher(typename rclcpp::Publisher<Msg>::SharedPtr pub)
        : holder_(std::make_shared<Model<Msg>>(std::move(pub))) {}

    template <typename Msg>
    void publish(const Msg& msg) const {
        auto typed = std::dynamic_pointer_cast<Model<Msg>>(holder_);
        if (typed && typed->pub) {
            typed->pub->publish(msg);
        }
    }

    std::size_t getNumSubscribers() const {
        return holder_ ? holder_->subscriptionCount() : 0;
    }

    std::string getTopic() const {
        return holder_ ? holder_->topic() : std::string();
    }

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::size_t subscriptionCount() const = 0;
        virtual std::string topic() const = 0;
    };

    template <typename Msg>
    struct Model final : Concept {
        explicit Model(typename rclcpp::Publisher<Msg>::SharedPtr p)
            : pub(std::move(p)) {}
        std::size_t subscriptionCount() const override {
            return pub ? pub->get_subscription_count() : 0;
        }
        std::string topic() const override {
            return pub ? pub->get_topic_name() : std::string();
        }
        typename rclcpp::Publisher<Msg>::SharedPtr pub;
    };

    std::shared_ptr<Concept> holder_;
};

class Subscriber {
public:
    Subscriber() = default;
    explicit Subscriber(rclcpp::SubscriptionBase::SharedPtr sub)
        : sub_(std::move(sub)) {}

private:
    rclcpp::SubscriptionBase::SharedPtr sub_;
};

class NodeHandle {
public:
    NodeHandle()
        : node_(detail::globalNode()) {}

    explicit NodeHandle(const std::string& ns)
        : node_(detail::globalNode()),
          namespace_(ns == "~" ? std::string() : ns),
          private_(ns == "~") {}

    explicit NodeHandle(rclcpp::Node::SharedPtr node,
                        std::string ns = {},
                        bool is_private = false)
        : node_(std::move(node)),
          namespace_(std::move(ns)),
          private_(is_private) {}

    template <typename T>
    void param(const std::string& key, T& value, const T& default_value) const {
        if (!node_) {
            value = default_value;
            return;
        }
        const std::string name = private_
            ? detail::normalizeParamName(key)
            : detail::normalizeParamName(namespace_.empty() ? key
                                                            : namespace_ + "/" + key);
        if (!node_->has_parameter(name)) {
            node_->declare_parameter<T>(name, default_value);
        }
        if (!node_->get_parameter(name, value)) {
            value = default_value;
        }
    }

    template <typename T>
    bool getParam(const std::string& key, T& value) const {
        if (!node_) return false;
        const std::string name = private_
            ? detail::normalizeParamName(key)
            : detail::normalizeParamName(
                  namespace_.empty() ? key : namespace_ + "/" + key);
        if (!node_->has_parameter(name)) return false;
        return node_->get_parameter(name, value);
    }

    template <typename Msg>
    Publisher advertise(const std::string& topic,
                        std::size_t queue_size,
                        bool latch = false) const {
        const auto qos = detail::makeQos<Msg>(queue_size, latch);
        return Publisher(node_->create_publisher<Msg>(
            detail::resolveTopic(namespace_, private_, topic), qos));
    }

    template <typename Msg, typename Obj>
    Subscriber subscribe(const std::string& topic,
                         std::size_t queue_size,
                         void (Obj::*callback)(typename Msg::ConstSharedPtr),
                         Obj* object) const {
        const auto qos = detail::makeQos<Msg>(queue_size);
        auto sub = node_->create_subscription<Msg>(
            detail::resolveTopic(namespace_, private_, topic),
            qos,
            [object, callback](typename Msg::ConstSharedPtr msg) {
                (object->*callback)(std::move(msg));
            });
        return Subscriber(sub);
    }

    template <typename Msg, typename Obj>
    Subscriber subscribe(const std::string& topic,
                         std::size_t queue_size,
                         void (Obj::*callback)(const typename Msg::ConstSharedPtr&),
                         Obj* object) const {
        const auto qos = detail::makeQos<Msg>(queue_size);
        auto sub = node_->create_subscription<Msg>(
            detail::resolveTopic(namespace_, private_, topic),
            qos,
            [object, callback](typename Msg::ConstSharedPtr msg) {
                (object->*callback)(msg);
            });
        return Subscriber(sub);
    }

    std::string getNamespace() const {
        if (!node_) return {};
        if (private_) {
            return std::string("/") + node_->get_name();
        }
        return namespace_.empty() ? node_->get_namespace() : namespace_;
    }

    rclcpp::Node::SharedPtr node() const { return node_; }

private:
    rclcpp::Node::SharedPtr node_;
    std::string namespace_;
    bool private_ = false;
};

inline void init(int argc, char** argv, const std::string& node_name) {
    rclcpp::init(argc, argv);
    detail::globalNode() = rclcpp::Node::make_shared(
        node_name,
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
}

inline void spin() {
    if (detail::globalNode()) {
        rclcpp::spin(detail::globalNode());
    }
}

inline bool isShuttingDown() {
    return !rclcpp::ok();
}

namespace console {
namespace levels {
enum Level { Debug, Info, Warn, Error, Fatal };
} // namespace levels
inline void set_logger_level(const char*, levels::Level) {}
inline void notifyLoggerLevelsChanged() {}
} // namespace console

} // namespace ros

#ifndef ROSCONSOLE_DEFAULT_NAME
#define ROSCONSOLE_DEFAULT_NAME "gmmslam"
#endif

#define ROS_INFO(...) RCLCPP_INFO(::ros::detail::logger(), __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(::ros::detail::logger(), __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(::ros::detail::logger(), __VA_ARGS__)
#define ROS_DEBUG(...) RCLCPP_DEBUG(::ros::detail::logger(), __VA_ARGS__)
#define ROS_INFO_THROTTLE(period, ...) \
    RCLCPP_INFO_THROTTLE(::ros::detail::logger(), ::ros::detail::clock(), \
                         static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define ROS_WARN_THROTTLE(period, ...) \
    RCLCPP_WARN_THROTTLE(::ros::detail::logger(), ::ros::detail::clock(), \
                         static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define ROS_ERROR_THROTTLE(period, ...) \
    RCLCPP_ERROR_THROTTLE(::ros::detail::logger(), ::ros::detail::clock(), \
                          static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define ROS_DEBUG_THROTTLE(period, ...) \
    RCLCPP_DEBUG_THROTTLE(::ros::detail::logger(), ::ros::detail::clock(), \
                          static_cast<int>((period) * 1000.0), __VA_ARGS__)
