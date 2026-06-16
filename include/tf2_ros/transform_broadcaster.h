#pragma once

#include "ros/ros.h"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace tf2_ros {

class TransformBroadcaster {
public:
    TransformBroadcaster() = default;

    void sendTransform(const geometry_msgs::msg::TransformStamped& transform) {
        ensurePublisher();
        if (!pub_) return;
        tf2_msgs::msg::TFMessage msg;
        msg.transforms.push_back(transform);
        pub_->publish(msg);
    }

private:
    void ensurePublisher() {
        if (pub_) return;
        auto node = ros::detail::globalNode();
        if (!node) return;
        pub_ = node->create_publisher<tf2_msgs::msg::TFMessage>(
            "/tf", rclcpp::QoS(100).reliable());
    }

    rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr pub_;
};

} // namespace tf2_ros
