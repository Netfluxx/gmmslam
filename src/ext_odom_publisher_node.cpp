#include "gmmslam/config.hpp"
#include "gmmslam/rclcpp_logging.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging.h>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace {

rclcpp::QoS reliableQos(std::size_t depth) {
    return rclcpp::QoS(rclcpp::KeepLast(depth)).reliable();
}

rclcpp::QoS sensorQos(std::size_t depth) {
    return rclcpp::SensorDataQoS().keep_last(depth);
}

template <typename T>
T declareAndGet(rclcpp::Node& node, const std::string& name, const T& value) {
    if (!node.has_parameter(name)) {
        node.declare_parameter<T>(name, value);
    }
    T out = value;
    node.get_parameter(name, out);
    return out;
}

void applyDebugPrints(bool enable) {
    if (!enable) {
        rcutils_logging_set_logger_level(
            "gmmslam", RCUTILS_LOG_SEVERITY_WARN);
    }
}

class ExtOdomPublisherNode : public rclcpp::Node {
public:
    ExtOdomPublisherNode()
        : rclcpp::Node(
              "ext_odom_publisher_node",
              rclcpp::NodeOptions()
                  .automatically_declare_parameters_from_overrides(true)) {
        std::string config_path =
            declareAndGet<std::string>(*this, "config_file", "");
        gmmslam::Config cfg;
        if (!config_path.empty()) {
            cfg = gmmslam::loadConfig(config_path);
        }

        std::string ref_topic = declareAndGet<std::string>(
            *this, "gt_topic", cfg.ros.gt_topic.empty()
                                    ? "/m500_1/mavros/local_position/pose"
                                    : cfg.ros.gt_topic);
        odom_frame_ = declareAndGet<std::string>(
            *this, "odom_frame", cfg.ros.odom_frame.empty()
                                      ? "world"
                                      : cfg.ros.odom_frame);
        const std::string pose_topic = declareAndGet<std::string>(
            *this, "pose_topic", "/gmmslam_node/ext_odom_pose");
        const std::string path_topic = declareAndGet<std::string>(
            *this, "path_topic", "/gmmslam_node/ext_odom_path");

        init_wait_s_ = declareAndGet<double>(
            *this, "ext_odom_init_wait_s", cfg.ext_odom.init_wait_s);
        noise_sigma_t_ = declareAndGet<double>(
            *this, "noise_sigma_t", cfg.ext_odom.sigma_t);
        noise_sigma_r_ = declareAndGet<double>(
            *this, "noise_sigma_r", cfg.ext_odom.sigma_r);
        initial_yaw_offset_deg_ = declareAndGet<double>(
            *this, "ext_odom_initial_yaw_offset_deg",
            cfg.ext_odom.initial_yaw_offset_deg);
        const int seed =
            declareAndGet<int>(*this, "ext_odom_seed", cfg.ext_odom.seed);

        const bool debug_prints = declareAndGet<bool>(
            *this, "debug_prints", cfg.debug_prints);
        applyDebugPrints(debug_prints);

        const double yaw_offset_rad =
            initial_yaw_offset_deg_ * M_PI / 180.0;
        T_initial_yaw_offset_.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        if (seed >= 0) {
            rng_.seed(static_cast<uint64_t>(seed));
        }

        gt_init_start_time_ = now();
        ext_odom_path_.header.frame_id = odom_frame_;

        pose_pub_ =
            create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic,
                                                              reliableQos(1));
        path_pub_ =
            create_publisher<nav_msgs::msg::Path>(path_topic, reliableQos(1));
        ref_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            ref_topic, sensorQos(1),
            [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
                refCallback(msg);
            });

        GMS_INFO("[ext_odom_pub] ready | ref=%s | pose=%s | path=%s | "
                 "sigma_t=%.4f m sigma_r=%.4f rad yaw_offset=%.2f deg "
                 "init_wait=%.2f s seed=%d",
                 ref_topic.c_str(), pose_topic.c_str(), path_topic.c_str(),
                 noise_sigma_t_, noise_sigma_r_, initial_yaw_offset_deg_,
                 init_wait_s_, seed);
    }

private:
    static Eigen::Matrix4d poseMsgToMatrix(
        const geometry_msgs::msg::Pose& pose) {
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
                             pose.orientation.y, pose.orientation.z);
        const double qn = q.norm();
        if (qn < 1e-12) {
            throw std::runtime_error("invalid quaternion (near-zero norm)");
        }
        q.normalize();
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = q.toRotationMatrix();
        T(0, 3) = pose.position.x;
        T(1, 3) = pose.position.y;
        T(2, 3) = pose.position.z;
        return T;
    }

    geometry_msgs::msg::PoseStamped matrixToPoseStamped(
        const Eigen::Matrix4d& T, const rclcpp::Time& stamp) const {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = stamp;
        ps.header.frame_id = odom_frame_;
        ps.pose.position.x = T(0, 3);
        ps.pose.position.y = T(1, 3);
        ps.pose.position.z = T(2, 3);
        Eigen::Quaterniond q(T.block<3, 3>(0, 0));
        q.normalize();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        return ps;
    }

    bool ensureOriginInitialized(const rclcpp::Time& stamp) {
        if (origin_initialized_) return true;
        if (!has_ref_pose_) return false;

        const double now_sec = stamp.seconds();
        const double t0_sec = gt_init_start_time_.seconds();
        if ((now_sec - t0_sec) < init_wait_s_) return false;

        try {
            const Eigen::Matrix4d T0 = poseMsgToMatrix(latest_ref_raw_->pose);
            origin_inv_ = T0.inverse();
            origin_initialized_ = true;
            ext_odom_path_ = nav_msgs::msg::Path();
            ext_odom_path_.header.frame_id = odom_frame_;
            GMS_INFO("[ext_odom_pub] origin initialized");
            return true;
        } catch (const std::exception& e) {
            GMS_WARN_THROTTLE(2.0, "[ext_odom_pub] failed to init origin: %s",
                              e.what());
            return false;
        }
    }

    void refCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
        latest_ref_raw_ = msg;
        has_ref_pose_ = true;
        if (!ensureOriginInitialized(msg->header.stamp)) return;

        Eigen::Matrix4d T_ref;
        try {
            T_ref = poseMsgToMatrix(msg->pose);
        } catch (...) {
            GMS_WARN_THROTTLE(2.0, "[ext_odom_pub] invalid reference pose");
            return;
        }
        const Eigen::Matrix4d T_odom = origin_inv_ * T_ref;

        const Eigen::Vector3d rot_noise(normal_dist_(rng_) * noise_sigma_r_,
                                        normal_dist_(rng_) * noise_sigma_r_,
                                        normal_dist_(rng_) * noise_sigma_r_);
        const Eigen::Vector3d trans_noise(
            normal_dist_(rng_) * noise_sigma_t_,
            normal_dist_(rng_) * noise_sigma_t_,
            normal_dist_(rng_) * noise_sigma_t_);

        const double angle = rot_noise.norm();
        Eigen::Matrix3d R_noise = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) {
            R_noise =
                Eigen::AngleAxisd(angle, rot_noise.normalized()).toRotationMatrix();
        }

        Eigen::Matrix4d T_out = T_odom * T_initial_yaw_offset_;
        T_out.block<3, 3>(0, 0) = R_noise * T_out.block<3, 3>(0, 0);
        T_out.block<3, 1>(0, 3) += trans_noise;

        const auto ps = matrixToPoseStamped(T_out, msg->header.stamp);
        ext_odom_path_.header.stamp = msg->header.stamp;
        ext_odom_path_.poses.push_back(ps);
        pose_pub_->publish(ps);
        path_pub_->publish(ext_odom_path_);
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ref_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    std::string odom_frame_;
    double init_wait_s_ = 3.0;
    double noise_sigma_t_ = 0.01;
    double noise_sigma_r_ = 0.01;
    double initial_yaw_offset_deg_ = 0.0;
    Eigen::Matrix4d T_initial_yaw_offset_ = Eigen::Matrix4d::Identity();
    std::mt19937_64 rng_{std::random_device{}()};
    std::normal_distribution<double> normal_dist_{0.0, 1.0};

    geometry_msgs::msg::PoseStamped::ConstSharedPtr latest_ref_raw_;
    bool has_ref_pose_ = false;
    bool origin_initialized_ = false;
    Eigen::Matrix4d origin_inv_ = Eigen::Matrix4d::Identity();
    rclcpp::Time gt_init_start_time_;
    nav_msgs::msg::Path ext_odom_path_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ExtOdomPublisherNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);
    rclcpp::shutdown();
    return 0;
}
