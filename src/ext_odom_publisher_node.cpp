#include "gmmslam/ros2_compat.hpp"
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>
#include <string>

class ExtOdomPublisherNode {
public:
    ExtOdomPublisherNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : gt_init_start_time_(ros::Time::now()) {
        std::string ref_topic, pose_topic, path_topic;
        pnh.param<std::string>("gt_topic", ref_topic,
                               "/m500_1/mavros/local_position/pose");
        pnh.param<std::string>("odom_frame", odom_frame_, "world");
        pnh.param<std::string>("pose_topic", pose_topic,
                               "/gmmslam_node/ext_odom_pose");
        pnh.param<std::string>("path_topic", path_topic,
                               "/gmmslam_node/ext_odom_path");

        // Private ~params override; fall back to gmmslam namespace (ext_odom: section).
        ros::NodeHandle nh_gmmslam("/gmmslam");
        bool debug_prints = true;
        nh_gmmslam.param("DEBUG_PRINTS", debug_prints, debug_prints);
        pnh.param("debug_prints", debug_prints, debug_prints);
        if (!debug_prints) {
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                           ros::console::levels::Warn);
            ros::console::notifyLoggerLevelsChanged();
        }

        if (!pnh.getParam("ext_odom_init_wait_s", init_wait_s_)) {
            nh_gmmslam.param("ext_odom/init_wait_s", init_wait_s_, 3.0);
        }
        if (!pnh.getParam("noise_sigma_t", noise_sigma_t_)) {
            nh_gmmslam.param("ext_odom/sigma_t", noise_sigma_t_, 0.01);
        }
        if (!pnh.getParam("noise_sigma_r", noise_sigma_r_)) {
            nh_gmmslam.param("ext_odom/sigma_r", noise_sigma_r_, 0.01);
        }
        if (!pnh.getParam("ext_odom_initial_yaw_offset_deg",
                          initial_yaw_offset_deg_)) {
            nh_gmmslam.param("ext_odom/initial_yaw_offset_deg",
                             initial_yaw_offset_deg_, 0.0);
        }
        int seed = -1;
        if (!pnh.getParam("ext_odom_seed", seed)) {
            nh_gmmslam.param("ext_odom/seed", seed, -1);
        }

        const double yaw_offset_rad =
            initial_yaw_offset_deg_ * M_PI / 180.0;
        T_initial_yaw_offset_.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ())
                .toRotationMatrix();

        if (seed >= 0) {
            rng_.seed(static_cast<uint64_t>(seed));
        }

        ext_odom_path_.header.frame_id = odom_frame_;

        pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>(pose_topic, 1);
        path_pub_ = nh.advertise<nav_msgs::Path>(path_topic, 1);
        ref_sub_ = nh.subscribe(ref_topic, 1,
                                &ExtOdomPublisherNode::refCallback, this);

        ROS_INFO("[ext_odom_pub] ready | ref=%s | pose=%s | path=%s | "
                 "sigma_t=%.4f m sigma_r=%.4f rad yaw_offset=%.2f deg "
                 "init_wait=%.2f s seed=%d",
                 ref_topic.c_str(), pose_topic.c_str(), path_topic.c_str(),
                 noise_sigma_t_, noise_sigma_r_,
                 initial_yaw_offset_deg_, init_wait_s_, seed);
    }

private:
    static Eigen::Matrix4d poseMsgToMatrix(const geometry_msgs::Pose& pose) {
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

    geometry_msgs::PoseStamped matrixToPoseStamped(
        const Eigen::Matrix4d& T, const ros::Time& stamp) const {
        geometry_msgs::PoseStamped ps;
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

    bool ensureOriginInitialized(const ros::Time& stamp) {
        if (origin_initialized_) return true;
        if (!has_ref_pose_) return false;

        const double now_sec = stamp.toSec();
        const double t0_sec = gt_init_start_time_.toSec();
        if ((now_sec - t0_sec) < init_wait_s_) return false;

        try {
            const Eigen::Matrix4d T0 = poseMsgToMatrix(latest_ref_raw_->pose);
            origin_inv_ = T0.inverse();
            origin_initialized_ = true;
            ext_odom_path_ = nav_msgs::Path();
            ext_odom_path_.header.frame_id = odom_frame_;
            ROS_INFO("[ext_odom_pub] origin initialized");
            return true;
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[ext_odom_pub] failed to init origin: %s",
                              e.what());
            return false;
        }
    }

    void refCallback(const geometry_msgs::PoseStamped::ConstSharedPtr& msg) {
        latest_ref_raw_ = msg;
        has_ref_pose_ = true;
        if (!ensureOriginInitialized(msg->header.stamp)) return;

        Eigen::Matrix4d T_ref;
        try {
            T_ref = poseMsgToMatrix(msg->pose);
        } catch (...) {
            ROS_WARN_THROTTLE(2.0, "[ext_odom_pub] invalid reference pose");
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
        pose_pub_.publish(ps);
        path_pub_.publish(ext_odom_path_);
    }

    ros::Subscriber ref_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher path_pub_;
    std::string odom_frame_;
    double init_wait_s_ = 3.0;
    double noise_sigma_t_ = 0.01;
    double noise_sigma_r_ = 0.01;
    double initial_yaw_offset_deg_ = 0.0;
    Eigen::Matrix4d T_initial_yaw_offset_ = Eigen::Matrix4d::Identity();
    std::mt19937_64 rng_{std::random_device{}()};
    std::normal_distribution<double> normal_dist_{0.0, 1.0};

    geometry_msgs::PoseStamped::ConstSharedPtr latest_ref_raw_;
    bool has_ref_pose_ = false;
    bool origin_initialized_ = false;
    Eigen::Matrix4d origin_inv_ = Eigen::Matrix4d::Identity();
    ros::Time gt_init_start_time_;
    nav_msgs::Path ext_odom_path_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ext_odom_publisher_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ExtOdomPublisherNode node(nh, pnh);
    ros::spin();
    return 0;
}
