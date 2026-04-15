#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>
#include <string>

class NoisyGTPublisherNode {
public:
    NoisyGTPublisherNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : gt_init_start_time_(ros::Time::now()) {
        std::string gt_topic, pose_topic, path_topic;
        pnh.param<std::string>("gt_topic", gt_topic,
                               "/m500_1/mavros/local_position/pose");
        pnh.param<std::string>("odom_frame", odom_frame_, "world");
        pnh.param<std::string>("pose_topic", pose_topic,
                               "/gmmslam_node/noisy_gt_pose");
        pnh.param<std::string>("path_topic", path_topic,
                               "/gmmslam_node/noisy_gt_path");
        pnh.param<double>("gt_init_wait_s", gt_init_wait_s_, 3.0);
        pnh.param<double>("gt_noise_sigma_t", gt_noise_sigma_t_, 0.0);
        pnh.param<double>("gt_noise_sigma_r", gt_noise_sigma_r_, 0.0);
        int seed = -1;
        pnh.param<int>("gt_noise_seed", seed, -1);

        if (seed >= 0) {
            rng_.seed(static_cast<uint64_t>(seed));
        }

        noisy_path_.header.frame_id = odom_frame_;

        pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>(pose_topic, 1);
        path_pub_ = nh.advertise<nav_msgs::Path>(path_topic, 1);
        gt_sub_ = nh.subscribe(gt_topic, 1,
                               &NoisyGTPublisherNode::gtCallback, this);

        ROS_INFO("[noisy_gt_pub] ready | gt=%s | pose=%s | path=%s",
                 gt_topic.c_str(), pose_topic.c_str(), path_topic.c_str());
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
        if (gt_origin_initialized_) return true;
        if (!has_gt_pose_) return false;

        const double now_sec = stamp.toSec();
        const double t0_sec = gt_init_start_time_.toSec();
        if ((now_sec - t0_sec) < gt_init_wait_s_) return false;

        try {
            const Eigen::Matrix4d T0 = poseMsgToMatrix(latest_gt_raw_->pose);
            gt_origin_inv_ = T0.inverse();
            gt_origin_initialized_ = true;
            noisy_path_ = nav_msgs::Path();
            noisy_path_.header.frame_id = odom_frame_;
            ROS_INFO("[noisy_gt_pub] GT origin initialized");
            return true;
        } catch (const std::exception& e) {
            ROS_WARN_THROTTLE(2.0, "[noisy_gt_pub] failed to init GT origin: %s",
                              e.what());
            return false;
        }
    }

    void gtCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        latest_gt_raw_ = msg;
        has_gt_pose_ = true;
        if (!ensureOriginInitialized(msg->header.stamp)) return;

        Eigen::Matrix4d T_gt;
        try {
            T_gt = poseMsgToMatrix(msg->pose);
        } catch (...) {
            ROS_WARN_THROTTLE(2.0, "[noisy_gt_pub] invalid GT pose");
            return;
        }
        const Eigen::Matrix4d T_odom = gt_origin_inv_ * T_gt;

        const Eigen::Vector3d rot_noise(normal_dist_(rng_) * gt_noise_sigma_r_,
                                        normal_dist_(rng_) * gt_noise_sigma_r_,
                                        normal_dist_(rng_) * gt_noise_sigma_r_);
        const Eigen::Vector3d trans_noise(
            normal_dist_(rng_) * gt_noise_sigma_t_,
            normal_dist_(rng_) * gt_noise_sigma_t_,
            normal_dist_(rng_) * gt_noise_sigma_t_);

        const double angle = rot_noise.norm();
        Eigen::Matrix3d R_noise = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) {
            R_noise =
                Eigen::AngleAxisd(angle, rot_noise.normalized()).toRotationMatrix();
        }

        Eigen::Matrix4d T_noisy = Eigen::Matrix4d::Identity();
        T_noisy.block<3, 3>(0, 0) = R_noise * T_odom.block<3, 3>(0, 0);
        T_noisy.block<3, 1>(0, 3) = T_odom.block<3, 1>(0, 3) + trans_noise;

        const auto ps = matrixToPoseStamped(T_noisy, msg->header.stamp);
        noisy_path_.header.stamp = msg->header.stamp;
        noisy_path_.poses.push_back(ps);
        pose_pub_.publish(ps);
        path_pub_.publish(noisy_path_);
    }

    ros::Subscriber gt_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher path_pub_;
    std::string odom_frame_;
    double gt_init_wait_s_ = 3.0;
    double gt_noise_sigma_t_ = 0.0;
    double gt_noise_sigma_r_ = 0.0;
    std::mt19937_64 rng_{std::random_device{}()};
    std::normal_distribution<double> normal_dist_{0.0, 1.0};

    geometry_msgs::PoseStamped::ConstPtr latest_gt_raw_;
    bool has_gt_pose_ = false;
    bool gt_origin_initialized_ = false;
    Eigen::Matrix4d gt_origin_inv_ = Eigen::Matrix4d::Identity();
    ros::Time gt_init_start_time_;
    nav_msgs::Path noisy_path_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "noisy_gt_publisher_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    NoisyGTPublisherNode node(nh, pnh);
    ros::spin();
    return 0;
}
