/**
 * solid_descriptor_benchmark — ROS node for SOLiD-only timing and metrics.
 *
 * Subscribes to a lidar PointCloud2 (same preprocessing as gmmslam) and optionally
 * a pose topic to log yaw vs SOLiD range-head cosine similarity vs. the first scan.
 *
 * Typical use with Webots apartment + orbit supervisor:
 *   roslaunch gmmslam solid_descriptor_benchmark.launch
 *
 * ~metrics (std_msgs/Float64MultiArray): [t_sec, yaw_rad, delta_yaw_deg, cos_range,
 *                                         solid_ms, n_pts, ref_valid]
 * Optional CSV: ~csv_path
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>

#include "gmmslam/config.hpp"
#include "gmmslam/ros_helpers.hpp"
#include "gmmslam/solid.hpp"

namespace {

double yawFromQuatXYZW(double x, double y, double z, double w) {
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

double radDiff(double a, double b) {
    double d = a - b;
    while (d > M_PI) {
        d -= 2.0 * M_PI;
    }
    while (d < -M_PI) {
        d += 2.0 * M_PI;
    }
    return d;
}

class BenchNode {
public:
    BenchNode(ros::NodeHandle nh, ros::NodeHandle pnh)
        : nh_(std::move(nh)), pnh_(std::move(pnh)) {
        std::string config_path;
        pnh_.param<std::string>("config_path", config_path, "");
        if (!config_path.empty()) {
            cfg_ = gmmslam::loadConfig(config_path);
            ROS_INFO("[solid_bench] Loaded config: %s", config_path.c_str());
        }

        gmmslam::SolidConfig sc = cfg_.solid;
        pnh_.param("solid_num_angle", sc.num_angle, sc.num_angle);
        pnh_.param("solid_num_range", sc.num_range, sc.num_range);
        pnh_.param("solid_num_height", sc.num_height, sc.num_height);
        pnh_.param("solid_fov_up_deg", sc.fov_up_deg, sc.fov_up_deg);
        pnh_.param("solid_fov_down_deg", sc.fov_down_deg, sc.fov_down_deg);
        pnh_.param("solid_min_distance_m", sc.min_distance_m, sc.min_distance_m);
        pnh_.param("solid_max_distance_m", sc.max_distance_m, sc.max_distance_m);
        cfg_.solid = sc;
        solid_ = std::make_unique<gmmslam::SOLiDModule>(cfg_.solid);

        pnh_.param<std::string>("lidar_topic", lidar_topic_,
                                std::string("/m500_1/mpa/lidar_pc"));
        pnh_.param<std::string>("pose_topic", pose_topic_,
                                std::string("/m500_1/mavros/local_position/pose"));
        pnh_.param<std::string>("odom_topic", odom_topic_, std::string(""));

        double log_hz = 10.0;
        pnh_.param("log_hz", log_hz, 10.0);
        log_interval_s_ = 1.0 / std::max(0.5, log_hz);

        pnh_.param<std::string>("csv_path", csv_path_, "");
        if (!csv_path_.empty()) {
            csv_.open(csv_path_, std::ios::out | std::ios::trunc);
            if (csv_.good()) {
                have_csv_ = true;
                csv_ << "t_sec,yaw_rad,delta_yaw_deg,cos_range,solid_ms,n_pts,ref_valid\n";
                csv_.flush();
                ROS_INFO("[solid_bench] Writing CSV: %s", csv_path_.c_str());
            } else {
                ROS_WARN("[solid_bench] Could not open csv_path=%s", csv_path_.c_str());
            }
        }

        cloud_sub_ = nh_.subscribe(lidar_topic_, 5, &BenchNode::onCloud, this);

        if (!odom_topic_.empty()) {
            odom_sub_ = nh_.subscribe(odom_topic_, 20, &BenchNode::onOdom, this);
            ROS_INFO("[solid_bench] Subscribing odom: %s", odom_topic_.c_str());
        } else {
            pose_sub_ = nh_.subscribe(pose_topic_, 20, &BenchNode::onPose, this);
            ROS_INFO("[solid_bench] Subscribing pose: %s", pose_topic_.c_str());
        }

        metrics_pub_ = pnh_.advertise<std_msgs::Float64MultiArray>("metrics", 50);
        last_log_wall_ = ros::Time(0);

        ROS_INFO("[solid_bench] lidar=%s preprocess: r=[%.2f,%.2f] voxel=%.3f",
                 lidar_topic_.c_str(), cfg_.preprocess.min_range,
                 cfg_.preprocess.max_range, cfg_.preprocess.voxel_leaf_size);
    }

private:
    void onPose(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        const auto& q = msg->pose.orientation;
        const double yaw = yawFromQuatXYZW(q.x, q.y, q.z, q.w);
        std::lock_guard<std::mutex> lk(pose_mu_);
        latest_yaw_ = yaw;
        latest_pose_stamp_ = msg->header.stamp;
    }

    void onOdom(const nav_msgs::Odometry::ConstPtr& msg) {
        const auto& q = msg->pose.pose.orientation;
        const double yaw = yawFromQuatXYZW(q.x, q.y, q.z, q.w);
        std::lock_guard<std::mutex> lk(pose_mu_);
        latest_yaw_ = yaw;
        latest_pose_stamp_ = msg->header.stamp;
    }

    void onCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        Eigen::MatrixXf pts = gmmslam::pc2ToEigen(*msg);
        pts = gmmslam::preprocess(pts, cfg_.preprocess.min_range,
                                  cfg_.preprocess.max_range,
                                  cfg_.preprocess.voxel_leaf_size,
                                  cfg_.preprocess.target_points);
        if (pts.rows() < cfg_.preprocess.min_points) {
            ROS_WARN_THROTTLE(2.0, "[solid_bench] Too few points: %ld",
                              static_cast<long>(pts.rows()));
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        gmmslam::SolidDescriptor desc = solid_->makeDescriptor(pts);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (desc.empty()) {
            ROS_WARN_THROTTLE(2.0, "[solid_bench] Empty SOLiD descriptor");
            return;
        }

        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (!ref_desc_.has_value()) {
                ref_desc_ = desc;
                ROS_INFO("[solid_bench] Reference descriptor set (pts=%ld, dim=%ld)",
                         static_cast<long>(pts.rows()),
                         static_cast<long>(desc.vec.size()));
                std::lock_guard<std::mutex> pl(pose_mu_);
                if (latest_yaw_.has_value()) {
                    ref_yaw_rad_ = latest_yaw_.value();
                }
            }
        }

        double cos_sim = 0.0;
        bool ref_valid = false;
        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (ref_desc_.has_value()) {
                cos_sim = solid_->rangeCosine(ref_desc_.value(), desc);
                ref_valid = true;
            }
        }

        double yaw_rad = 0.0;
        double d_yaw_deg = 0.0;
        bool have_synced_yaw = false;
        {
            std::lock_guard<std::mutex> lk(pose_mu_);
            if (latest_yaw_.has_value()) {
                const double dt =
                    std::abs((msg->header.stamp - latest_pose_stamp_).toSec());
                if (dt <= 0.2) {
                    yaw_rad = latest_yaw_.value();
                    have_synced_yaw = true;
                }
            }
        }

        if (have_synced_yaw) {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (ref_yaw_rad_.has_value()) {
                d_yaw_deg =
                    radDiff(yaw_rad, ref_yaw_rad_.value()) * 180.0 / M_PI;
            }
        }

        const ros::Time now = ros::Time::now();
        if ((now - last_log_wall_).toSec() < log_interval_s_) {
            return;
        }
        last_log_wall_ = now;

        const double t_sec = msg->header.stamp.toSec();

        std_msgs::Float64MultiArray out;
        out.data = {t_sec, yaw_rad, d_yaw_deg, cos_sim, ms,
                    static_cast<double>(desc.point_count),
                    ref_valid ? 1.0 : 0.0};
        metrics_pub_.publish(out);

        ROS_INFO("[solid_bench] t=%.3f d_yaw=%.1f° cos_range=%.4f SOLiD=%.2fms n=%d",
                 t_sec, d_yaw_deg, cos_sim, ms, desc.point_count);

        if (have_csv_) {
            csv_ << t_sec << ',' << yaw_rad << ',' << d_yaw_deg << ',' << cos_sim
                 << ',' << ms << ',' << desc.point_count << ','
                 << (ref_valid ? 1 : 0) << '\n';
            csv_.flush();
        }
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    gmmslam::Config cfg_;
    std::unique_ptr<gmmslam::SOLiDModule> solid_;

    std::string lidar_topic_;
    std::string pose_topic_;
    std::string odom_topic_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher metrics_pub_;

    std::mutex pose_mu_;
    std::optional<double> latest_yaw_;
    ros::Time latest_pose_stamp_;

    std::mutex ref_mu_;
    std::optional<gmmslam::SolidDescriptor> ref_desc_;
    std::optional<double> ref_yaw_rad_;

    std::ofstream csv_;
    bool have_csv_ = false;
    std::string csv_path_;

    double log_interval_s_ = 0.1;
    ros::Time last_log_wall_;
};

} // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "solid_descriptor_benchmark");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    BenchNode node(nh, pnh);
    ros::spin();
    return 0;
}
