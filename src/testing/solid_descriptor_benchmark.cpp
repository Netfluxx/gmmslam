/**
 * solid_descriptor_benchmark - ROS node for SOLiD-only timing and metrics.
 *
 * Subscribes to a lidar PointCloud2 (same preprocessing as gmmslam) and optionally
 * a pose topic to log pose/yaw vs SOLiD descriptor similarity vs. the first scan.
 *
 * Typical use with Webots apartment + orbit supervisor:
 *   roslaunch gmmslam solid_descriptor_benchmark.launch
 *
 * ~metrics (std_msgs/Float64MultiArray): [t_sec, x_m, y_m, z_m, yaw_rad,
 *                                         delta_yaw_deg, solid_score,
 *                                         range_cos, angle_cos, polar_cos,
 *                                         solid_ms, n_pts, ref_valid]
 * Optional CSV: ~csv_path
 */

#include <algorithm>
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

double cosineSafe(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    const double denom = a.norm() * b.norm();
    if (denom < 1e-12) {
        return 0.0;
    }
    return std::clamp(a.dot(b) / denom, 0.0, 1.0);
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
                csv_ << "t_sec,x_m,y_m,z_m,yaw_rad,delta_yaw_deg,solid_score,"
                        "range_cos,angle_cos,polar_cos,solid_ms,n_pts,ref_valid\n";
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
        latest_x_m_ = msg->pose.position.x;
        latest_y_m_ = msg->pose.position.y;
        latest_z_m_ = msg->pose.position.z;
        latest_pose_stamp_ = msg->header.stamp;
        latest_pose_valid_ = true;
    }

    void onOdom(const nav_msgs::Odometry::ConstPtr& msg) {
        const auto& q = msg->pose.pose.orientation;
        const double yaw = yawFromQuatXYZW(q.x, q.y, q.z, q.w);
        std::lock_guard<std::mutex> lk(pose_mu_);
        latest_yaw_ = yaw;
        latest_x_m_ = msg->pose.pose.position.x;
        latest_y_m_ = msg->pose.pose.position.y;
        latest_z_m_ = msg->pose.pose.position.z;
        latest_pose_stamp_ = msg->header.stamp;
        latest_pose_valid_ = true;
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

        const Eigen::VectorXd polar_context = makePolarContext(pts);

        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (!ref_desc_.has_value()) {
                ref_desc_ = desc;
                ref_polar_context_ = polar_context;
                ROS_INFO("[solid_bench] Reference descriptor set (pts=%ld, dim=%ld)",
                         static_cast<long>(pts.rows()),
                         static_cast<long>(desc.vec.size()));
                std::lock_guard<std::mutex> pl(pose_mu_);
                if (latest_yaw_.has_value()) {
                    ref_yaw_rad_ = latest_yaw_.value();
                }
            }
        }

        double solid_score = 0.0;
        double range_cos = 0.0;
        double angle_cos = 0.0;
        double polar_cos = 0.0;
        bool ref_valid = false;
        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (ref_desc_.has_value()) {
                const auto& ref = ref_desc_.value();
                const int nr = solid_->numRange();
                const int na = solid_->numAngle();
                solid_score = cosineSafe(ref.vec, desc.vec);
                range_cos = solid_->rangeCosine(ref, desc);
                angle_cos = cosineSafe(ref.vec.segment(nr, na), desc.vec.segment(nr, na));
                if (ref_polar_context_.has_value()) {
                    polar_cos = cosineSafe(ref_polar_context_.value(), polar_context);
                }
                ref_valid = true;
            }
        }

        double x_m = 0.0;
        double y_m = 0.0;
        double z_m = 0.0;
        double yaw_rad = 0.0;
        double d_yaw_deg = 0.0;
        bool have_synced_pose = false;
        {
            std::lock_guard<std::mutex> lk(pose_mu_);
            if (latest_pose_valid_) {
                const double dt =
                    std::abs((msg->header.stamp - latest_pose_stamp_).toSec());
                if (dt <= 0.2) {
                    x_m = latest_x_m_;
                    y_m = latest_y_m_;
                    z_m = latest_z_m_;
                    yaw_rad = latest_yaw_.value();
                    have_synced_pose = true;
                }
            }
        }

        if (have_synced_pose) {
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
        out.data = {t_sec, x_m, y_m, z_m, yaw_rad, d_yaw_deg, solid_score,
                    range_cos, angle_cos, polar_cos, ms,
                    static_cast<double>(desc.point_count),
                    ref_valid ? 1.0 : 0.0};
        metrics_pub_.publish(out);

        ROS_INFO("[solid_bench] t=%.3f p=(%.2f,%.2f,%.2f) d_yaw=%.1fdeg "
                 "solid=%.4f range=%.4f angle=%.4f polar=%.4f SOLiD=%.2fms n=%d",
                 t_sec, x_m, y_m, z_m, d_yaw_deg, solid_score, range_cos,
                 angle_cos, polar_cos, ms, desc.point_count);

        if (have_csv_) {
            csv_ << t_sec << ',' << x_m << ',' << y_m << ',' << z_m << ','
                 << yaw_rad << ',' << d_yaw_deg << ',' << solid_score << ','
                 << range_cos << ',' << angle_cos << ',' << polar_cos << ','
                 << ms << ',' << desc.point_count << ',' << (ref_valid ? 1 : 0)
                 << '\n';
            csv_.flush();
        }
    }

    Eigen::VectorXd makePolarContext(const Eigen::MatrixXf& pts) const {
        const int nr = std::max(1, cfg_.solid.num_range);
        const int na = std::max(1, cfg_.solid.num_angle);
        Eigen::VectorXd context = Eigen::VectorXd::Zero(nr * na);

        const double min_r = cfg_.solid.min_distance_m;
        const double max_r = std::max(cfg_.solid.max_distance_m, min_r + 1.0e-3);
        const double inv_range = static_cast<double>(nr) / (max_r - min_r);
        const double angle_bin_deg = 360.0 / static_cast<double>(na);

        for (int i = 0; i < pts.rows(); ++i) {
            const double x = pts(i, 0);
            const double y = pts(i, 1);
            const double z = pts(i, 2);
            const double dist_xy = std::sqrt(x * x + y * y);
            if (dist_xy < min_r || dist_xy > max_r) {
                continue;
            }

            const double phi_deg = std::atan2(z, dist_xy) * 180.0 / M_PI;
            if (phi_deg < cfg_.solid.fov_down_deg || phi_deg > cfg_.solid.fov_up_deg) {
                continue;
            }

            double theta_deg = std::atan2(y, x) * 180.0 / M_PI;
            if (theta_deg < 0.0) {
                theta_deg += 360.0;
            }

            const int r = std::clamp(static_cast<int>((dist_xy - min_r) * inv_range),
                                     0, nr - 1);
            const int a = std::clamp(static_cast<int>(theta_deg / angle_bin_deg),
                                     0, na - 1);
            context(r * na + a) += 1.0;
        }

        return context;
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
    double latest_x_m_ = 0.0;
    double latest_y_m_ = 0.0;
    double latest_z_m_ = 0.0;
    ros::Time latest_pose_stamp_;
    bool latest_pose_valid_ = false;

    std::mutex ref_mu_;
    std::optional<gmmslam::SolidDescriptor> ref_desc_;
    std::optional<Eigen::VectorXd> ref_polar_context_;
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
