/**
 * solid_descriptor_benchmark - ROS node for SOLiD-only timing and metrics.
 *
 * Subscribes to a lidar PointCloud2 (same preprocessing as gmmslam) and optionally
 * a pose topic to log pose/yaw vs SOLiD descriptor similarity vs. a reference scan.
 * reference_mode=first uses the first descriptor; reference_mode=mid_arc buffers the
 * forward arc leg and sets the reference to its temporal midpoint (true cosine).
 *
 * Typical use with Webots apartment + orbit supervisor:
 *   ros2 launch gmmslam solid_descriptor_benchmark.launch.py
 *
 * ~metrics (std_msgs/Float64MultiArray): [t_sec, x_m, y_m, z_m, yaw_rad,
 *                                         delta_yaw_deg, solid_score,
 *                                         range_cos, angle_cos, polar_cos,
 *                                         solid_ms, n_pts, ref_valid]
 * Optional CSV: ~csv_path
 */

#include <algorithm>
#include "gmmslam/rclcpp_logging.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "gmmslam/config.hpp"
#include "gmmslam/ros_helpers.hpp"
#include "gmmslam/solid.hpp"

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

class BenchNode : public rclcpp::Node {
public:
    struct Sample {
        double t_sec = 0.0;
        double x_m = 0.0;
        double y_m = 0.0;
        double z_m = 0.0;
        double yaw_rad = 0.0;
        gmmslam::SolidDescriptor desc;
        Eigen::VectorXd polar_context;
        double solid_ms = 0.0;
        int n_pts = 0;
    };

    BenchNode()
        : rclcpp::Node(
              "solid_descriptor_benchmark",
              rclcpp::NodeOptions()
                  .automatically_declare_parameters_from_overrides(true)) {
        std::string config_path;
        config_path = declareAndGet<std::string>(*this, "config_path", "");
        if (config_path.empty()) {
            config_path = declareAndGet<std::string>(*this, "config_file", "");
        }
        if (!config_path.empty()) {
            cfg_ = gmmslam::loadConfig(config_path);
            GMS_INFO("[solid_bench] Loaded config: %s", config_path.c_str());
        }

        gmmslam::SolidConfig sc = cfg_.solid;
        sc.num_angle = declareAndGet<int>(*this, "solid_num_angle", sc.num_angle);
        sc.num_range = declareAndGet<int>(*this, "solid_num_range", sc.num_range);
        sc.num_height =
            declareAndGet<int>(*this, "solid_num_height", sc.num_height);
        sc.fov_up_deg =
            declareAndGet<double>(*this, "solid_fov_up_deg", sc.fov_up_deg);
        sc.fov_down_deg =
            declareAndGet<double>(*this, "solid_fov_down_deg", sc.fov_down_deg);
        sc.min_distance_m = declareAndGet<double>(
            *this, "solid_min_distance_m", sc.min_distance_m);
        sc.max_distance_m = declareAndGet<double>(
            *this, "solid_max_distance_m", sc.max_distance_m);
        cfg_.solid = sc;
        solid_ = std::make_unique<gmmslam::SOLiDModule>(cfg_.solid);

        lidar_topic_ = declareAndGet<std::string>(
            *this, "lidar_topic", "/m500_1/mpa/lidar_pc");
        pose_topic_ = declareAndGet<std::string>(
            *this, "pose_topic", "/m500_1/mavros/local_position/pose");
        odom_topic_ = declareAndGet<std::string>(*this, "odom_topic", "");

        double log_hz = 10.0;
        log_hz = declareAndGet<double>(*this, "log_hz", 10.0);
        log_interval_s_ = 1.0 / std::max(0.5, log_hz);

        csv_path_ = declareAndGet<std::string>(*this, "csv_path", "");
        reference_mode_ =
            declareAndGet<std::string>(*this, "reference_mode", "first");
        arc_return_drop_m_ =
            declareAndGet<double>(*this, "arc_return_drop_m", 0.15);
        use_mid_arc_ref_ = (reference_mode_ == "mid_arc");

        if (!csv_path_.empty()) {
            csv_.open(csv_path_, std::ios::out | std::ios::trunc);
            if (csv_.good()) {
                have_csv_ = true;
                csv_ << "t_sec,x_m,y_m,z_m,yaw_rad,delta_yaw_deg,solid_score,"
                        "range_cos,angle_cos,polar_cos,solid_ms,n_pts,ref_valid\n";
                csv_.flush();
                GMS_INFO("[solid_bench] Writing CSV: %s", csv_path_.c_str());
            } else {
                GMS_WARN("[solid_bench] Could not open csv_path=%s", csv_path_.c_str());
            }
        }

        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic_, sensorQos(5),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
                onCloud(msg);
            });

        if (!odom_topic_.empty()) {
            odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                odom_topic_, sensorQos(20),
                [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
                    onOdom(msg);
                });
            GMS_INFO("[solid_bench] Subscribing odom: %s", odom_topic_.c_str());
        } else {
            pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                pose_topic_, sensorQos(20),
                [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
                    onPose(msg);
                });
            GMS_INFO("[solid_bench] Subscribing pose: %s", pose_topic_.c_str());
        }

        metrics_pub_ =
            create_publisher<std_msgs::msg::Float64MultiArray>("~/metrics",
                                                               reliableQos(50));
        last_log_wall_ = gmmslam::timeFromSeconds(0.0);

        GMS_INFO(
            "[solid_bench] lidar=%s preprocess: r=[%.2f,%.2f] voxel=%.3f ref=%s",
            lidar_topic_.c_str(), cfg_.preprocess.min_range,
            cfg_.preprocess.max_range, cfg_.preprocess.voxel_leaf_size,
            reference_mode_.c_str());
    }

private:
    void onPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) {
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

    void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
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

    void onCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        Eigen::MatrixXf pts = gmmslam::pc2ToEigen(*msg);
        pts = gmmslam::preprocess(pts, cfg_.preprocess.min_range,
                                  cfg_.preprocess.max_range,
                                  cfg_.preprocess.voxel_leaf_size,
                                  cfg_.preprocess.target_points);
        if (pts.rows() < cfg_.preprocess.min_points) {
            GMS_WARN_THROTTLE(2.0, "[solid_bench] Too few points: %ld",
                              static_cast<long>(pts.rows()));
            return;
        }

        const auto t0 = std::chrono::steady_clock::now();
        gmmslam::SolidDescriptor desc = solid_->makeDescriptor(pts);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (desc.empty()) {
            GMS_WARN_THROTTLE(2.0, "[solid_bench] Empty SOLiD descriptor");
            return;
        }

        const Eigen::VectorXd polar_context = makePolarContext(pts);

        double x_m = 0.0;
        double y_m = 0.0;
        double z_m = 0.0;
        double yaw_rad = 0.0;
        bool have_synced_pose = false;
        {
            std::lock_guard<std::mutex> lk(pose_mu_);
            if (latest_pose_valid_) {
                const double dt =
                    std::abs((msg->header.stamp - latest_pose_stamp_).seconds());
                if (dt <= 0.2) {
                    x_m = latest_x_m_;
                    y_m = latest_y_m_;
                    z_m = latest_z_m_;
                    yaw_rad = latest_yaw_.value();
                    have_synced_pose = true;
                }
            }
        }

        if (!have_synced_pose) {
            return;
        }

        const double t_sec = rclcpp::Time(msg->header.stamp).seconds();

        if (use_mid_arc_ref_ && !ref_ready_) {
            updateMidArcBuffer(t_sec, x_m, y_m, z_m, yaw_rad, desc, polar_context,
                               ms, static_cast<int>(pts.rows()));
            return;
        }

        if (!use_mid_arc_ref_) {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (!ref_desc_.has_value()) {
                ref_desc_ = desc;
                ref_polar_context_ = polar_context;
                ref_yaw_rad_ = yaw_rad;
                ref_ready_ = true;
                GMS_INFO("[solid_bench] Reference descriptor set (first scan, "
                         "pts=%ld, dim=%ld)",
                         static_cast<long>(pts.rows()),
                         static_cast<long>(desc.vec.size()));
            }
        }

        if (!ref_ready_) {
            return;
        }

        publishSample(t_sec, x_m, y_m, z_m, yaw_rad, desc, polar_context, ms,
                      static_cast<int>(pts.rows()));
    }

    void updateMidArcBuffer(double t_sec, double x_m, double y_m, double z_m,
                            double yaw_rad, const gmmslam::SolidDescriptor& desc,
                            const Eigen::VectorXd& polar_context, double ms,
                            int n_pts) {
        if (!arc_origin_valid_) {
            arc_origin_x_m_ = x_m;
            arc_origin_y_m_ = y_m;
            arc_origin_valid_ = true;
            return;
        }

        const double dist =
            std::hypot(x_m - arc_origin_x_m_, y_m - arc_origin_y_m_);
        if (dist < arc_move_tol_m_) {
            return;
        }

        Sample sample;
        sample.t_sec = t_sec;
        sample.x_m = x_m;
        sample.y_m = y_m;
        sample.z_m = z_m;
        sample.yaw_rad = yaw_rad;
        sample.desc = desc;
        sample.polar_context = polar_context;
        sample.solid_ms = ms;
        sample.n_pts = n_pts;
        forward_buffer_.push_back(std::move(sample));

        if (dist > arc_max_dist_m_) {
            arc_max_dist_m_ = dist;
        }

        const bool return_started =
            dist < (arc_max_dist_m_ - arc_return_drop_m_) &&
            arc_max_dist_m_ > arc_move_tol_m_ &&
            forward_buffer_.size() >= 10;

        if (return_started) {
            finalizeMidArcReference();
        }
    }

    void finalizeMidArcReference() {
        if (forward_buffer_.empty() || ref_ready_) {
            return;
        }

        const std::size_t mid_idx = forward_buffer_.size() / 2;
        const Sample& mid = forward_buffer_[mid_idx];

        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            ref_desc_ = mid.desc;
            ref_polar_context_ = mid.polar_context;
            ref_yaw_rad_ = mid.yaw_rad;
            ref_ready_ = true;
        }

        GMS_INFO(
            "[solid_bench] Mid-arc reference set (idx=%zu/%zu, p=(%.2f,%.2f), "
            "dim=%ld)",
            mid_idx, forward_buffer_.size(), mid.x_m, mid.y_m,
            static_cast<long>(mid.desc.vec.size()));

        for (const Sample& sample : forward_buffer_) {
            publishSample(sample.t_sec, sample.x_m, sample.y_m, sample.z_m,
                          sample.yaw_rad, sample.desc, sample.polar_context,
                          sample.solid_ms, sample.n_pts, true);
        }
        forward_buffer_.clear();
        last_log_wall_ = gmmslam::timeFromSeconds(0.0);
    }

    void publishSample(double t_sec, double x_m, double y_m, double z_m,
                       double yaw_rad, const gmmslam::SolidDescriptor& desc,
                       const Eigen::VectorXd& polar_context, double ms,
                       int n_pts, bool force_log = false) {
        double solid_score = 0.0;
        double range_cos = 0.0;
        double angle_cos = 0.0;
        double polar_cos = 0.0;
        double d_yaw_deg = 0.0;

        {
            std::lock_guard<std::mutex> lk(ref_mu_);
            if (!ref_desc_.has_value()) {
                return;
            }
            const auto& ref = ref_desc_.value();
            const int nr = solid_->numRange();
            const int na = solid_->numAngle();
            solid_score = cosineSafe(ref.vec, desc.vec);
            range_cos = solid_->rangeCosine(ref, desc);
            angle_cos =
                cosineSafe(ref.vec.segment(nr, na), desc.vec.segment(nr, na));
            if (ref_polar_context_.has_value()) {
                polar_cos =
                    cosineSafe(ref_polar_context_.value(), polar_context);
            }
            if (ref_yaw_rad_.has_value()) {
                d_yaw_deg =
                    radDiff(yaw_rad, ref_yaw_rad_.value()) * 180.0 / M_PI;
            }
        }

        const rclcpp::Time now = this->now();
        if (!force_log && (now - last_log_wall_).seconds() < log_interval_s_) {
            return;
        }
        last_log_wall_ = now;

        std_msgs::msg::Float64MultiArray out;
        out.data = {t_sec, x_m, y_m, z_m, yaw_rad, d_yaw_deg, solid_score,
                    range_cos, angle_cos, polar_cos, ms,
                    static_cast<double>(n_pts), 1.0};
        metrics_pub_->publish(out);

        GMS_INFO("[solid_bench] t=%.3f p=(%.2f,%.2f,%.2f) d_yaw=%.1fdeg "
                 "solid=%.4f range=%.4f angle=%.4f polar=%.4f SOLiD=%.2fms n=%d",
                 t_sec, x_m, y_m, z_m, d_yaw_deg, solid_score, range_cos,
                 angle_cos, polar_cos, ms, n_pts);

        if (have_csv_) {
            csv_ << t_sec << ',' << x_m << ',' << y_m << ',' << z_m << ','
                 << yaw_rad << ',' << d_yaw_deg << ',' << solid_score << ','
                 << range_cos << ',' << angle_cos << ',' << polar_cos << ','
                 << ms << ',' << n_pts << ",1\n";
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

    gmmslam::Config cfg_;
    std::unique_ptr<gmmslam::SOLiDModule> solid_;

    std::string lidar_topic_;
    std::string pose_topic_;
    std::string odom_topic_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr metrics_pub_;

    std::mutex pose_mu_;
    std::optional<double> latest_yaw_;
    double latest_x_m_ = 0.0;
    double latest_y_m_ = 0.0;
    double latest_z_m_ = 0.0;
    rclcpp::Time latest_pose_stamp_;
    bool latest_pose_valid_ = false;

    std::mutex ref_mu_;
    std::optional<gmmslam::SolidDescriptor> ref_desc_;
    std::optional<Eigen::VectorXd> ref_polar_context_;
    std::optional<double> ref_yaw_rad_;
    bool ref_ready_ = false;

    std::string reference_mode_ = "first";
    bool use_mid_arc_ref_ = false;
    std::vector<Sample> forward_buffer_;
    bool arc_origin_valid_ = false;
    double arc_origin_x_m_ = 0.0;
    double arc_origin_y_m_ = 0.0;
    double arc_max_dist_m_ = 0.0;
    double arc_move_tol_m_ = 0.05;
    double arc_return_drop_m_ = 0.15;

    std::ofstream csv_;
    bool have_csv_ = false;
    std::string csv_path_;

    double log_interval_s_ = 0.1;
    rclcpp::Time last_log_wall_;
};

} // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BenchNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
