#pragma once

#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <cmath>

namespace gmmslam {

inline rclcpp::Logger defaultLogger() {
    return rclcpp::get_logger("gmmslam");
}

inline rclcpp::Clock& defaultClock() {
    static rclcpp::Clock clock(RCL_SYSTEM_TIME);
    return clock;
}

inline rclcpp::Time now() {
    return defaultClock().now();
}

inline rclcpp::Time timeFromSeconds(double seconds) {
    if (!std::isfinite(seconds)) {
        seconds = 0.0;
    }
    return rclcpp::Time(
        static_cast<std::int64_t>(seconds * 1000000000.0), RCL_SYSTEM_TIME);
}

}  // namespace gmmslam

#define GMS_INFO(...) RCLCPP_INFO(::gmmslam::defaultLogger(), __VA_ARGS__)
#define GMS_WARN(...) RCLCPP_WARN(::gmmslam::defaultLogger(), __VA_ARGS__)
#define GMS_ERROR(...) RCLCPP_ERROR(::gmmslam::defaultLogger(), __VA_ARGS__)
#define GMS_DEBUG(...) RCLCPP_DEBUG(::gmmslam::defaultLogger(), __VA_ARGS__)
#define GMS_INFO_THROTTLE(period, ...) \
    RCLCPP_INFO_THROTTLE(::gmmslam::defaultLogger(), ::gmmslam::defaultClock(), \
                         static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define GMS_WARN_THROTTLE(period, ...) \
    RCLCPP_WARN_THROTTLE(::gmmslam::defaultLogger(), ::gmmslam::defaultClock(), \
                         static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define GMS_ERROR_THROTTLE(period, ...) \
    RCLCPP_ERROR_THROTTLE(::gmmslam::defaultLogger(), ::gmmslam::defaultClock(), \
                          static_cast<int>((period) * 1000.0), __VA_ARGS__)
#define GMS_DEBUG_THROTTLE(period, ...) \
    RCLCPP_DEBUG_THROTTLE(::gmmslam::defaultLogger(), ::gmmslam::defaultClock(), \
                          static_cast<int>((period) * 1000.0), __VA_ARGS__)
