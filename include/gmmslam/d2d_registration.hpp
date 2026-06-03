#pragma once
#include <Eigen/Core>
#include <limits>
#include <string>

namespace gmmslam {

struct RegistrationResult {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float score = -std::numeric_limits<float>::infinity();
    float coarse_score = -std::numeric_limits<float>::infinity();
    int n_source = 0;
    int n_target = 0;
    bool success = false;
};

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path);

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path);

} // namespace gmmslam
