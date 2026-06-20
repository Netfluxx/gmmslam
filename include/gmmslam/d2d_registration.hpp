#pragma once
#include <Eigen/Core>
#include <gmm/GMM3.h>
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
    bool initial_score_fast_path_used = false;
};

struct D2DRegistrationOptions {
    unsigned long coarse_max_iterations = 20;
    double objective_delta_stop = 1e-5;
    double initial_trust_radius = 2.0;
    bool fine_refine_enable = true;
    double fine_min_coarse_score = 0.5;
    unsigned long fine_max_iterations = 4;
    double fine_objective_delta_stop = 1e-6;
    double fine_initial_trust_radius = 0.25;
    bool initial_score_fast_path = false;
    double initial_score_margin = 0.05;
    double initial_score_threshold = -std::numeric_limits<double>::infinity();
    double hessian_damping = 1e-3;
    bool openmp_raw_score_enable = true;
    int openmp_min_pairs = 4096;
    int openmp_max_threads = 2;
    bool openmp_timing_log_enable = true;
};

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const D2DRegistrationOptions& options = D2DRegistrationOptions{});

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path,
    const D2DRegistrationOptions& options = D2DRegistrationOptions{});

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const D2DRegistrationOptions& options = D2DRegistrationOptions{});

float anisotropicInitialScore(
    const Eigen::Matrix4f& T_init,
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const D2DRegistrationOptions& options = D2DRegistrationOptions{});

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path,
    const D2DRegistrationOptions& options = D2DRegistrationOptions{});

} // namespace gmmslam
