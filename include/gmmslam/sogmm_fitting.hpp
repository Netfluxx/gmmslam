#pragma once
#include "gmmslam/types.hpp"
#include "gmmslam/config.hpp"
#include <Eigen/Core>

namespace gmmslam {

GmmModel fitSogmm(
    const Eigen::MatrixXf& xyz,
    const SogmmConfig& config);

} // namespace gmmslam
