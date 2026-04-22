#include "gmmslam/sogmm_fitting.hpp"
#include <self_organizing_gmm/SOGMMCPU.h>
#include <self_organizing_gmm/SOGMMLearner.h>

#include <ros/ros.h>

#include <chrono>
#include <numeric>
#include <random>
#include <vector>

using MatrixX4 = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatrixX2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Container = sogmm::cpu::SOGMM<float, 4>;

namespace gmmslam {

namespace {

// Random subset without replacement (same idea as Python np.random.choice).
Eigen::MatrixXf downsampleMaxPoints(const Eigen::MatrixXf& xyz, int max_points) {
    const int n = static_cast<int>(xyz.rows());
    if (max_points <= 0 || n <= max_points) {
        return xyz;
    }
    std::vector<int> pop(static_cast<std::size_t>(n));
    std::iota(pop.begin(), pop.end(), 0);
    std::vector<int> pick(static_cast<std::size_t>(max_points));
    thread_local std::mt19937 rng{std::random_device{}()};
    std::sample(pop.begin(), pop.end(), pick.begin(),
                static_cast<std::ptrdiff_t>(max_points), rng);
    Eigen::MatrixXf out(max_points, xyz.cols());
    for (int i = 0; i < max_points; ++i) {
        out.row(i) = xyz.row(pick[static_cast<std::size_t>(i)]);
    }
    return out;
}

} // namespace

GmmModel fitSogmm(const Eigen::MatrixXf& xyz, const SogmmConfig& config) {
    GmmModel result;
    const int n_in = static_cast<int>(xyz.rows());
    if (n_in == 0) {
        return result;
    }

    const Eigen::MatrixXf xyz_fit = downsampleMaxPoints(xyz, config.max_points);
    const int N = static_cast<int>(xyz_fit.rows());

    const auto t0 = std::chrono::steady_clock::now();

    MatrixX4 X(N, 4);
    X.leftCols(3) = xyz_fit;
    X.col(3) = xyz_fit.rowwise().norm();

    MatrixX2 Y(N, 2);
    Y.col(0) = X.col(3);
    Y.col(1) = xyz_fit.col(2);

    Container model;
    sogmm::cpu::SOGMMLearner<float> learner(static_cast<float>(config.bandwidth));

    if (config.n_components > 0) {
        learner.fit_em(X, static_cast<unsigned int>(config.n_components), model);
    } else {
        learner.fit(Y, X, model);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double dt_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    ROS_INFO("[sogmm] fit N_in=%d N_fit=%d K=%d %.1f ms bw=%.3f "
             "(max_points=%d)",
             n_in, N, model.n_components_, dt_ms, config.bandwidth,
             config.max_points);

    const int K = model.n_components_;
    result.components.reserve(static_cast<size_t>(K));

    for (int k = 0; k < K; ++k) {
        GmmComponent comp;
        comp.mean = model.means_.row(k).head(3).template cast<double>();
        // covariances_ is (K × D²); row k holds a flattened 4×4 covariance
        Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> cov_4x4(
            &model.covariances_(k, 0));
        comp.covariance = cov_4x4.block<3, 3>(0, 0).template cast<double>();
        comp.weight = static_cast<double>(model.weights_(k));
        result.components.push_back(std::move(comp));
    }

    return result;
}

} // namespace gmmslam
