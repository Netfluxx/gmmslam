#include "gmmslam/sogmm_fitting.hpp"
#include <self_organizing_gmm/SOGMMCPU.h>
#include <self_organizing_gmm/SOGMMLearner.h>

using MatrixX4 = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatrixX2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Container = sogmm::cpu::SOGMM<float, 4>;

namespace gmmslam {

GmmModel fitSogmm(const Eigen::MatrixXf& xyz, const SogmmConfig& config) {
    const int N = static_cast<int>(xyz.rows());
    GmmModel result;
    if (N == 0) return result;

    MatrixX4 X(N, 4);
    X.leftCols(3) = xyz;
    X.col(3) = xyz.rowwise().norm();

    MatrixX2 Y(N, 2);
    Y.col(0) = X.col(3);
    Y.col(1) = xyz.col(2);

    Container model;
    sogmm::cpu::SOGMMLearner<float> learner(static_cast<float>(config.bandwidth));

    if (config.n_components > 0) {
        learner.fit_em(X, static_cast<unsigned int>(config.n_components), model);
    } else {
        learner.fit(Y, X, model);
    }

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
