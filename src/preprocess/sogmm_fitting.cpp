// Fit a CPU SOGMM to a pre-processed point cloud and return a GmmFrame.

#include <gmmslam/preprocess/sogmm_fitting.hpp>

#include <stdexcept>
#include <self_organizing_gmm/SOGMMCPU.h>
#include <self_organizing_gmm/SOGMMCPULearner.h>  // sogmm::cpu::SOGMMLearner

using MatrixX4 = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatrixX2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Container = sogmm::cpu::SOGMM<float, 4>;

// ---------------------------------------------------------------------------
GmmFrame::Ptr fitSogmm(
    const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>& xyz,
    double timestamp,
    int    frame_id,
    const  SogmmFittingConfig& cfg)
{
    const int N = static_cast<int>(xyz.rows());
    if (N == 0)
        return nullptr;

    // Build 4-D feature matrix X (N×4)
    MatrixX4 X(N, 4);
    X.leftCols(3) = xyz;
    if (cfg.feature_4th == "z")
        X.col(3) = xyz.col(2);
    else  // "range" (default)
        X.col(3) = xyz.rowwise().norm();

    // Build 2-D selector matrix Y for mean-shift (Option A only)
    MatrixX2 Y(N, 2);
    Y.col(0) = X.col(3);      // range
    Y.col(1) = xyz.col(2);    // Z height

    Container model;
    sogmm::cpu::SOGMMLearner<float> learner(cfg.bandwidth);

    if (cfg.use_fixed_k)
    {
        // Option B: fixed K, skip mean-shift
        model = Container(cfg.n_components);
        model.support_size_ = N;
        learner.fit_em(X, model);
    }
    else
    {
        // Option A: auto-estimate K via mean-shift on Y
        learner.fit(Y, X, model);
    }

    auto frame       = std::make_shared<GmmFrame>();
    frame->sogmm     = std::move(model);
    frame->timestamp = timestamp;
    frame->frame_id  = frame_id;
    return frame;
}

