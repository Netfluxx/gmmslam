// input: raw pointcloud
// output: deskew the pointcloud ? + Fit a SOGMM and wrap in GmmFrame

#include <sogmm_open3d/SOGMMLearner.h>
#include <sogmm_open3d/SOGMMInference.h>
#include <sogmm_open3d/SOGMMGPU.h>
#include <sogmm_open3d/KInit.h>
#include <sogmm_open3d/GMM.h>
#include <sogmm_open3d/EM.h>



// sogmm_open3d : C++ codebase accelerate with CUDA and Open3D for Intel/AMD CPUs and NVIDIA GPUs with CUDA version <= 12.1
//see : https://github.com/gira3d/sogmm_open3d/tree/master/include/sogmm_open3d

// how the SOGMM Open3D library works:

// SOGMMLearner.h exposes two C++ fitting interfaces: 
// option A: Requires a 2D selector matrix Y (e.g. depth + intensity) to estimate the number of Gaussian components via Mean Shift, then runs EM on the full 4D point cloud X (XYZ + feature).
// option B: Fixed number of components (KInit → EM)
// Uses SOGMMLearner::fit_em(). You supply K directly; no need for a 2D descriptor.


    //   void fit(const MatrixX2 &Y, const MatrixX4 &X, Container &sogmm)
    //   {
    //     if (Y.rows() != X.rows())
    //     {
    //       throw std::runtime_error("[SOGMMLearner] Number of samples are not the same in image and point cloud.");
    //     }

    //     // Run GBMS to estimate number of components.
    //     ms_->fit(Y);

    //     // Initialize this sogmm.
    //     sogmm = Container(ms_->get_num_modes());
    //     sogmm.support_size_ = X.rows();

    //     // Compute initial responsibility matrix.
    //     Matrix resp = Matrix::Zero(sogmm.support_size_, sogmm.n_components_);
    //     kinit_->getRespMat(X, resp);

    //     // Take stuff to the GPU
    //     Xt_ = EigenMatrixToTensor(X, sogmm.device_);
    //     Respt_ = EigenMatrixToTensor(resp, sogmm.device_);

    //     // Fit GMM.
    //     em_->fit(Xt_, Respt_, sogmm);
    //   }


//option A:

using MatrixX4 = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using MatrixX2 = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Container = sogmm::gpu::SOGMM<float, 4>;

const float bandwidth = 0.1f; // bandwidth
sogmm::gpu::SOGMMLearner<float> learner(bandwidth);

MatrixX4 X ;  // point cloud input (xyz) : ros1 noetic pointcloud message is a collection of 3d points
                //can also be depth + intensity for ex
X.resize(X_3d.rows(), 4);
X.leftCols(3) = X_3d;
X.col(3) = X_3d.rowwise().norm();  // range as 4th dim


MatrixX2 Y;
Y.resize(X_3d.rows(), 2);
Y.col(0) = X_3d.rowwise().norm();          // range
Y.col(1) = X_3d.col(2);                    // Z height, or just range again


Container sogmm;
learner.fit(Y, X, sogmm);
// Result:
// sogmm.n_components_  — number of fitted Gaussians
// sogmm.weights_       — Open3D Tensor [1, K, 1]
// sogmm.means_         — Open3D Tensor [1, K, 4]
// sogmm.covariances_   — Open3D Tensor [1, K, 4, 4]