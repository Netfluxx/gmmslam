// D2D GMM registration
//code based on : https://github.com/gira3d/gmm_d2d_registration_py/blob/ee4d61630db1b8f0ac5c1da614c00ea449ff920c/src/gmm_d2d_registration_py/gmm_d2d_registration_py.cpp

//   GmmFrame (4D SOGMM, xyz+intensity) ──► sogmm_to_gmm3f()
//                                        │  extract xyz block (3×3 cov, 3D mean)
//                                        ▼
//                                     gmm_utils::GMM3f
//                                        │
//                                        ▼
//                                     MatcherD2D::match()
//                                        │
//                                        ▼
//                                     Eigen::Affine3f  T_source_in_target

#include <gmm/GMM3.h>                                      // gmm_utils::GMM3f
#include <gmm_d2d_registration/GMMD2DRegistration.h>       // MatcherD2D
#include <open3d/core/Tensor.h>                            // open3d::core::Tensor

#include <gmmslam/common/gmm_frame.hpp>

std::pair<Eigen::Matrix<float, 4, 4>, float> anisotropic_registration(const Eigen::Matrix<float, 4, 4>Tin,
								      const std::string& source_file,
								      const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit = Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

std::pair<Eigen::Matrix<float, 4, 4>, float> isoplanar_registration(const Eigen::Matrix<float, 4, 4>Tin,
								    const std::string& source_file,
								    const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit = Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);
  source_gmm.makeCovsIsoplanar();

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);
  target_gmm.makeCovsIsoplanar();

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

//IF GPU:
// ─────────────────────────────────────────────────────────────────────────────
// Convert a 4-D SOGMM (xyz + range) to a 3-D gmm_utils::GMM3f.
//
// The SOGMM tensors live on GPU; we move them to CPU first.
// Layout (row-major Open3D tensors):
//   weights_     : [1, K, 1]
//   means_       : [1, K, 4]   — columns 0-2 are xyz
//   covariances_ : [1, K, 4, 4] — top-left 3×3 block is the spatial cov
// ─────────────────────────────────────────────────────────────────────────────
static gmm_utils::GMM3f sogmm_to_gmm3f(const sogmm::gpu::SOGMM<float, 4>& sogmm)
{
    const int K = sogmm.n_components_;

    // Move to CPU and ensure contiguous layout
    auto w_t = sogmm.weights_.To(open3d::core::Device("CPU:0")).Contiguous();
    auto m_t = sogmm.means_.To(open3d::core::Device("CPU:0")).Contiguous();
    auto c_t = sogmm.covariances_.To(open3d::core::Device("CPU:0")).Contiguous();

    const float* w_ptr = w_t.GetDataPtr<float>(); // [1, K, 1]  → K floats
    const float* m_ptr = m_t.GetDataPtr<float>(); // [1, K, 4]  → K*4 floats
    const float* c_ptr = c_t.GetDataPtr<float>(); // [1, K,4,4] → K*16 floats

    // Eigen storage:
    //   weights : K × 1
    //   means   : 3 × K  (column per component)
    //   covs    : 9 × K  (flattened 3×3 Eigen col-major matrix per column)
    Eigen::Matrix<float, -1, 1> weights(K);
    Eigen::Matrix<float, 3, -1> means(3, K);
    Eigen::Matrix<float, 9, -1> covs(9, K);

    for (int k = 0; k < K; ++k)
    {
        weights(k) = w_ptr[k];

        means(0, k) = m_ptr[k * 4 + 0]; // x
        means(1, k) = m_ptr[k * 4 + 1]; // y
        means(2, k) = m_ptr[k * 4 + 2]; // z

        // Open3D covariance is row-major [4×4]; extract top-left 3×3 block
        const float* cov4 = c_ptr + k * 16;
        Eigen::Matrix3f cov3;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov3(r, c) = cov4[r * 4 + c];

        // Store as Eigen col-major flat vector
        covs.col(k) = Eigen::Map<Eigen::Matrix<float, 9, 1>>(cov3.data());
    }

    gmm_utils::GMM3f gmm;
    gmm.setNClusters(K);
    gmm.setWeights(weights);
    gmm.setMeans(means);
    gmm.setCovs(covs);
    gmm.setSupportSize(static_cast<double>(sogmm.support_size_));
    return gmm;
}

// ─────────────────────────────────────────────────────────────────────────────
// register_gmm_frames()
//
//   Aligns `source` onto `target` using on-manifold D2D GMM registration.
//
//   source    : the frame to be aligned (moving)
//   target    : the reference frame (fixed)
//   T_init    : initial guess  (SE3, source expressed in target frame)
//   T_out     : result transform (source-in-target)
//
//   Returns the *negative* log-likelihood score (lower = better alignment).
// ─────────────────────────────────────────────────────────────────────────────
float register_gmm_frames_gpu(const GmmFrame&      source,
                           const GmmFrame&      target,
                           const Eigen::Affine3f& T_init,
                           Eigen::Affine3f&       T_out)
{
    gmm_utils::GMM3f source_gmm = sogmm_to_gmm3f(source.sogmm);
    gmm_utils::GMM3f target_gmm = sogmm_to_gmm3f(target.sogmm);

    MatcherD2D matcher;
    return matcher.match(source_gmm, target_gmm, T_init, T_out);
}

float register_gmm_frames_cpu(const GmmFrame&      source,
                           const GmmFrame&      target,
                           const Eigen::Affine3f& T_init,
                           Eigen::Affine3f&       T_out)
{
    gmm_utils::GMM3f source_gmm;
    source_gmm.setNClusters(source.sogmm.n_components_);
    source_gmm.setWeights(source.sogmm.weights_.data().cpu().numpy());
    source_gmm.setMeans(source.sogmm.means_.data().cpu().numpy());
    source_gmm.setCovs(source.sogmm.covariances_.data().cpu().numpy());
    source_gmm.setSupportSize(static_cast<double>(source.sogmm.support_size_));

    gmm_utils::GMM3f target_gmm;
    target_gmm.setNClusters(target.sogmm.n_components_);
    target_gmm.setWeights(target.sogmm.weights_.data().cpu().numpy());
    target_gmm.setMeans(target.sogmm.means_.data().cpu().numpy());
    target_gmm.setCovs(target.sogmm.covariances_.data().cpu().numpy());
    target_gmm.setSupportSize(static_cast<double>(target.sogmm.support_size_));

    MatcherD2D matcher;
    return matcher.match(source_gmm, target_gmm, T_init, T_out);
}