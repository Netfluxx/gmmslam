#include "gmmslam/d2d_registration.hpp"

#include <dlib/optimization.h>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <gmm/GMM3.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace gmmslam {

namespace {

constexpr int kNumDims = 3;
constexpr float kInitialInflation = 0.3f * 0.3f;
constexpr float kFineInitialInflation = 0.03f * 0.03f;
constexpr float kGateRadiusMultiplierSq = 16.0f;
constexpr float kTranslationPriorWeight = 0.0f;
constexpr float kRotationPriorWeight = 2.5f;
constexpr float kPi32 = 1.0f / (2.0f * static_cast<float>(M_PI) *
                                std::sqrt(2.0f * static_cast<float>(M_PI)));

using ColumnVector = dlib::matrix<float, 0, 1>;

struct CachedComponent {
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
    float trace = 0.0f;
    float weight = 0.0f;
    bool valid = false;
};

using ComponentCache = std::vector<CachedComponent>;

Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>
toAffine(const Eigen::Matrix4f& T) {
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> out;
    out = Eigen::Translation3f(T.block<3, 1>(0, 3)) *
          Eigen::Quaternionf(T.block<3, 3>(0, 0));
    return out;
}

Eigen::Matrix3f rotationFromAxisAngleVector(const Eigen::Vector3f& u) {
    const float unorm = u.norm();
    if (unorm < 1e-10f) {
        return Eigen::Matrix3f::Identity();
    }
    return Eigen::AngleAxisf(unorm, u / unorm).toRotationMatrix();
}

Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>
transformFromState(const ColumnVector& x) {
    const Eigen::Vector3f u(x(3), x(4), x(5));
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> T =
        Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>::Identity();
    T.linear() = rotationFromAxisAngleVector(u);
    T.translation() = Eigen::Vector3f(x(0), x(1), x(2));
    return T;
}

Eigen::Matrix3f covarianceFromColumn(const Eigen::MatrixXf& covs,
                                     std::uint32_t k) {
    return Eigen::Map<const Eigen::Matrix3f>(covs.col(k).data(),
                                             kNumDims, kNumDims);
}

ComponentCache buildComponentCache(const gmm_utils::GMM3f& gmm) {
    ComponentCache cache;
    const std::uint32_t n = gmm.getNClusters();
    cache.reserve(static_cast<std::size_t>(n));
    const auto& weights = gmm.getWeights();
    const auto& covs = gmm.getCovs();
    const auto& means = gmm.getMeans();
    for (std::uint32_t k = 0; k < n; ++k) {
        CachedComponent c;
        c.mean = means.col(k);
        c.covariance = covarianceFromColumn(covs, k);
        c.trace = c.covariance.trace();
        c.weight = weights(k);
        c.valid = c.mean.allFinite() && c.covariance.allFinite() &&
                  std::isfinite(c.trace) && std::isfinite(c.weight) &&
                  c.weight > 0.0f;
        cache.push_back(c);
    }
    return cache;
}

bool inverseAndDeterminant(const Eigen::Matrix3f& S,
                           Eigen::Matrix3f& S_inv,
                           float& det) {
    Eigen::Matrix3d Sd = S.cast<double>();
    Eigen::Matrix3d invd;
    double detd = 0.0;
    bool invertible = false;
    Sd.computeInverseAndDetWithCheck(invd, detd, invertible, 1e-30);

    if (!invertible || !std::isfinite(detd) || detd <= 0.0 ||
        !invd.allFinite()) {
        // Try one small isotropic jitter before giving up on the pair.
        Sd.diagonal().array() += 1e-9;
        Sd.computeInverseAndDetWithCheck(invd, detd, invertible, 1e-30);
    }

    if (!invertible || !std::isfinite(detd) || detd <= 0.0 ||
        !invd.allFinite()) {
        return false;
    }

    det = static_cast<float>(detd);
    S_inv = invd.cast<float>();
    return true;
}

class GmmRegistrationModel {
public:
    using column_vector = ColumnVector;
    using general_matrix = dlib::matrix<float>;

    GmmRegistrationModel(const gmm_utils::GMM3f& source_gmm,
                         const gmm_utils::GMM3f& target_gmm,
                         const Eigen::Vector3f& translation_prior,
                         const Eigen::Vector3f& rotation_prior_u,
                         const D2DRegistrationOptions& options,
                         float initial_inflation = kInitialInflation)
        : source_cache_(buildComponentCache(source_gmm)),
          target_cache_(buildComponentCache(target_gmm)),
          translation_prior_(translation_prior),
          rotation_prior_u_(rotation_prior_u),
          options_(options),
          initial_inflation_(initial_inflation),
          inflation_factor_(initial_inflation) {
        const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        const Eigen::Vector3f t = Eigen::Vector3f::Zero();
        Et_ = rawScore(R, t, target_cache_, target_cache_, 1.0f,
                       inflation_factor_);
        if (!std::isfinite(Et_) || Et_ <= std::numeric_limits<float>::min()) {
            Et_ = 1.0f;
        }
    }

    float operator()(const column_vector& x) const {
        const Eigen::Vector3f t(x(0), x(1), x(2));
        const Eigen::Vector3f u(x(3), x(4), x(5));
        const float d2d_score =
            rawScore(rotationFromAxisAngleVector(u), t,
                     target_cache_, source_cache_, Et_, inflation_factor_);

        const Eigen::Vector3f trans_delta = t - translation_prior_;
        const Eigen::Vector3f rot_delta = u - rotation_prior_u_;
        return d2d_score -
               kTranslationPriorWeight * trans_delta.squaredNorm() -
               kRotationPriorWeight * rot_delta.squaredNorm();
    }

    void get_derivative_and_hessian(const column_vector& x,
                                    column_vector& der,
                                    general_matrix& hess) const {
        constexpr int n = 6;
        der.set_size(n);
        hess.set_size(n, n);
        for (int r = 0; r < n; ++r) {
            der(r) = 0.0f;
            for (int c = 0; c < n; ++c) {
                hess(r, c) = 0.0f;
            }
        }
        for (int i = 0; i < n; ++i) {
            hess(i, i) = -1.0f;
        }

        const Eigen::Vector3f t(x(0), x(1), x(2));
        const Eigen::Vector3f u(x(3), x(4), x(5));
        const Eigen::Matrix3f R = rotationFromAxisAngleVector(u);

        std::vector<Eigen::Matrix3f> dRdu;
        std::vector<std::vector<Eigen::Matrix3f>> d2Rdu2;
        partialWrtU(x(3), x(4), x(5), dRdu);

        float score = 0.0f;
        Eigen::Matrix<float, 6, 1> analytic_grad =
            Eigen::Matrix<float, 6, 1>::Zero();
        Eigen::Matrix<float, 6, 6> unused_full_hess =
            Eigen::Matrix<float, 6, 6>::Zero();
        corrGradHess(R, t,
                     target_cache_,
                     source_cache_,
                     Et_,
                     dRdu, d2Rdu2,
                     false,
                     score, analytic_grad, unused_full_hess);

        if (analytic_grad.allFinite()) {
            analytic_grad.template head<3>() -=
                2.0f * kTranslationPriorWeight * (t - translation_prior_);
            analytic_grad.template tail<3>() -=
                2.0f * kRotationPriorWeight * (u - rotation_prior_u_);
            der = dlib::mat(analytic_grad);
        }

        updateInflation(der, hess);
    }

private:
    static float rawScore(const Eigen::Matrix3f& R,
                          const Eigen::Vector3f& t,
                          const ComponentCache& target_gmm,
                          const ComponentCache& source_gmm,
                          float normalizer,
                          float inflation_factor) {
        if (target_gmm.empty() || source_gmm.empty() || normalizer == 0.0f) {
            return 0.0f;
        }

        const Eigen::Matrix3f Minf =
            Eigen::Matrix3f::Identity() * inflation_factor;
        float score = 0.0f;

        for (const auto& cm : target_gmm) {
            if (!cm.valid) {
                continue;
            }
            const Eigen::Matrix3f& L = cm.covariance;
            const Eigen::Vector3f& mu = cm.mean;
            for (const auto& ck : source_gmm) {
                if (!ck.valid) {
                    continue;
                }
                const Eigen::Matrix3f& Om = ck.covariance;
                const Eigen::Vector3f& nu = ck.mean;
                const float w = cm.weight * ck.weight;
                if (!std::isfinite(w) || w <= 0.0f) {
                    continue;
                }

                const Eigen::Vector3f Rnu = R * nu;
                const Eigen::Vector3f y = mu - Rnu - t;

                const float radius2 = std::max(
                    cm.trace + ck.trace,
                    std::numeric_limits<float>::epsilon());
                if (y.squaredNorm() > kGateRadiusMultiplierSq * radius2) {
                    continue;
                }

                const Eigen::Matrix3f ROm = R * (Om + Minf);
                const Eigen::Matrix3f S = L + Minf + ROm * R.transpose();
                Eigen::Matrix3f S_inv;
                float det = 0.0f;
                if (!inverseAndDeterminant(S, S_inv, det)) {
                    continue;
                }

                const Eigen::Vector3f Sy = S_inv * y;
                const float exponent = -0.5f * y.dot(Sy);
                if (!std::isfinite(exponent)) {
                    continue;
                }

                const float contribution =
                    w * kPi32 * (1.0f / std::sqrt(det)) *
                    std::exp(exponent) / normalizer;
                if (std::isfinite(contribution)) {
                    score += contribution;
                }
            }
        }

        return score;
    }

    void corrGradHess(
        const Eigen::Matrix3f& R,
        const Eigen::Vector3f& t,
        const ComponentCache& target_gmm,
        const ComponentCache& source_gmm,
        float normalizer,
        const std::vector<Eigen::Matrix3f>& dRdu,
        const std::vector<std::vector<Eigen::Matrix3f>>& d2Rdu2,
        bool compute_hessian,
        float& score,
        Eigen::Matrix<float, 6, 1>& J,
        Eigen::Matrix<float, 6, 6>& H) const {
        score = 0.0f;
        J.setZero();
        H.setZero();
        if (target_gmm.empty() || source_gmm.empty() || normalizer == 0.0f) {
            return;
        }

        Eigen::Vector3f dFdt = Eigen::Vector3f::Zero();
        Eigen::Matrix3f dFdR = Eigen::Matrix3f::Zero();
        Eigen::Matrix<float, 3, 6> Htr =
            Eigen::Matrix<float, 3, 6>::Zero();
        Eigen::Matrix<float, 9, 3> Delta =
            Eigen::Matrix<float, 9, 3>::Zero();
        Eigen::Matrix<float, 9, 3> Gamma =
            Eigen::Matrix<float, 9, 3>::Zero();
        const Eigen::Matrix3f Minf =
            Eigen::Matrix3f::Identity() * inflation_factor_;

        for (const auto& cm : target_gmm) {
            if (!cm.valid) {
                continue;
            }
            const Eigen::Matrix3f& L = cm.covariance;
            const Eigen::Vector3f& mu = cm.mean;
            for (const auto& ck : source_gmm) {
                if (!ck.valid) {
                    continue;
                }
                const Eigen::Matrix3f& Om = ck.covariance;
                const Eigen::Vector3f& nu = ck.mean;
                const float w = cm.weight * ck.weight;
                if (!std::isfinite(w) || w <= 0.0f) {
                    continue;
                }

                const Eigen::Vector3f Rnu = R * nu;
                const Eigen::Vector3f y = mu - Rnu - t;
                const float radius2 = std::max(
                    cm.trace + ck.trace,
                    std::numeric_limits<float>::epsilon());
                if (y.squaredNorm() > kGateRadiusMultiplierSq * radius2) {
                    continue;
                }

                const Eigen::Matrix3f ROm = R * (Om + Minf);
                const Eigen::Matrix3f S = L + Minf + ROm * R.transpose();
                Eigen::Matrix3f S_inv;
                float det = 0.0f;
                if (!inverseAndDeterminant(S, S_inv, det)) {
                    continue;
                }

                const Eigen::Vector3f Sy = S_inv * y;
                const float exponent = -0.5f * y.dot(Sy);
                if (!std::isfinite(exponent)) {
                    continue;
                }

                const float fmk =
                    w * kPi32 * (1.0f / std::sqrt(det)) *
                    std::exp(exponent) / normalizer;
                if (!std::isfinite(fmk)) {
                    continue;
                }
                score += fmk;

                dFdt.noalias() += fmk * Sy;

                const Eigen::Matrix3f Synut = Sy * nu.transpose();
                const Eigen::Matrix3f SySyT = Sy * Sy.transpose();
                const Eigen::Matrix3f SySyTROm = SySyT * ROm;
                const Eigen::Matrix3f SROm = S_inv * ROm;

                dFdR.noalias() += fmk * (Synut + SySyTROm - SROm);
                if (compute_hessian) {
                    Htr.template block<3, 3>(0, 0).noalias() +=
                        fmk * (SySyT - S_inv);
                }

                if (!compute_hessian) {
                    continue;
                }

                for (int ix = 0; ix < kNumDims; ++ix) {
                    const Eigen::Matrix3f A = dRdu[ix];
                    const Eigen::Matrix3f ROmAt = ROm * A.transpose();
                    const Eigen::Matrix3f Za = ROmAt + ROmAt.transpose();
                    const Eigen::Vector3f Anu = A * nu;
                    const Eigen::Vector3f ZaSy = Za * Sy;
                    const Eigen::Matrix3f SROmAt = S_inv * ROmAt;

                    const float da = -SROmAt.trace();
                    const float qa = (ZaSy + 2.0f * Anu).dot(Sy);
                    const float da_hqa = da + 0.5f * qa;
                    const Eigen::Vector3f S_ZaSy_Anu =
                        S_inv * (ZaSy + Anu);

                    Htr.template block<3, 1>(0, 3 + ix).noalias() +=
                        fmk * (da_hqa * Sy - S_ZaSy_Anu);

                    const Eigen::Matrix3f OmAt = (Om + Minf) * A.transpose();
                    const Eigen::Matrix3f SROmTZa = SROm.transpose() * Za;
                    const Eigen::Matrix3f Dba = (-OmAt + SROmTZa) * S_inv;

                    const Eigen::Matrix3f common =
                        -SROm + SySyTROm + Synut;
                    const Eigen::Matrix3f da_qa_db_da_qa_qb =
                        2.0f * da_hqa * common.transpose();

                    const Eigen::Matrix3f nuAnuT = nu * Anu.transpose();
                    const Eigen::Matrix3f AnuYT = Anu * y.transpose();
                    const Eigen::Matrix3f ZaSyYT = ZaSy * y.transpose();

                    const Eigen::Matrix3f dqadrb_b =
                        -2.0f * nu * ZaSy.transpose() * S_inv
                        -2.0f * SROm.transpose() *
                            (ZaSyYT + ZaSyYT.transpose()) * S_inv
                        +2.0f * OmAt * SySyT
                        -2.0f * SROm.transpose() *
                            (AnuYT + AnuYT.transpose()) * S_inv
                        -2.0f * nuAnuT * S_inv;

                    Delta.template block<3, 3>(3 * ix, 0).noalias() +=
                        0.5f * fmk *
                        (2.0f * Dba + da_qa_db_da_qa_qb + dqadrb_b);
                    Gamma.template block<3, 3>(3 * ix, 0).noalias() +=
                        fmk * (-SROm.transpose() + common.transpose());
                }
            }
        }

        J.template topRows<3>() = dFdt;
        for (int ix = 0; ix < kNumDims; ++ix) {
            J(kNumDims + ix) = (dFdR.transpose() * dRdu[ix]).trace();
        }

        if (compute_hessian) {
            H.template topLeftCorner<3, 3>() =
                Htr.template topLeftCorner<3, 3>();
            H.template topRightCorner<3, 3>() =
                Htr.template topRightCorner<3, 3>();
            H.template bottomLeftCorner<3, 3>() =
                H.template topRightCorner<3, 3>().transpose();

            for (int ix = 0; ix < kNumDims; ++ix) {
                for (int kx = 0; kx < kNumDims; ++kx) {
                    H(kNumDims + ix, kNumDims + kx) =
                        (Delta.template block<3, 3>(3 * kx, 0) *
                         dRdu[ix]).trace() +
                        (Gamma.template block<3, 3>(3 * kx, 0) *
                         d2Rdu2[ix][kx]).trace();
                }
            }
        }
    }

    void partialWrtU(const float& u1, const float& u2, const float& u3,
                     std::vector<Eigen::Matrix3f>& J) const {
        std::vector<std::vector<Eigen::Matrix3f>> H_tmp;
        partialWrtU(u1, u2, u3, false, J, H_tmp);
    }

    void partialWrtU(const float& u1, const float& u2, const float& u3,
                     std::vector<Eigen::Matrix3f>& J,
                     std::vector<std::vector<Eigen::Matrix3f>>& H) const {
        partialWrtU(u1, u2, u3, true, J, H);
    }

    void partialWrtU(const float& u1, const float& u2, const float& u3,
                     const bool compute_hessian_flag,
                     std::vector<Eigen::Matrix3f>& J,
                     std::vector<std::vector<Eigen::Matrix3f>>& H) const {
        float t3 = u1 * u1;
        float t4 = u2 * u2;
        float t5 = u3 * u3;
        float t6 = t3 + t4 + t5;
        float t7 = std::sqrt(t6);
        if (t7 < 1e-10f) {
            float J_arr[] =
                {0.f,  0.f,  0.f,    0.f, 0.f,  1.f,   0.f,-1.f, 0.f,
                 0.f,  0.f, -1.f,    0.f, 0.f,  0.f,   1.f, 0.f, 0.f,
                 0.f,  1.f,  0.f,   -1.f, 0.f,  0.f,   0.f, 0.f, 0.f};
            J.resize(kNumDims);
            for (int ix = 0; ix < kNumDims; ix++) {
                J[ix] = Eigen::Map<Eigen::Matrix3f>(
                    J_arr + ix * kNumDims * kNumDims);
            }
            if (compute_hessian_flag) {
                float H_arr[] =
                    {0.f,  0.f,  0.f,  0.f,  -1.f,  0.f,   0.f,  0.f,  -1.f,
                     0.f, 0.5f,  0.f, 0.5f,   0.f,  0.f,   0.f,  0.f,   0.f,
                     0.f,  0.f, 0.5f,  0.f,   0.f,  0.f,  0.5f,  0.f,   0.f,
                     0.f, 0.5f,  0.f, 0.5f,   0.f,  0.f,   0.f,  0.f,   0.f,
                     -1.f,  0.f,  0.f,  0.f,  0.0f,  0.f,   0.f,  0.f,  -1.f,
                     0.f,  0.f,  0.f,  0.f,   0.f, 0.5f,   0.f, 0.5f,   0.f,
                     0.f,  0.f, 0.5f,  0.f,   0.f,  0.f,  0.5f,  0.f,   0.f,
                     0.f,  0.f,  0.f,  0.f,   0.f, 0.5f,   0.f, 0.5f,   0.f,
                     -1.f,  0.f,  0.f,  0.f,  -1.f,  0.f,   0.f,  0.f,   0.f};
                H.resize(kNumDims);
                for (int ix = 0; ix < kNumDims; ix++) {
                    H[ix].resize(kNumDims);
                    for (int jx = 0; jx < kNumDims; jx++) {
                        const std::uint32_t offset =
                            kNumDims * kNumDims * (ix + jx * kNumDims);
                        H[ix][jx] = Eigen::Map<Eigen::Matrix3f>(
                            H_arr + offset);
                    }
                }
            }
            return;
        }

        float t8 = t7 * 0.5f;
        float t2 = std::sin(t8);
        float t9 = t2 * t2;
        float tt6 = t6 * t6;
        float t10 = 1.0f / tt6;
        float t11 = std::cos(t8);
        float st6 = std::sqrt(t6);
        float t12 = 1.0f / (t6 * st6);
        float t13 = 1.0f / t6;
        float t14 = 1.0f / st6;
        float t15 = t9 * t13 * u1 * 2.0f;
        float t16 = t11 * t11;
        float t17 = t2 * t11 * t12 * u1 * u2 * u3 * 2.0f;
        float t18 = t9 * t13 * u2 * u3;
        float t19 = t2 * t11 * t12 * u2 * u3 * 2.0f;
        float t20 = t9 * t13 * u2 * 2.0f;
        float t21 = t9 * t13 * u1 * u3;
        float t22 = t2 * t11 * t12 * u1 * u3 * 2.0f;
        float t23 = t2 * t3 * t11 * t12 * u2 * 2.0f;
        float t24 = t13 * t16 * u2 * u3;
        float t25 = t2 * t4 * t11 * t12 * u1 * 2.0f;
        float t26 = t2 * t11 * t14 * 2.0f;
        float t27 = t5 * t9 * t13;
        float t28 = t2 * t5 * t11 * t12 * 2.0f;
        float t29 = t4 * t9 * t10 * u1 * 2.0f;
        float t30 = t5 * t9 * t10 * u1 * 2.0f;
        float t31 = t2 * t3 * t11 * t12 * u1;
        float t32 = t4 * t9 * t10 * u2 * 2.0f;
        float t33 = t5 * t9 * t10 * u2 * 2.0f;
        float t34 = t2 * t3 * t11 * t12 * u2;
        float t35 = t9 * t13 * u3 * 2.0f;
        float t36 = t5 * t9 * t10 * u3 * 2.0f;
        float t37 = t4 * t9 * t10 * u3 * 2.0f;
        float t38 = t2 * t3 * t11 * t12 * u3;
        float t39 = t13 * t16 * u1 * u2;
        float t40 = t13 * t16 * u1 * u3;
        float t41 = t9 * t13 * u1 * u2;
        float t42 = t2 * t11 * t12 * u1 * u2 * 2.0f;
        float t43 = t2 * t3 * t11 * t12 * u3 * 2.0f;
        float t44 = t4 * t13 * t16;
        float t45 = t2 * t5 * t11 * t12 * u1 * 2.0f;
        float t46 = t3 * t9 * t13;
        float t47 = t2 * t3 * t11 * t12 * 2.0f;
        float t48 = t2 * t4 * t11 * t12 * u3 * 2.0f;
        float t49 = t2 * t5 * t11 * t12 * u2 * 2.0f;
        float t50 = t3 * t9 * t10 * u1 * 2.0f;
        float t51 = t2 * t4 * t11 * t12 * u1;
        float t52 = t3 * t9 * t10 * u2 * 2.0f;
        float t53 = t2 * t4 * t11 * t12 * u2;
        float t54 = t3 * t9 * t10 * u3 * 2.0f;
        float t55 = t2 * t4 * t11 * t12 * u3;
        float t56 = t3 * t3;
        float t57 = 1.0f / (tt6 * t6);
        float t58 = 1.0f / (tt6 * st6);
        float t59 = t3 * t10 * t16 * u1 * u2 * 0.5f;
        float t60 = t4 * t9 * t10 * u1 * u2 * 0.5f;
        float t61 = t3 * t9 * t57 * u1 * u2 * 8.0f;
        float t62 = t9 * t13 * u1 * u2 * 0.5f;
        float t63 = t2 * t11 * t12 * u1 * u2;
        float t64 = t5 * t9 * t10 * u1 * u2 * 0.5f;
        float t65 = t2 * t4 * t11 * t58 * u1 * u2 * 5.0f;
        float t66 = t2 * t5 * t11 * t58 * u1 * u2 * 5.0f;
        float t172 = t4 * t10 * t16 * u1 * u2 * 0.5f;
        float t173 = t3 * t9 * t10 * u1 * u2 * 0.5f;
        float t174 = t4 * t9 * t57 * u1 * u2 * 8.0f;
        float t175 = t13 * t16 * u1 * u2 * 0.5f;
        float t176 = t5 * t10 * t16 * u1 * u2 * 0.5f;
        float t177 = t5 * t9 * t57 * u1 * u2 * 8.0f;
        float t178 = t2 * t3 * t11 * t58 * u1 * u2 * 5.0f;
        float t67 = t59 + t60 + t61 + t62 + t63 + t64 + t65 + t66
            - t172 - t173 - t174 - t175 - t176 - t177 - t178;
        float t68 = t9 * t13 * 2.0f;
        float t69 = t5 * t9 * t10 * 2.0f;
        float t70 = t4 * t4;
        float t71 = t3 * t4 * t9 * t10 * 0.5f;
        float t72 = t2 * t3 * t4 * t11 * t58 * 5.0f;
        float t73 = t3 * t10 * t16 * u1 * u3 * 0.5f;
        float t74 = t5 * t9 * t10 * u1 * u3 * 0.5f;
        float t75 = t3 * t9 * t57 * u1 * u3 * 8.0f;
        float t76 = t9 * t13 * u1 * u3 * 0.5f;
        float t77 = t2 * t11 * t12 * u1 * u3;
        float t78 = t4 * t9 * t10 * u1 * u3 * 0.5f;
        float t79 = t2 * t5 * t11 * t58 * u1 * u3 * 5.0f;
        float t80 = t2 * t4 * t11 * t58 * u1 * u3 * 5.0f;
        float t179 = t5 * t10 * t16 * u1 * u3 * 0.5f;
        float t180 = t3 * t9 * t10 * u1 * u3 * 0.5f;
        float t181 = t5 * t9 * t57 * u1 * u3 * 8.0f;
        float t182 = t13 * t16 * u1 * u3 * 0.5f;
        float t183 = t4 * t10 * t16 * u1 * u3 * 0.5f;
        float t184 = t4 * t9 * t57 * u1 * u3 * 8.0f;
        float t185 = t2 * t3 * t11 * t58 * u1 * u3 * 5.0f;
        float t81 = t73 + t74 + t75 + t76 + t77 + t78 + t79 + t80
            - t179 - t180 - t181 - t182 - t183 - t184 - t185;
        float t82 = t5 * t9 * t10 * u2 * u3 * 0.5f;
        float t83 = t4 * t9 * t10 * u2 * u3 * 0.5f;
        float t84 = t9 * t13 * u2 * u3 * 0.5f;
        float t85 = t9 * t10 * u2 * u3 * 8.0f;
        float t86 = t3 * t10 * t16 * u2 * u3 * 0.5f;
        float t87 = t3 * t9 * t57 * u2 * u3 * 8.0f;
        float t88 = t2 * t5 * t11 * t58 * u2 * u3 * 5.0f;
        float t89 = t2 * t4 * t11 * t58 * u2 * u3 * 5.0f;
        float t191 = t5 * t10 * t16 * u2 * u3 * 0.5f;
        float t192 = t4 * t10 * t16 * u2 * u3 * 0.5f;
        float t193 = t5 * t9 * t57 * u2 * u3 * 8.0f;
        float t194 = t4 * t9 * t57 * u2 * u3 * 8.0f;
        float t195 = t13 * t16 * u2 * u3 * 0.5f;
        float t196 = t3 * t9 * t10 * u2 * u3 * 0.5f;
        float t197 = t2 * t3 * t11 * t58 * u2 * u3 * 5.0f;
        float t90 = t82 + t83 + t84 + t85 + t86 + t87 + t88 + t89
            - t191 - t192 - t193 - t194 - t195 - t196 - t197
            - t2 * t11 * t12 * u2 * u3 * 3.0f;
        float t91 = t4 * t9 * t10 * 2.0f;
        float t92 = t5 * t5;
        float t93 = t2 * t3 * t11 * t12;
        float t94 = t3 * t5 * t9 * t10 * 0.5f;
        float t95 = t4 * t5 * t9 * t10 * 0.5f;
        float t96 = t2 * t3 * t5 * t11 * t58 * 5.0f;
        float t97 = t2 * t4 * t5 * t11 * t58 * 5.0f;
        float t98 = t2 * t4 * t11 * t12 * 2.0f;

        float J_arr[] = {
            t15+t29+t30+t31-t3*t9*t10*u1*2.0f-t2*t11*t14*u1-t2*t4*t11*t12*u1-t2*t5*t11*t12*u1,
            t20-t21-t22+t23+t40-t3*t9*t10*u2*4.0f,
            t35-t39+t41+t42+t43-t3*t9*t10*u3*4.0f,
            t20+t21+t22+t23-t3*t9*t10*u2*4.0f-t13*t16*u1*u3,
            -t15-t29+t30-t31+t50+t51-t2*t11*t14*u1-t2*t5*t11*t12*u1,
            t17+t26-t46-t47+t3*t13*t16-t9*t10*u1*u2*u3*4.0f,
            t35+t39+t43-t3*t9*t10*u3*4.0f-t9*t13*u1*u2-t2*t11*t12*u1*u2*2.0f,
            t17-t26+t46+t47-t3*t13*t16-t9*t10*u1*u2*u3*4.0f,
            -t15+t29-t30-t31+t50-t51-t2*t11*t14*u1+t2*t5*t11*t12*u1,

            t32+t33+t34-t9*t13*u2*2.0f-t3*t9*t10*u2*2.0f-t2*t11*t14*u2-t2*t4*t11*t12*u2-t2*t5*t11*t12*u2,
            t15-t18-t19+t24+t25-t4*t9*t10*u1*4.0f,
            t17-t26-t44+t98+t4*t9*t13-t9*t10*u1*u2*u3*4.0f,
            t15+t18+t19+t25-t4*t9*t10*u1*4.0f-t13*t16*u2*u3,
            t20-t32+t33-t34+t52+t53-t2*t11*t14*u2-t2*t5*t11*t12*u2,
            t35+t39-t41-t42+t48-t4*t9*t10*u3*4.0f,
            t17+t26+t44-t4*t9*t13-t2*t4*t11*t12*2.0f-t9*t10*u1*u2*u3*4.0f,
            t35-t39+t41+t42+t48-t4*t9*t10*u3*4.0f,
            -t20+t32-t33-t34+t52-t53-t2*t11*t14*u2+t2*t5*t11*t12*u2,

            t36+t37+t38-t9*t13*u3*2.0f-t3*t9*t10*u3*2.0f-t2*t11*t14*u3-t2*t4*t11*t12*u3-t2*t5*t11*t12*u3,
            t17+t26-t27-t28+t5*t13*t16-t9*t10*u1*u2*u3*4.0f,
            t15+t18+t19-t24+t45-t5*t9*t10*u1*4.0f,
            t17+t27+t28-t2*t11*t14*2.0f-t5*t13*t16-t9*t10*u1*u2*u3*4.0f,
            -t35+t36-t37-t38+t54+t55-t2*t11*t14*u3-t2*t5*t11*t12*u3,
            t20-t21-t22+t40+t49-t5*t9*t10*u2*4.0f,
            t15-t18-t19+t24+t45-t5*t9*t10*u1*4.0f,
            t20+t21+t22-t40+t49-t5*t9*t10*u2*4.0f,
            t35-t36+t37-t38+t54-t55-t2*t11*t14*u3+t2*t5*t11*t12*u3};
        J.resize(kNumDims);
        for (int ix = 0; ix < kNumDims; ix++) {
            J[ix] = Eigen::Map<Eigen::Matrix3f>(
                J_arr + ix * kNumDims * kNumDims);
        }

        if (!compute_hessian_flag) {
            return;
        }

        float t99 = t3*t4*t10*t16;
        float t100 = t3*t4*t9*t57*16.f;
        float t101 = t10*t16*u1*u2*u3*3.0f;
        float t123 = t3*t9*t10*4.0f;
        float t124 = t9*t10*u1*u2*u3*3.0f;
        float t125 = t2*t11*t58*u1*u2*u3*6.0f;
        float t148 = t4*t9*t10*4.0f;
        float t149 = t3*t4*t9*t10;
        float t150 = t2*t3*t4*t11*t58*10.f;
        float t102 = t17+t47+t68+t98+t99+t100+t101-t123-t124-t125-t148-t149-t150;
        float t103 = t9*t13*u3;
        float t104 = t2*t11*t12*u3*2.0f;
        float t105 = t2*t11*t12*u1*u2*6.0f;
        float t106 = t9*t13*u1;
        float t107 = t2*t11*t12*u1*2.0f;
        float t108 = t5*t10*t16*u1*3.0f;
        float t109 = t3*t10*t16*u2*u3;
        float t110 = t3*t9*t57*u2*u3*16.f;
        float t119 = t13*t16*u1;
        float t120 = t9*t10*u2*u3*4.0f;
        float t121 = t3*t9*t10*u2*u3;
        float t122 = t2*t3*t11*t58*u2*u3*10.f;
        float t151 = t5*t9*t10*u1*3.0f;
        float t152 = t2*t5*t11*t58*u1*6.0f;
        float t111 = t19+t45+t106+t107+t108+t109+t110-t119-t120-t121-t122-t151-t152;
        float t112 = t9*t13*u2;
        float t113 = t2*t11*t12*u2*2.0f;
        float t114 = t5*t10*t16*u2*3.0f;
        float t115 = t4*t10*t16*u1*u3;
        float t116 = t4*t9*t57*u1*u3*16.f;
        float t118 = t13*t16*u2;
        float t129 = t9*t10*u1*u3*4.0f;
        float t130 = t4*t9*t10*u1*u3;
        float t131 = t2*t4*t11*t58*u1*u3*10.f;
        float t142 = t5*t9*t10*u2*3.0f;
        float t144 = t2*t5*t11*t58*u2*6.0f;
        float t117 = t22+t49+t112+t113+t114+t115+t116-t118-t129-t130-t131-t142-t144;
        float t126 = t4*t9*t10*u1*3.0f;
        float t127 = t2*t4*t11*t58*u1*6.0f;
        float t214 = t4*t10*t16*u1*3.0f;
        float t128 = t19-t25-t106-t107+t109+t110+t119-t120-t121-t122+t126+t127-t214;
        float t132 = t4*t10*t16*u3*3.0f;
        float t133 = t5*t10*t16*u1*u2;
        float t134 = t5*t9*t57*u1*u2*16.f;
        float t135 = t3*t5*t10*t16;
        float t136 = t3*t5*t9*t57*16.f;
        float t215 = t5*t9*t10*4.0f;
        float t223 = t3*t5*t9*t10;
        float t224 = t2*t3*t5*t11*t58*10.f;
        float t137 = -t17+t28+t47+t68-t101-t123+t124+t125+t135+t136-t215-t223-t224;
        float t138 = t13*t16*u3;
        float t139 = t4*t9*t10*u3*3.0f;
        float t140 = t2*t4*t11*t58*u3*6.0f;
        float t162 = t9*t10*u1*u2*4.0f;
        float t163 = t5*t9*t10*u1*u2;
        float t164 = t2*t5*t11*t58*u1*u2*10.f;
        float t141 = t42-t48-t103-t104-t132+t133+t134+t138+t139+t140-t162-t163-t164;
        float t143 = t2*t11*t12*u1*u3*6.0f;
        float t145 = t3*t10*t16*u3*3.0f;
        float t146 = t3*t10*t16*u1*u2;
        float t147 = t3*t9*t57*u1*u2*16.f;
        float t153 = -t17+t47+t68+t98+t99+t100-t101-t123+t124+t125-t148-t149-t150;
        float t154 = t4*t10*t16*u1*u2;
        float t155 = t4*t9*t57*u1*u2*16.f;
        float t156 = t19-t45-t106-t107-t108+t109+t110+t119-t120-t121-t122+t151+t152;
        float t157 = t22-t49-t112-t113-t114+t115+t116+t118-t129-t130-t131+t142+t144;
        float t158 = t9*t13*u3*3.0f;
        float t159 = t5*t10*t16*u3*3.0f;
        float t160 = t2*t11*t12*u3*6.0f;
        float t161 = t2*t5*t11*t12*u3*2.0f;
        float t165 = t10*t16*t56*0.5f;
        float t166 = t3*t9*t13*0.5f;
        float t167 = t9*t56*t57*8.0f;
        float t168 = t3*t4*t10*t16*0.5f;
        float t169 = t3*t5*t10*t16*0.5f;
        float t170 = t3*t4*t9*t57*8.0f;
        float t171 = t3*t5*t9*t57*8.0f;
        float t186 = -t59-t60-t61+t62+t63+t64-t65+t66+t172+t173+t174-t175-t176-t177+t178;
        float t187 = t4*t9*t13*0.5f;
        float t188 = t4*t9*t10*10.f;
        float t189 = t9*t10*t70*0.5f;
        float t190 = t2*t11*t58*t70*5.0f;
        float t198 = t9*t10*u1*u3*8.0f;
        float t199 = -t73+t74-t75+t76-t78+t79-t80-t179+t180-t181-t182+t183+t184+t185+t198-t2*t11*t12*u1*u3*3.0f;
        float t200 = t2*t11*t12*u2*u3;
        float t201 = t82-t83+t84-t86-t87+t88-t89-t191+t192-t193+t194-t195+t196+t197+t200;
        float t202 = t3*t9*t10*2.0f;
        float t203 = t5*t9*t13*0.5f;
        float t204 = t5*t9*t10*10.f;
        float t205 = t9*t10*t92*0.5f;
        float t206 = t2*t4*t11*t12;
        float t207 = t2*t11*t58*t92*5.0f;
        float t208 = t3*t9*t10*u2*3.0f;
        float t209 = t2*t3*t11*t58*u2*6.0f;
        float t210 = t3*t9*t10*u3*3.0f;
        float t211 = t2*t3*t11*t58*u3*6.0f;
        float t212 = t3*t10*t16*u2*3.0f;
        float t213 = t22+t23+t112+t113+t115+t116-t118-t129-t130-t131-t208-t209+t212;
        float t216 = t42+t43+t103+t104+t133+t134-t138+t145-t162-t163-t164-t210-t211;
        float t217 = t4*t5*t10*t16;
        float t218 = t4*t5*t9*t57*16.f;
        float t240 = t4*t5*t9*t10;
        float t241 = t2*t4*t5*t11*t58*10.f;
        float t219 = t17+t28+t68+t98+t101-t124-t125-t148-t215+t217+t218-t240-t241;
        float t220 = t2*t11*t12*u2*u3*6.0f;
        float t221 = t3*t10*t16*u1*u3;
        float t222 = t3*t9*t57*u1*u3*16.f;
        float t225 = t19+t25+t106+t107+t109+t110-t119-t120-t121-t122-t126-t127+t214;
        float t226 = t13*t16*u2*3.0f;
        float t227 = t4*t9*t10*u2*3.0f;
        float t228 = t2*t4*t11*t58*u2*6.0f;
        float t229 = t17+t28+t47+t68+t101-t123-t124-t125+t135+t136-t215-t223-t224;
        float t230 = t42+t48+t103+t104+t132+t133+t134-t138-t139-t140-t162-t163-t164;
        float t231 = t5*t10*t16*u1*u3;
        float t232 = t5*t9*t57*u1*u3*16.f;
        float t233 = t9*t13*u1*3.0f;
        float t234 = t3*t10*t16*u1*3.0f;
        float t235 = t2*t11*t12*u1*6.0f;
        float t236 = t2*t3*t11*t12*u1*2.0f;
        float t237 = t22-t23-t112-t113+t115+t116+t118-t129-t130-t131+t208+t209-t212;
        float t238 = t4*t10*t16*u2*u3;
        float t239 = t4*t9*t57*u2*u3*16.f;
        float t242 = t42-t43-t103-t104+t133+t134+t138-t145-t162-t163-t164+t210+t211;
        float t243 = -t17+t28+t68+t98-t101+t124+t125-t148-t215+t217+t218-t240-t241;
        float t244 = t5*t10*t16*u2*u3;
        float t245 = t5*t9*t57*u2*u3*16.f;
        float t246 = t3*t9*t10*10.f;
        float t247 = t9*t10*t56*0.5f;
        float t248 = t2*t11*t56*t58*5.0f;
        float t249 = t9*t10*u1*u2*8.0f;
        float t250 = -t59+t60-t61+t62-t64+t65-t66-t172+t173-t174-t175+t176+t177+t178+t249-t2*t11*t12*u1*u2*3.0f;
        float t251 = t10*t16*t70*0.5f;
        float t252 = t9*t57*t70*8.0f;
        float t253 = t2*t5*t11*t12;
        float t254 = t4*t5*t10*t16*0.5f;
        float t255 = t4*t5*t9*t57*8.0f;
        float t256 = -t73-t74-t75+t76+t77+t78-t79+t80+t179+t180+t181-t182-t183-t184+t185;
        float t257 = -t82+t83+t84-t86-t87-t88+t89+t191-t192+t193-t194-t195+t196+t197+t200;
        float H_arr[] = {
            t68+t69+t71+t72+t91+t94+t96+t165+t166+t167-t3*t9*t10*10.f-t2*t11*t14 - t3*t13*t16*0.5f-t9*t10*t56*0.5f+t2*t3*t11*t12*6.0f - t2*t4*t11*t12-t2*t5*t11*t12-t3*t4*t10*t16*0.5f-t3*t5*t10*t16*0.5f - t3*t4*t9*t57*8.0f-t3*t5*t9*t57*8.0f-t2*t11*t56*t58*5.0f,
            -t43-t103-t104+t105+t138-t145+t146+t147+t210+t211-t9*t10*u1*u2*12.f-t3*t9*t10*u1*u2-t2*t3*t11*t58*u1*u2*10.f,
            t23+t112+t113-t118+t143-t208-t209+t212+t221+t222-t9*t10*u1*u3*12.f-t3*t9*t10*u1*u3-t2*t3*t11*t58*u1*u3*10.f,
            t43+t103+t104+t105+t145+t146+t147-t13*t16*u3-t3*t9*t10*u3*3.0f-t9*t10*u1*u2*12.f - t2*t3*t11*t58*u3*6.0f-t3*t9*t10*u1*u2-t2*t3*t11*t58*u1*u2*10.f,
            -t68+t69-t71-t72-t91+t94+t96-t165+t166-t167+t168-t169+t170-t171+t206+t246+t247+t248 - t2*t11*t14-t3*t13*t16*0.5f-t2*t3*t11*t12*4.0f-t2*t5*t11*t12,
            t19+t109+t110-t120-t121-t122-t233-t234-t235-t236+t13*t16*u1*3.0f+t3*t9*t10*u1*3.0f+t2*t3*t11*t58*u1*6.0f,
            -t23-t112-t113+t118+t143+t208+t209+t221+t222-t3*t10*t16*u2*3.0f-t9*t10*u1*u3*12.f - t3*t9*t10*u1*u3-t2*t3*t11*t58*u1*u3*10.f,
            t19+t109+t110-t120-t121-t122+t233+t234+t235+t236-t13*t16*u1*3.0f-t3*t9*t10*u1*3.0f-t2*t3*t11*t58*u1*6.0f,
            -t68-t69+t71+t72+t91-t94-t96-t165+t166-t167-t168+t169-t170+t171-t206+t246+t247+t248+t253 - t2*t11*t14-t3*t13*t16*0.5f-t2*t3*t11*t12*4.0f,
            t67, t153, t225,
            t102, t186, t237,
            t128, t213, t250,
            t81,  t156, t229,
            t111, t199, t242,
            t137, t216, t256,
            t67,  t153, t225,
            t102, t186, t237,
            t128, t213, t250,
            -t68+t69-t71-t72+t93+t95+t97+t168+t170+t187+t188+t189+t190-t3*t9*t10*2.0f - t2*t11*t14-t4*t13*t16*0.5f-t10*t16*t70*0.5f-t9*t57*t70*8.0f - t2*t4*t11*t12*4.0f-t2*t5*t11*t12-t4*t5*t10*t16*0.5f-t4*t5*t9*t57*8.0f,
            -t48-t103-t104+t105-t132+t138+t139+t140+t154+t155-t9*t10*u1*u2*12.f-t4*t9*t10*u1*u2-t2*t4*t11*t58*u1*u2*10.f,
            t22+t115+t116-t129-t130-t131-t226-t227-t228+t9*t13*u2*3.0f+t2*t11*t12*u2*6.0f+t4*t10*t16*u2*3.0f+t2*t4*t11*t12*u2*2.0f,
            t48+t103+t104+t105+t132+t154+t155-t13*t16*u3-t4*t9*t10*u3*3.0f-t9*t10*u1*u2*12.f - t2*t4*t11*t58*u3*6.0f-t4*t9*t10*u1*u2-t2*t4*t11*t58*u1*u2*10.f,
            t68+t69+t71+t72-t93+t95+t97-t168-t170+t187-t188-t189-t190+t202+t251+t252-t2*t11*t14 - t4*t13*t16*0.5f+t2*t4*t11*t12*6.0f-t2*t5*t11*t12-t4*t5*t10*t16*0.5f-t4*t5*t9*t57*8.0f,
            -t25-t106-t107+t119+t126+t127-t214+t220+t238+t239-t9*t10*u2*u3*12.f-t4*t9*t10*u2*u3-t2*t4*t11*t58*u2*u3*10.f,
            t22+t115+t116-t129-t130-t131+t226+t227+t228-t9*t13*u2*3.0f-t2*t11*t12*u2*6.0f-t4*t10*t16*u2*3.0f-t2*t4*t11*t12*u2*2.0f,
            t25+t106+t107-t119-t126-t127+t214+t220+t238+t239-t9*t10*u2*u3*12.f-t4*t9*t10*u2*u3-t2*t4*t11*t58*u2*u3*10.f,
            -t68-t69+t71+t72-t93-t95-t97-t168-t170+t187+t188+t189+t190+t202-t251-t252+t253+t254+t255 - t2*t11*t14-t4*t13*t16*0.5f-t2*t4*t11*t12*4.0f,
            t90,  t157, t230,
            t117, t201, t243,
            t141, t219, t257,
            t81,  t156, t229,
            t111, t199, t242,
            t137, t216, t256,
            t90,  t157, t230,
            t117, t201, t243,
            t141, t219, t257,
            -t68+t91+t93-t94+t95-t96+t97+t169+t171+t203+t204+t205+t207-t3*t9*t10*2.0f - t2*t11*t14-t5*t13*t16*0.5f-t10*t16*t92*0.5f-t9*t57*t92*8.0f - t2*t4*t11*t12-t2*t5*t11*t12*4.0f-t4*t5*t10*t16*0.5f-t4*t5*t9*t57*8.0f,
            t42+t133+t134-t158-t159-t160-t161-t162-t163-t164+t13*t16*u3*3.0f+t5*t9*t10*u3*3.0f+t2*t5*t11*t58*u3*6.0f,
            t49+t112+t113+t114-t118-t142+t143-t144+t231+t232-t9*t10*u1*u3*12.f-t5*t9*t10*u1*u3-t2*t5*t11*t58*u1*u3*10.f,
            t42+t133+t134+t158+t159+t160+t161-t13*t16*u3*3.0f-t5*t9*t10*u3*3.0f-t9*t10*u1*u2*4.0f - t2*t5*t11*t58*u3*6.0f-t5*t9*t10*u1*u2-t2*t5*t11*t58*u1*u2*10.f,
            -t68-t91-t93+t94-t95+t96-t97-t169-t171+t202+t203+t204+t205+t206+t207+t254+t255 - t2*t11*t14-t5*t13*t16*0.5f-t10*t16*t92*0.5f-t9*t57*t92*8.0f-t2*t5*t11*t12*4.0f,
            -t45-t106-t107-t108+t119+t151+t152+t220+t244+t245-t9*t10*u2*u3*12.f-t5*t9*t10*u2*u3-t2*t5*t11*t58*u2*u3*10.f,
            -t49-t112-t113-t114+t118+t142+t143+t144+t231+t232-t9*t10*u1*u3*12.f-t5*t9*t10*u1*u3-t2*t5*t11*t58*u1*u3*10.f,
            t45+t106+t107+t108-t119-t151-t152+t220+t244+t245-t9*t10*u2*u3*12.f-t5*t9*t10*u2*u3-t2*t5*t11*t58*u2*u3*10.f,
            t68+t91-t93+t94+t95+t96+t97-t169-t171+t202+t203-t204-t205-t206-t207-t254-t255-t2*t11*t14-t5*t13*t16*0.5f+t10*t16*t92*0.5f+t9*t57*t92*8.0f+t2*t5*t11*t12*6.0f
        };

        H.resize(kNumDims);
        for (int ix = 0; ix < kNumDims; ix++) {
            H[ix].resize(kNumDims);
        }
        for (int ix = 0; ix < kNumDims; ix++) {
            for (int jx = 0; jx < kNumDims; jx++) {
                const std::uint32_t offset =
                    kNumDims * kNumDims * (jx + ix * kNumDims);
                H[jx][ix] = Eigen::Map<Eigen::Matrix3f>(H_arr + offset);
            }
        }
    }

    void updateInflation(const Eigen::Matrix<float, 6, 1>& grad,
                         const Eigen::Matrix<float, 6, 6>& hess) const {
        const Eigen::Matrix<float, 6, 6> damped =
            hess - static_cast<float>(std::max(options_.hessian_damping, 1e-9)) *
                       Eigen::Matrix<float, 6, 6>::Identity();
        const Eigen::Matrix<float, 6, 1> dx = -damped.ldlt().solve(grad);
        if (!dx.allFinite()) {
            return;
        }

        const float trans_step = dx.template head<3>().norm();
        const float rot_step = dx.template tail<3>().norm();
        const float rot_norm = 5.0f * static_cast<float>(M_PI) / 180.0f;
        const float normalized =
            std::max(trans_step / 0.1f, rot_step / rot_norm);
        const float alpha = std::clamp(normalized, 0.0f, 1.0f);
        inflation_factor_ =
            std::clamp(alpha * initial_inflation_, 0.0f, inflation_factor_);
    }

    void updateInflation(const column_vector& grad,
                         const general_matrix& hess) const {
        constexpr int n = 6;
        dlib::matrix<float> damped = hess;
        for (int i = 0; i < n; ++i) {
            damped(i, i) -=
                static_cast<float>(std::max(options_.hessian_damping, 1e-9));
        }

        column_vector dx;
        try {
            dx = -dlib::inv(damped) * grad;
        } catch (...) {
            return;
        }

        const float trans_step =
            std::sqrt(dx(0) * dx(0) + dx(1) * dx(1) + dx(2) * dx(2));
        const float rot_step =
            std::sqrt(dx(3) * dx(3) + dx(4) * dx(4) + dx(5) * dx(5));
        const float rot_norm = 5.0f * static_cast<float>(M_PI) / 180.0f;
        const float normalized =
            std::max(trans_step / 0.1f, rot_step / rot_norm);
        const float alpha = std::clamp(normalized, 0.0f, 1.0f);
        inflation_factor_ =
            std::clamp(alpha * initial_inflation_, 0.0f, inflation_factor_);
    }

    ComponentCache source_cache_;
    ComponentCache target_cache_;
    float Et_ = 1.0f;
    const Eigen::Vector3f translation_prior_;
    const Eigen::Vector3f rotation_prior_u_;
    const D2DRegistrationOptions options_;
    const float initial_inflation_;
    mutable float inflation_factor_;
};

float runD2DRegistration(
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>& Tinit,
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>& Tout,
    float& out_coarse_score,
    bool& out_fast_path_used,
    const D2DRegistrationOptions& options) {
    out_fast_path_used = false;
    ColumnVector x(6);
    const Eigen::Vector3f t = Tinit.translation();
    const Eigen::AngleAxisf aa(Tinit.rotation());
    Eigen::Vector3f u = Eigen::Vector3f::Zero();
    if (std::isfinite(aa.angle()) && aa.axis().allFinite()) {
        u = aa.axis() * aa.angle();
    }
    x = t(0), t(1), t(2), u(0), u(1), u(2);

    GmmRegistrationModel coarse_model(source_gmm, target_gmm, t, u, options);
    if (options.initial_score_fast_path &&
        std::isfinite(options.initial_score_threshold)) {
        const float initial_score = coarse_model(x);
        if (std::isfinite(initial_score) &&
            initial_score >= static_cast<float>(
                options.initial_score_threshold + options.initial_score_margin)) {
            out_coarse_score = initial_score;
            Tout = Tinit;
            out_fast_path_used = true;
            return initial_score;
        }
    }

    const float coarse_score = dlib::find_max_trust_region(
        dlib::objective_delta_stop_strategy(
            options.objective_delta_stop,
            std::max<unsigned long>(1, options.coarse_max_iterations)),
        coarse_model,
        x,
        options.initial_trust_radius);
    out_coarse_score = coarse_score;

    float score = coarse_score;
    if (options.fine_refine_enable &&
        std::isfinite(coarse_score) &&
        coarse_score >= static_cast<float>(options.fine_min_coarse_score)) {
        ColumnVector x_coarse = x;
        const Eigen::Vector3f t_coarse(x_coarse(0), x_coarse(1), x_coarse(2)); // 0, 1, 2 --> translation tx, ty, tz
        const Eigen::Vector3f u_coarse(x_coarse(3), x_coarse(4), x_coarse(5)); // 3, 4, 5 --> rotation vector (axis-angle representation) ux, uy, uz
        const float fine_score = dlib::find_max_trust_region(
            dlib::objective_delta_stop_strategy(
                options.fine_objective_delta_stop,
                std::max<unsigned long>(1, options.fine_max_iterations)),
            GmmRegistrationModel(source_gmm, target_gmm, t_coarse, u_coarse,
                                 options,
                                 kFineInitialInflation),
            x,
            options.fine_initial_trust_radius);
        if (std::isfinite(fine_score) && fine_score > coarse_score) {
            score = fine_score;
        } else {
            x = x_coarse;
        }
    }

    Tout = transformFromState(x);
    return score;
}

} // namespace

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const D2DRegistrationOptions& options) {

    RegistrationResult result;

    auto Tinit = toAffine(T_init);
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;

    float coarse_score = -std::numeric_limits<float>::infinity();
    bool fast_path_used = false;
    float score = runD2DRegistration(source_gmm, target_gmm, Tinit, Tout,
                                     coarse_score, fast_path_used, options);

    result.transform = Tout.matrix();
    result.score = score;
    result.coarse_score = coarse_score;
    result.n_source = static_cast<int>(source_gmm.getNClusters());
    result.n_target = static_cast<int>(target_gmm.getNClusters());
    result.success = std::isfinite(score) && !result.transform.hasNaN();
    result.initial_score_fast_path_used = fast_path_used;
    return result;
}

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const D2DRegistrationOptions& options) {

    RegistrationResult result;

    auto Tinit = toAffine(T_init);
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;

    float coarse_score = -std::numeric_limits<float>::infinity();
    bool fast_path_used = false;
    float score = runD2DRegistration(source_gmm, target_gmm, Tinit, Tout,
                                     coarse_score, fast_path_used, options);

    result.transform = Tout.matrix();
    result.score = score;
    result.coarse_score = coarse_score;
    result.n_source = static_cast<int>(source_gmm.getNClusters());
    result.n_target = static_cast<int>(target_gmm.getNClusters());
    result.success = std::isfinite(score) && !result.transform.hasNaN();
    result.initial_score_fast_path_used = fast_path_used;
    return result;
}

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path,
    const D2DRegistrationOptions& options) {

    gmm_utils::GMM3f source_gmm, target_gmm;
    source_gmm.load(source_path);
    source_gmm.makeCovsIsoplanar();
    target_gmm.load(target_path);
    target_gmm.makeCovsIsoplanar();
    return isoplanarRegistration(T_init, source_gmm, target_gmm, options);
}

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path,
    const D2DRegistrationOptions& options) {

    gmm_utils::GMM3f source_gmm, target_gmm;
    source_gmm.load(source_path);
    target_gmm.load(target_path);
    return anisotropicRegistration(T_init, source_gmm, target_gmm, options);
}

} // namespace gmmslam
