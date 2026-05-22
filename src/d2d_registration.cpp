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
constexpr float kInitialInflation = 0.2f * 0.2f; // 20cm inflation
constexpr float kGateRadiusMultiplierSq = 16.0f;
constexpr float kPi32 = 1.0f / (2.0f * static_cast<float>(M_PI) *
                                std::sqrt(2.0f * static_cast<float>(M_PI)));

using ColumnVector = dlib::matrix<float, 0, 1>;

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
                         const gmm_utils::GMM3f& target_gmm)
        : source_gmm_(source_gmm),
          target_gmm_(target_gmm),
          inflation_factor_(kInitialInflation) {
        const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        const Eigen::Vector3f t = Eigen::Vector3f::Zero();
        Et_ = rawScore(R, t, target_gmm_, target_gmm_, 1.0f,
                       inflation_factor_);
        if (!std::isfinite(Et_) || Et_ <= std::numeric_limits<float>::min()) {
            Et_ = 1.0f;
        }
    }

    float operator()(const column_vector& x) const {
        const Eigen::Vector3f t(x(0), x(1), x(2));
        const Eigen::Vector3f u(x(3), x(4), x(5));
        return rawScore(rotationFromAxisAngleVector(u), t,
                        target_gmm_, source_gmm_, Et_, inflation_factor_);
    }

    void get_derivative_and_hessian(const column_vector& x,
                                    column_vector& der,
                                    general_matrix& hess) const {
        constexpr int n = 6;
        const float f0 = (*this)(x);
        der.set_size(n);
        hess.set_size(n, n);
        for (int r = 0; r < n; ++r) {
            der(r) = 0.0f;
            for (int c = 0; c < n; ++c) {
                hess(r, c) = 0.0f;
            }
        }

        float step[n] = {1e-3f, 1e-3f, 1e-3f, 1e-4f, 1e-4f, 1e-4f};
        float f_plus[n];
        float f_minus[n];

        for (int i = 0; i < n; ++i) {
            column_vector xp = x;
            column_vector xm = x;
            xp(i) += step[i];
            xm(i) -= step[i];
            f_plus[i] = (*this)(xp);
            f_minus[i] = (*this)(xm);
            der(i) = (f_plus[i] - f_minus[i]) / (2.0f * step[i]);
            hess(i, i) = (f_plus[i] - 2.0f * f0 + f_minus[i]) /
                         (step[i] * step[i]);
        }

        updateInflation(der, hess);
    }

private:
    static float rawScore(const Eigen::Matrix3f& R,
                          const Eigen::Vector3f& t,
                          const gmm_utils::GMM3f& target_gmm,
                          const gmm_utils::GMM3f& source_gmm,
                          float normalizer,
                          float inflation_factor) {
        const std::uint32_t Nm = target_gmm.getNClusters();
        const std::uint32_t Nk = source_gmm.getNClusters();
        const auto& wm = target_gmm.getWeights();
        const auto& wk = source_gmm.getWeights();
        const auto& Lm = target_gmm.getCovs();
        const auto& Omk = source_gmm.getCovs();
        const auto& mu_m = target_gmm.getMeans();
        const auto& nu_k = source_gmm.getMeans();

        if (Nm == 0 || Nk == 0 || normalizer == 0.0f) {
            return 0.0f;
        }

        const Eigen::Matrix3f Minf =
            Eigen::Matrix3f::Identity() * inflation_factor;
        float score = 0.0f;

        for (std::uint32_t m = 0; m < Nm; ++m) {
            const Eigen::Matrix3f L = covarianceFromColumn(Lm, m);
            const Eigen::Vector3f mu = mu_m.col(m);
            for (std::uint32_t k = 0; k < Nk; ++k) {
                const Eigen::Matrix3f Om = covarianceFromColumn(Omk, k);
                const Eigen::Vector3f nu = nu_k.col(k);
                const float w = wm(m) * wk(k);
                if (!std::isfinite(w) || w <= 0.0f) {
                    continue;
                }

                const Eigen::Vector3f Rnu = R * nu;
                const Eigen::Vector3f y = mu - Rnu - t;

                const float radius2 = std::max(
                    L.trace() + Om.trace(),
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

    void updateInflation(const column_vector& grad,
                         const general_matrix& hess) const {
        constexpr int n = 6;
        dlib::matrix<float> damped = hess;
        for (int i = 0; i < n; ++i) {
            damped(i, i) += 1e-3f;
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
            std::clamp(alpha * kInitialInflation, 0.0f, inflation_factor_);
    }

    const gmm_utils::GMM3f& source_gmm_;
    const gmm_utils::GMM3f& target_gmm_;
    float Et_ = 1.0f;
    mutable float inflation_factor_;
};

float runD2DRegistration(
    const gmm_utils::GMM3f& source_gmm,
    const gmm_utils::GMM3f& target_gmm,
    const Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>& Tinit,
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>& Tout) {
    ColumnVector x(6);
    const Eigen::Vector3f t = Tinit.translation();
    const Eigen::AngleAxisf aa(Tinit.rotation());
    Eigen::Vector3f u = Eigen::Vector3f::Zero();
    if (std::isfinite(aa.angle()) && aa.axis().allFinite()) {
        u = aa.axis() * aa.angle();
    }
    x = t(0), t(1), t(2), u(0), u(1), u(2);

    const float score = dlib::find_max_trust_region(
        dlib::objective_delta_stop_strategy(1e-7),
        GmmRegistrationModel(source_gmm, target_gmm),
        x,
        5.0);

    Tout = transformFromState(x);
    return score;
}

} // namespace

RegistrationResult isoplanarRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path) {

    RegistrationResult result;

    auto Tinit = toAffine(T_init);
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;

    gmm_utils::GMM3f source_gmm, target_gmm;
    source_gmm.load(source_path);
    source_gmm.makeCovsIsoplanar();
    target_gmm.load(target_path);
    target_gmm.makeCovsIsoplanar();

    float score = runD2DRegistration(source_gmm, target_gmm, Tinit, Tout);

    result.transform = Tout.matrix();
    result.score = score;
    result.success = std::isfinite(score) && !result.transform.hasNaN();
    return result;
}

RegistrationResult anisotropicRegistration(
    const Eigen::Matrix4f& T_init,
    const std::string& source_path,
    const std::string& target_path) {

    RegistrationResult result;

    auto Tinit = toAffine(T_init);
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;

    gmm_utils::GMM3f source_gmm, target_gmm;
    source_gmm.load(source_path);
    target_gmm.load(target_path);

    float score = runD2DRegistration(source_gmm, target_gmm, Tinit, Tout);

    result.transform = Tout.matrix();
    result.score = score;
    result.success = std::isfinite(score) && !result.transform.hasNaN();
    return result;
}

} // namespace gmmslam
