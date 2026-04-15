#include "gmmslam/d2d_registration.hpp"
#include <Eigen/Geometry>
#include <gmm/GMM3.h>
#include <gmm_d2d_registration/GMMD2DRegistration.h>
#include <cmath>

namespace gmmslam {

namespace {

Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor>
toAffine(const Eigen::Matrix4f& T) {
    Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> out;
    out = Eigen::Translation3f(T.block<3, 1>(0, 3)) *
          Eigen::Quaternionf(T.block<3, 3>(0, 0));
    return out;
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

    MatcherD2D matcher;
    float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

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

    MatcherD2D matcher;
    float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

    result.transform = Tout.matrix();
    result.score = score;
    result.success = std::isfinite(score) && !result.transform.hasNaN();
    return result;
}

} // namespace gmmslam
