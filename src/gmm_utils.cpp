#include "gmmslam/gmm_utils.hpp"
#include <Eigen/Eigenvalues>
#include <gmm/GMM3.h>
#include <cmath>
#include <limits>

namespace gmmslam {

namespace {

const std::vector<std::array<double, 3>> SUBMAP_COLORS = {
    {0.12, 0.47, 0.71}, {1.00, 0.50, 0.05}, {0.17, 0.63, 0.17},
    {0.84, 0.15, 0.16}, {0.58, 0.40, 0.74}, {0.55, 0.34, 0.29},
    {0.89, 0.47, 0.76}, {0.74, 0.74, 0.13}, {0.09, 0.75, 0.81},
    {0.98, 0.60, 0.60}};

} // namespace

GmmModel filterWellConditioned(const GmmModel& model, double reg) {
    GmmModel filtered;
    double weight_sum = 0.0;

    for (const auto& comp : model.components) {
        Matrix3d cov = 0.5 * (comp.covariance + comp.covariance.transpose());
        cov += reg * Matrix3d::Identity();

        Eigen::SelfAdjointEigenSolver<Matrix3d> solver(cov);
        const Vector3d& eigenvalues = solver.eigenvalues();

        if (!eigenvalues.allFinite()) continue;
        if ((eigenvalues.array() <= 0.0).any()) continue;

        GmmComponent good;
        good.mean = comp.mean;
        good.covariance = cov;
        good.weight = comp.weight;
        weight_sum += good.weight;
        filtered.components.push_back(std::move(good));
    }

    if (weight_sum > 0.0) {
        for (auto& c : filtered.components) {
            c.weight /= weight_sum;
        }
    }

    return filtered;
}

std::vector<GmmLocalData> precomputeGmmLocalData(const GmmModel& model) {
    std::vector<GmmLocalData> out;
    out.reserve(model.components.size());

    for (const auto& comp : model.components) {
        Matrix3d cov = 0.5 * (comp.covariance + comp.covariance.transpose());

        Eigen::SelfAdjointEigenSolver<Matrix3d> solver(cov);
        Vector3d evals = solver.eigenvalues();
        Matrix3d evecs = solver.eigenvectors();

        evals = evals.cwiseMax(1e-9);

        if (evecs.determinant() < 0.0) {
            evecs.col(0) *= -1.0;
        }

        GmmLocalData ld;
        ld.scales = evals.cwiseSqrt();
        ld.rotation = evecs;
        ld.mean_local = comp.mean;
        out.push_back(std::move(ld));
    }

    return out;
}

GmmModel mergeGmmsConcatenate(
    const std::vector<std::pair<GmmModel, Matrix4d>>& gmms_with_poses,
    const Matrix4d& T_ref) {

    GmmModel merged;
    const Matrix4d T_ref_inv = T_ref.inverse();

    for (const auto& [gmm, T_world] : gmms_with_poses) {
        const Matrix4d T_kf_in_ref = T_ref_inv * T_world;
        const Matrix3d R = T_kf_in_ref.block<3, 3>(0, 0);
        const Vector3d t = T_kf_in_ref.block<3, 1>(0, 3);

        for (const auto& comp : gmm.components) {
            GmmComponent c;
            c.mean = R * comp.mean + t;
            Matrix3d cov_ref = R * comp.covariance * R.transpose();
            c.covariance = 0.5 * (cov_ref + cov_ref.transpose());
            c.weight = comp.weight;
            merged.components.push_back(std::move(c));
        }
    }

    double weight_sum = 0.0;
    for (const auto& c : merged.components) {
        weight_sum += c.weight;
    }
    if (weight_sum > 0.0) {
        for (auto& c : merged.components) {
            c.weight /= weight_sum;
        }
    }

    return merged;
}

void saveGmmToFile(const GmmModel& model, const std::string& filepath) {
    const int K = model.numComponents();
    if (K == 0) return;

    gmm_utils::GMM3f gmm;

    Eigen::VectorXf weights(K);
    Eigen::Matrix3Xf means(3, K);
    Eigen::Matrix<float, 9, Eigen::Dynamic> covs(9, K);

    for (int k = 0; k < K; ++k) {
        const auto& comp = model.components[static_cast<size_t>(k)];
        weights(k) = static_cast<float>(comp.weight);
        means.col(k) = comp.mean.cast<float>();

        // Flatten 3x3 covariance column-major into a 9-element vector
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> cov_flat(comp.covariance.data());
        covs.col(k) = cov_flat.cast<float>();
    }

    gmm.setWeights(weights);
    gmm.setMeans(means);
    gmm.setCovs(covs);
    gmm.save(filepath);
}

const std::vector<std::array<double, 3>>& submapColors() {
    return SUBMAP_COLORS;
}

std::array<double, 3> submapColor(int sid) {
    return SUBMAP_COLORS[static_cast<size_t>(sid) % SUBMAP_COLORS.size()];
}

} // namespace gmmslam
