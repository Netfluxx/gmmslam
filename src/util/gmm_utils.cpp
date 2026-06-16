#include "gmmslam/util/gmm_utils.hpp"
#include "util/rtree.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <gmm/GMM3.h>
#include "gmmslam/ros2_compat.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace gmmslam {

namespace {

const std::vector<std::array<double, 3>> SUBMAP_COLORS = {
    {0.12, 0.47, 0.71}, {1.00, 0.50, 0.05}, {0.17, 0.63, 0.17},
    {0.84, 0.15, 0.16}, {0.58, 0.40, 0.74}, {0.55, 0.34, 0.29},
    {0.89, 0.47, 0.76}, {0.74, 0.74, 0.13}, {0.09, 0.75, 0.81},
    {0.98, 0.60, 0.60}};

std::mutex gmm_file_io_mutex;

// Symmetrize and add Tikhonov regularization so the covariance is safe to
// Cholesky-factor even when the source GMM produced a near-degenerate one.
Matrix3d symRegularize(const Matrix3d& C, double reg) {
    Matrix3d S = 0.5 * (C + C.transpose());
    S.diagonal().array() += reg;
    return S;
}

GmmComponent keepLowerUncertainty(const GmmComponent& a,
                                  const GmmComponent& b) {
    const double total_weight = a.weight + b.weight;
    const bool keep_a =
        (a.pose_uncertainty < b.pose_uncertainty) ||
        (a.pose_uncertainty == b.pose_uncertainty &&
         a.source_key_idx <= b.source_key_idx);

    GmmComponent kept = keep_a ? a : b;
    if (total_weight > 0.0) {
        kept.weight = total_weight;
    }
    return kept;
}

GmmComponent keepOlderMeasurement(const GmmComponent& a,
                                  const GmmComponent& b) {
    const double total_weight = a.weight + b.weight;
    const bool a_known = a.source_key_idx >= 0;
    const bool b_known = b.source_key_idx >= 0;

    bool keep_a = true;
    if (a_known && b_known && a.source_key_idx != b.source_key_idx) {
        keep_a = a.source_key_idx < b.source_key_idx;
    } else {
        keep_a = (a.pose_uncertainty < b.pose_uncertainty) ||
                 (a.pose_uncertainty == b.pose_uncertainty &&
                  a.source_key_idx <= b.source_key_idx);
    }

    GmmComponent kept = keep_a ? a : b;
    if (total_weight > 0.0) {
        kept.weight = total_weight;
    }
    return kept;
}

bool sameKnownSourceFrame(const GmmComponent& a, const GmmComponent& b) {
    return a.source_key_idx >= 0 &&
           b.source_key_idx >= 0 &&
           a.source_key_idx == b.source_key_idx;
}

void logFramePruneMerge(const GmmComponent& a,
                        const GmmComponent& b,
                        const GmmComponent& kept,
                        double d_b,
                        int pass,
                        std::size_t i,
                        std::size_t j) {
    (void)a;
    (void)b;
    (void)kept;
    (void)d_b;
    (void)pass;
    (void)i;
    (void)j;
    // Per-pair frame-to-frame prune logs are too noisy for normal runs.
    // ROS_INFO("[gmm_utils] frame-to-frame prune pass=%d pair=(%zu,%zu) "
    //          "D_B=%.4f dist=%.3fm keep_frame=X(%d) drop_newer_frame=X(%d) "
    //          "weights=(%.4f, %.4f)->%.4f",
    //          pass + 1, i, j, d_b, delta.norm(),
    //          kept.source_key_idx, dropped.source_key_idx,
    //          a.weight, b.weight, kept.weight);
}

// Axis-aligned box that contains the Mahalanobis ellipsoid
// { x : (x-μ)ᵀ Σ⁻¹ (x-μ) ≤ χ² } by bounding it in the principal frame.
void componentToWorldAabb(
        const GmmComponent& c,
        double chi_sq,
        double min_out[3],
        double max_out[3]) {
    Matrix3d cov = 0.5 * (c.covariance + c.covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
    if (es.info() != Eigen::Success) {
        const double r = 0.5;
        for (int d = 0; d < 3; ++d) {
            min_out[d] = c.mean(d) - r;
            max_out[d] = c.mean(d) + r;
        }
        return;
    }
    Vector3d lam = es.eigenvalues().cwiseMax(1e-9);
    Matrix3d R = es.eigenvectors();
    const double chi = std::max(chi_sq, 1e-6);
    Vector3d half = (lam * chi).cwiseSqrt();

    Vector3d mn = c.mean;
    Vector3d mx = c.mean;
    for (int mask = 0; mask < 8; ++mask) {
        Vector3d local(
            (mask & 1) ? half(0) : -half(0),
            (mask & 2) ? half(1) : -half(1),
            (mask & 4) ? half(2) : -half(2));
        const Vector3d p = c.mean + R * local;
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    for (int d = 0; d < 3; ++d) {
        min_out[d] = mn(d);
        max_out[d] = mx(d);
        if (!(min_out[d] < max_out[d])) {
            min_out[d] -= 1e-6;
            max_out[d] += 1e-6;
        }
    }
}

void inflateAabb(double min_b[3], double max_b[3], double margin) {
    if (margin <= 0.0) {
        return;
    }
    for (int d = 0; d < 3; ++d) {
        min_b[d] -= margin;
        max_b[d] += margin;
    }
}

} // namespace

double bhattacharyyaDistance(const Vector3d& mu_a, const Matrix3d& cov_a,
                             const Vector3d& mu_b, const Matrix3d& cov_b,
                             double cov_reg) {
    const Matrix3d Ca = symRegularize(cov_a, cov_reg);
    const Matrix3d Cb = symRegularize(cov_b, cov_reg);
    const Matrix3d C  = 0.5 * (Ca + Cb);

    Eigen::LLT<Matrix3d> llt(C); // CHolesky
    Eigen::LLT<Matrix3d> llt_a(Ca);
    Eigen::LLT<Matrix3d> llt_b(Cb);
    if (llt.info()  != Eigen::Success ||
        llt_a.info() != Eigen::Success ||
        llt_b.info() != Eigen::Success) {
        return std::numeric_limits<double>::infinity();
    }

    const Vector3d d = mu_a - mu_b;
    const double mahal_sq = d.dot(llt.solve(d));

    auto logDetFromLLT = [](const Eigen::LLT<Matrix3d>& f) {
        const Matrix3d& L = f.matrixLLT();
        return 2.0 * (std::log(L(0, 0)) + std::log(L(1, 1)) + std::log(L(2, 2)));
    };
    const double log_det_C  = logDetFromLLT(llt);
    const double log_det_Ca = logDetFromLLT(llt_a);
    const double log_det_Cb = logDetFromLLT(llt_b);

    return 0.125 * mahal_sq + 0.5 * (log_det_C - 0.5 * (log_det_Ca + log_det_Cb));
}

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
        good.source_key_idx = comp.source_key_idx;
        good.pose_uncertainty = comp.pose_uncertainty;
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

Matrix3d covarianceForRegistrationFile(const Matrix3d& cov_in) {
    Matrix3d cov = 0.5 * (cov_in + cov_in.transpose());
    if (!cov.allFinite()) {
        return Matrix3d::Identity() * 1e-3;
    }

    Eigen::SelfAdjointEigenSolver<Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
        return Matrix3d::Identity() * 1e-3;
    }

    // D2D registration in the external GIRA3D backend explicitly inverts these
    // matrices and becomes unstable with very thin / high-condition covariances
    // (inverse entries around 1e4 show up as "Matrix not invertible" NaNs).
    // Save a gentler registration copy than the visualization/map covariance.
    Vector3d lam = solver.eigenvalues();
    lam = lam.cwiseMax(1e-2).cwiseMin(9.0);
    const double min_lam = lam.minCoeff();
    const double max_allowed = std::max(min_lam * 50.0, min_lam);
    lam = lam.cwiseMin(max_allowed);
    Matrix3d safe = solver.eigenvectors() * lam.asDiagonal() *
                    solver.eigenvectors().transpose();
    safe = 0.5 * (safe + safe.transpose());
    safe.diagonal().array() += 1e-6;
    return safe;
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
    std::vector<PosedGmmInput> inputs;
    inputs.reserve(gmms_with_poses.size());
    for (const auto& [gmm, pose] : gmms_with_poses) {
        PosedGmmInput input;
        input.model = gmm;
        input.pose = pose;
        inputs.push_back(std::move(input));
    }
    return mergeGmmsConcatenate(inputs, T_ref);
}

GmmModel mergeGmmsConcatenate(
    const std::vector<PosedGmmInput>& gmms_with_poses,
    const Matrix4d& T_ref) {

    GmmModel merged;
    const Matrix4d T_ref_inv = T_ref.inverse();

    for (const auto& input : gmms_with_poses) {
        const Matrix4d T_kf_in_ref = T_ref_inv * input.pose;
        const Matrix3d R = T_kf_in_ref.block<3, 3>(0, 0);
        const Vector3d t = T_kf_in_ref.block<3, 1>(0, 3);

        for (const auto& comp : input.model.components) {
            GmmComponent c;
            c.mean = R * comp.mean + t;
            Matrix3d cov_ref = R * comp.covariance * R.transpose();
            c.covariance = 0.5 * (cov_ref + cov_ref.transpose());
            c.weight = comp.weight;
            c.source_key_idx = input.key_idx;
            c.pose_uncertainty = input.pose_uncertainty;
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

GmmModel pruneSimilarComponentsImpl(const GmmModel& in, const MapConfig& map_cfg,
                                    bool only_between_source_frames,
                                    bool prefer_older_measurement,
                                    const char* log_label) {
    if (!map_cfg.prune_enable || in.components.size() < 2) {
        return in;
    }

    const int components_before = static_cast<int>(in.components.size());
    std::vector<GmmComponent> comps = in.components;
    const double radius_sq = map_cfg.prune_search_radius_m *
                              map_cfg.prune_search_radius_m;
    const int max_passes = std::max(1, map_cfg.prune_max_passes);
    const double margin_m = std::max(0.0, map_cfg.prune_search_radius_m);
    const double chi_sq = std::max(1e-3, map_cfg.prune_rtree_chi_sq);
    std::size_t total_merges = 0;
    int passes_run = 0;

    for (int pass = 0; pass < max_passes; ++pass) {
        const std::size_t n = comps.size();
        std::vector<bool> absorbed(n, false);
        std::vector<GmmComponent> kept;
        kept.reserve(n);
        std::size_t merges_this_pass = 0;

        using ComponentRTree = RTree<int, double, 3>;
        ComponentRTree tree;
        if (map_cfg.prune_use_rtree) {
            for (std::size_t k = 0; k < n; ++k) {
                double mn[3], mx[3];
                componentToWorldAabb(comps[k], chi_sq, mn, mx);
                inflateAabb(mn, mx, margin_m);
                tree.Insert(mn, mx, static_cast<int>(k));
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (absorbed[i]) {
                continue;
            }
            GmmComponent base = comps[i];

            if (map_cfg.prune_use_rtree) {
                // Re-query the tree after each merge: `base` moves/grows so its
                // AABB changes; new overlaps can appear that were not in the
                // first search window.
                for (;;) {
                    double bmin[3], bmax[3];
                    componentToWorldAabb(base, chi_sq, bmin, bmax);
                    inflateAabb(bmin, bmax, margin_m);

                    std::vector<int> candidates;
                    candidates.reserve(32);
                    tree.Search(bmin, bmax, [&](int j) {
                        const std::size_t sj = static_cast<std::size_t>(j);
                        if (sj <= i || absorbed[sj]) {
                            return true;
                        }
                        candidates.push_back(j);
                        return true;
                    });
                    std::sort(candidates.begin(), candidates.end());

                    bool merged_one = false;
                    for (int j : candidates) {
                        const std::size_t sj = static_cast<std::size_t>(j);
                        if (absorbed[sj]) {
                            continue;
                        }
                        if (only_between_source_frames &&
                            sameKnownSourceFrame(base, comps[sj])) {
                            continue;
                        }
                        const double d_b = bhattacharyyaDistance(
                            base.mean, base.covariance,
                            comps[sj].mean, comps[sj].covariance,
                            map_cfg.prune_cov_reg);
                        if (!std::isfinite(d_b)) {
                            continue;
                        }
                        if (d_b < map_cfg.prune_bhatt_threshold) {
                            GmmComponent kept = prefer_older_measurement
                                                ? keepOlderMeasurement(base, comps[sj])
                                                : keepLowerUncertainty(base, comps[sj]);
                            if (prefer_older_measurement) {
                                logFramePruneMerge(base, comps[sj], kept,
                                                   d_b, pass, i, sj);
                            }
                            base = std::move(kept);
                            absorbed[sj] = true;
                            ++merges_this_pass;
                            merged_one = true;
                            break;
                        }
                    }
                    if (!merged_one) {
                        break;
                    }
                }
            } else {
                for (std::size_t j = i + 1; j < n; ++j) {
                    if (absorbed[j]) {
                        continue;
                    }
                    if (only_between_source_frames &&
                        sameKnownSourceFrame(base, comps[j])) {
                        continue;
                    }
                    const Vector3d delta = base.mean - comps[j].mean;
                    if (delta.squaredNorm() > radius_sq) {
                        continue;
                    }

                    const double d_b = bhattacharyyaDistance(
                        base.mean, base.covariance,
                        comps[j].mean, comps[j].covariance,
                        map_cfg.prune_cov_reg);
                    if (!std::isfinite(d_b)) {
                        continue;
                    }
                    if (d_b < map_cfg.prune_bhatt_threshold) {
                        GmmComponent kept = prefer_older_measurement
                                            ? keepOlderMeasurement(base, comps[j])
                                            : keepLowerUncertainty(base, comps[j]);
                        if (prefer_older_measurement) {
                            logFramePruneMerge(base, comps[j], kept,
                                               d_b, pass, i, j);
                        }
                        base = std::move(kept);
                        absorbed[j] = true;
                        ++merges_this_pass;
                    }
                }
            }

            kept.push_back(std::move(base));
        }

        comps.swap(kept);
        passes_run = pass + 1;
        total_merges += merges_this_pass;
        if (merges_this_pass > 0) {
            ROS_INFO("[gmm_utils] prune pass %d/%d: merged %zu pair(s), "
                     "%zu -> %zu component(s)",
                     pass + 1, max_passes, merges_this_pass, n, comps.size());
        }
        if (merges_this_pass == 0) {
            break;
        }
    }

    double weight_sum = 0.0;
    for (const auto& c : comps) {
        weight_sum += c.weight;
    }
    if (weight_sum > 0.0) {
        for (auto& c : comps) {
            c.weight /= weight_sum;
        }
    }

    GmmModel out;
    out.components = std::move(comps);
    const int components_after = out.numComponents();
    const int removed = components_before - components_after;
    ROS_INFO("[gmm_utils] %s prune summary: %d -> %d component(s) "
             "(%d removed), %zu merge(s) in %d pass(es), "
             "D_B < %.4f, rtree=%s, chi_sq=%.2f, margin_m=%.3f, "
             "between_frames=%s, prefer_older=%s",
             log_label, components_before, components_after, removed,
             total_merges, passes_run,
             map_cfg.prune_bhatt_threshold,
             map_cfg.prune_use_rtree ? "on" : "off",
             chi_sq, margin_m,
             only_between_source_frames ? "true" : "false",
             prefer_older_measurement ? "true" : "false");
    return out;
}

GmmModel pruneSimilarComponents(const GmmModel& in, const MapConfig& map_cfg) {
    return pruneSimilarComponentsImpl(
        in, map_cfg,
        false, false,
        "GMM");
}

GmmModel pruneNewerFrameComponents(const GmmModel& in,
                                   const MapConfig& map_cfg) {
    if (!map_cfg.prune_frame_to_frame_enable) {
        ROS_INFO("[gmm_utils] frame-to-frame GMM prune disabled; keeping %d component(s)",
                 in.numComponents());
        return in;
    }
    return pruneSimilarComponentsImpl(
        in, map_cfg,
        true, true,
        "frame-to-frame GMM");
}

GmmModel mergeGmmsAndPrune(
    const std::vector<std::pair<GmmModel, Matrix4d>>& gmms_with_poses,
    const Matrix4d& T_ref,
    const MapConfig& map_cfg) {
    std::vector<PosedGmmInput> inputs;
    inputs.reserve(gmms_with_poses.size());
    for (const auto& [gmm, pose] : gmms_with_poses) {
        PosedGmmInput input;
        input.model = gmm;
        input.pose = pose;
        inputs.push_back(std::move(input));
    }
    return mergeGmmsAndPrune(inputs, T_ref, map_cfg);
}

GmmModel mergeGmmsAndPrune(
    const std::vector<PosedGmmInput>& gmms_with_poses,
    const Matrix4d& T_ref,
    const MapConfig& map_cfg) {
    GmmModel concatenated = mergeGmmsConcatenate(gmms_with_poses, T_ref);
    if (!map_cfg.prune_enable) {
        return concatenated;
    }
    return pruneNewerFrameComponents(concatenated, map_cfg);
}

void saveGmmToFile(const GmmModel& model, const std::string& filepath) {
    std::vector<const GmmComponent*> components;
    components.reserve(model.components.size());
    for (const auto& comp : model.components) {
        if (!comp.mean.allFinite() || !comp.covariance.allFinite() ||
            !std::isfinite(comp.weight) || comp.weight <= 0.0) {
            continue;
        }
        // Far-fill depth support is useful for RViz/map coverage, but extreme
        // oblique components destabilize the GIRA D2D backend covariance inversions.
        if (comp.mean.norm() > 60.0) {
            continue;
        }
        components.push_back(&comp);
    }

    const int K = static_cast<int>(components.size());
    if (K == 0) return;

    std::lock_guard<std::mutex> io_lk(gmm_file_io_mutex);

    gmm_utils::GMM3f gmm;

    Eigen::VectorXf weights(K);
    Eigen::Matrix3Xf means(3, K);
    Eigen::Matrix<float, 9, Eigen::Dynamic> covs(9, K);

    for (int k = 0; k < K; ++k) {
        const auto& comp = *components[static_cast<size_t>(k)];
        weights(k) = static_cast<float>(comp.weight);
        means.col(k) = comp.mean.cast<float>();

        const Matrix3d cov_safe = covarianceForRegistrationFile(comp.covariance);
        // Flatten 3x3 covariance column-major into a 9-element vector
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> cov_flat(cov_safe.data());
        covs.col(k) = cov_flat.cast<float>();
    }

    gmm.setWeights(weights);
    gmm.setMeans(means);
    gmm.setCovs(covs);

    // GMM3f::save is used from SOGMM worker threads and global-submap pruning.
    // Serialize it and replace atomically so D2D never reads a half-written file.
    const std::string tmp_path = filepath + ".tmp.gmm";
    try {
        gmm.save(tmp_path);
        if (std::rename(tmp_path.c_str(), filepath.c_str()) != 0) {
            const int err = errno;
            std::remove(tmp_path.c_str());
            throw std::runtime_error(
                "rename(" + tmp_path + " -> " + filepath + ") failed: " +
                std::strerror(err));
        }
    } catch (...) {
        std::remove(tmp_path.c_str());
        throw;
    }
}

const std::vector<std::array<double, 3>>& submapColors() {
    return SUBMAP_COLORS;
}

std::array<double, 3> submapColor(int sid) {
    return SUBMAP_COLORS[static_cast<size_t>(sid) % SUBMAP_COLORS.size()];
}

} // namespace gmmslam
