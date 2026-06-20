#include "gmmslam/util/gmm_utils.hpp"
#include "gmmslam/rclcpp_logging.hpp"
#include "util/rtree.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <gmm/GMM3.h>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
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
    // GMS_INFO("[gmm_utils] frame-to-frame prune pass=%d pair=(%zu,%zu) "
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

struct CandidateEdge {
    std::size_t a = 0;
    std::size_t b = 0;
    double d_b = 0.0;
};

class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    std::size_t find(std::size_t x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    void unite(std::size_t a, std::size_t b) {
        std::size_t ra = find(a);
        std::size_t rb = find(b);
        if (ra == rb) {
            return;
        }
        if (rank_[ra] < rank_[rb]) {
            std::swap(ra, rb);
        }
        parent_[rb] = ra;
        if (rank_[ra] == rank_[rb]) {
            ++rank_[ra];
        }
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<int> rank_;
};

bool betterRepresentative(const GmmComponent& a,
                          const GmmComponent& b,
                          bool prefer_older_measurement) {
    if (prefer_older_measurement) {
        const bool a_known = a.source_key_idx >= 0;
        const bool b_known = b.source_key_idx >= 0;
        if (a_known && b_known && a.source_key_idx != b.source_key_idx) {
            return a.source_key_idx < b.source_key_idx;
        }
    }
    if (a.pose_uncertainty != b.pose_uncertainty) {
        return a.pose_uncertainty < b.pose_uncertainty;
    }
    return a.source_key_idx <= b.source_key_idx;
}

std::size_t bestRepresentativeIndex(const std::vector<GmmComponent>& comps,
                                    const std::vector<std::size_t>& cluster,
                                    bool prefer_older_measurement) {
    std::size_t best = cluster.front();
    for (std::size_t idx : cluster) {
        if (betterRepresentative(comps[idx], comps[best],
                                 prefer_older_measurement)) {
            best = idx;
        }
    }
    return best;
}

bool hasClearlyBestUncertainty(const std::vector<GmmComponent>& comps,
                               const std::vector<std::size_t>& cluster,
                               std::size_t best) {
    const double best_u = comps[best].pose_uncertainty;
    if (!std::isfinite(best_u)) {
        return false;
    }
    constexpr double kUncertaintyTieEps = 1.0e-6;
    for (std::size_t idx : cluster) {
        if (idx == best) {
            continue;
        }
        const double u = comps[idx].pose_uncertainty;
        if (!std::isfinite(u)) {
            continue;
        }
        if (u <= best_u + kUncertaintyTieEps) {
            return false;
        }
    }
    return true;
}

GmmComponent representativeWithClusterWeight(
        const std::vector<GmmComponent>& comps,
        const std::vector<std::size_t>& cluster,
        std::size_t best) {
    GmmComponent out = comps[best];
    double weight_sum = 0.0;
    for (std::size_t idx : cluster) {
        weight_sum += comps[idx].weight;
    }
    if (weight_sum > 0.0) {
        out.weight = weight_sum;
    }
    return out;
}

GmmComponent momentMatchCluster(const std::vector<GmmComponent>& comps,
                                const std::vector<std::size_t>& cluster,
                                std::size_t best) {
    double weight_sum = 0.0;
    for (std::size_t idx : cluster) {
        weight_sum += std::max(0.0, comps[idx].weight);
    }
    if (weight_sum <= 0.0) {
        return representativeWithClusterWeight(comps, cluster, best);
    }

    Vector3d mean = Vector3d::Zero();
    for (std::size_t idx : cluster) {
        mean += std::max(0.0, comps[idx].weight) * comps[idx].mean;
    }
    mean /= weight_sum;

    Matrix3d covariance = Matrix3d::Zero();
    for (std::size_t idx : cluster) {
        const double w = std::max(0.0, comps[idx].weight);
        const Vector3d d = comps[idx].mean - mean;
        covariance += w * (comps[idx].covariance + d * d.transpose());
    }
    covariance /= weight_sum;

    GmmComponent out = comps[best];
    out.mean = mean;
    out.covariance = 0.5 * (covariance + covariance.transpose());
    out.weight = weight_sum;
    return out;
}

bool aabbOverlap(const double amin[3], const double amax[3],
                 const double bmin[3], const double bmax[3]) {
    for (int d = 0; d < 3; ++d) {
        if (amax[d] < bmin[d] || bmax[d] < amin[d]) {
            return false;
        }
    }
    return true;
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

PruneResult pruneSimilarComponentsImpl(const GmmModel& in,
                                       const MapConfig& map_cfg,
                                       bool only_between_source_frames,
                                       bool prefer_older_measurement,
                                       const char* log_label) {
    PruneResult result;
    if (!map_cfg.prune_enable || in.components.size() < 2) {
        result.model = in;
        return result;
    }

    const int components_before = static_cast<int>(in.components.size());
    const std::vector<GmmComponent>& comps = in.components;
    const double radius_sq = map_cfg.prune_search_radius_m *
                              map_cfg.prune_search_radius_m;
    const double margin_m = std::max(0.0, map_cfg.prune_search_radius_m);
    const double chi_sq = std::max(1e-3, map_cfg.prune_rtree_chi_sq);
    const std::size_t n = comps.size();

    std::vector<CandidateEdge> edges;
    edges.reserve(n);
    if (map_cfg.prune_use_rtree) {
        using ComponentRTree = RTree<int, double, 3>;
        ComponentRTree tree;
        std::vector<std::array<double, 3>> mins(n), maxs(n);
        for (std::size_t k = 0; k < n; ++k) {
            componentToWorldAabb(comps[k], chi_sq, mins[k].data(), maxs[k].data());
            inflateAabb(mins[k].data(), maxs[k].data(), margin_m);
            tree.Insert(mins[k].data(), maxs[k].data(), static_cast<int>(k));
        }

        for (std::size_t i = 0; i < n; ++i) {
            std::vector<int> candidates;
            candidates.reserve(32);
            tree.Search(mins[i].data(), maxs[i].data(), [&](int j) {
                const std::size_t sj = static_cast<std::size_t>(j);
                if (sj > i) {
                    candidates.push_back(j);
                }
                return true;
            });
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()),
                             candidates.end());

            for (int j_raw : candidates) {
                const std::size_t j = static_cast<std::size_t>(j_raw);
                if (!aabbOverlap(mins[i].data(), maxs[i].data(),
                                 mins[j].data(), maxs[j].data())) {
                    continue;
                }
                if (only_between_source_frames &&
                    sameKnownSourceFrame(comps[i], comps[j])) {
                    continue;
                }
                const double d_b = bhattacharyyaDistance(
                    comps[i].mean, comps[i].covariance,
                    comps[j].mean, comps[j].covariance,
                    map_cfg.prune_cov_reg);
                if (std::isfinite(d_b) && d_b < map_cfg.prune_bhatt_threshold) {
                    edges.push_back({i, j, d_b});
                }
            }
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                if (only_between_source_frames &&
                    sameKnownSourceFrame(comps[i], comps[j])) {
                    continue;
                }
                const Vector3d delta = comps[i].mean - comps[j].mean;
                if (delta.squaredNorm() > radius_sq) {
                    continue;
                }
                const double d_b = bhattacharyyaDistance(
                    comps[i].mean, comps[i].covariance,
                    comps[j].mean, comps[j].covariance,
                    map_cfg.prune_cov_reg);
                if (std::isfinite(d_b) && d_b < map_cfg.prune_bhatt_threshold) {
                    edges.push_back({i, j, d_b});
                }
            }
        }
    }

    UnionFind uf(n);
    for (const auto& edge : edges) {
        uf.unite(edge.a, edge.b);
    }

    std::map<std::size_t, std::vector<std::size_t>> clusters_by_root;
    for (std::size_t i = 0; i < n; ++i) {
        clusters_by_root[uf.find(i)].push_back(i);
    }

    std::vector<GmmComponent> out_components;
    out_components.reserve(n);
    int cluster_id = 0;
    int clusters_merged = 0;
    for (const auto& [root, cluster] : clusters_by_root) {
        (void)root;
        if (cluster.size() == 1) {
            out_components.push_back(comps[cluster.front()]);
            continue;
        }

        const std::size_t best =
            bestRepresentativeIndex(comps, cluster, prefer_older_measurement);
        const bool keep_rep = prefer_older_measurement ||
                              hasClearlyBestUncertainty(comps, cluster, best);
        GmmComponent kept = keep_rep
            ? representativeWithClusterWeight(comps, cluster, best)
            : momentMatchCluster(comps, cluster, best);
        const int kept_output_index = static_cast<int>(out_components.size());

        for (std::size_t idx : cluster) {
            if (idx == best && keep_rep) {
                continue;
            }
            PruneDebugRecord rec;
            rec.kept_mean = kept.mean;
            rec.merged_mean = comps[idx].mean;
            rec.distance = bhattacharyyaDistance(
                kept.mean, kept.covariance,
                comps[idx].mean, comps[idx].covariance,
                map_cfg.prune_cov_reg);
            rec.cluster_id = cluster_id;
            rec.kept_source_key_idx = kept.source_key_idx;
            rec.merged_source_key_idx = comps[idx].source_key_idx;
            rec.kept_component_index = kept_output_index;
            rec.merged_component_index = static_cast<int>(idx);
            rec.reason = keep_rep ? "keep_representative" : "moment_match";
            result.debug_records.push_back(std::move(rec));
        }

        if (prefer_older_measurement && cluster.size() > 1) {
            for (std::size_t idx : cluster) {
                if (idx != best) {
                    logFramePruneMerge(comps[best], comps[idx], kept,
                                       result.debug_records.empty()
                                           ? 0.0
                                           : result.debug_records.back().distance,
                                       0, best, idx);
                }
            }
        }

        out_components.push_back(std::move(kept));
        ++cluster_id;
        ++clusters_merged;
    }

    double weight_sum = 0.0;
    for (const auto& c : out_components) {
        weight_sum += c.weight;
    }
    if (weight_sum > 0.0) {
        for (auto& c : out_components) {
            c.weight /= weight_sum;
        }
    }

    result.model.components = std::move(out_components);
    result.clusters_merged = clusters_merged;
    result.components_removed = components_before - result.model.numComponents();
    const int components_after = result.model.numComponents();
    const int removed = components_before - components_after;
    GMS_INFO("[gmm_utils] %s prune summary: %d -> %d component(s) "
             "(%d removed), %d cluster(s), %zu edge(s), "
             "D_B < %.4f, rtree=%s, chi_sq=%.2f, margin_m=%.3f, "
             "between_frames=%s, prefer_older=%s",
             log_label, components_before, components_after, removed,
             clusters_merged, edges.size(),
             map_cfg.prune_bhatt_threshold,
             map_cfg.prune_use_rtree ? "on" : "off",
             chi_sq, margin_m,
             only_between_source_frames ? "true" : "false",
             prefer_older_measurement ? "true" : "false");
    return result;
}

GmmModel pruneSimilarComponents(const GmmModel& in, const MapConfig& map_cfg) {
    return pruneSimilarComponentsDetailed(in, map_cfg).model;
}

PruneResult pruneSimilarComponentsDetailed(const GmmModel& in,
                                           const MapConfig& map_cfg) {
    return pruneSimilarComponentsImpl(
        in, map_cfg,
        false, false,
        "GMM");
}

GmmModel pruneNewerFrameComponents(const GmmModel& in,
                                   const MapConfig& map_cfg) {
    return pruneNewerFrameComponentsDetailed(in, map_cfg).model;
}

PruneResult pruneNewerFrameComponentsDetailed(const GmmModel& in,
                                              const MapConfig& map_cfg) {
    if (!map_cfg.prune_frame_to_frame_enable) {
        GMS_INFO("[gmm_utils] frame-to-frame GMM prune disabled; keeping %d component(s)",
                 in.numComponents());
        PruneResult result;
        result.model = in;
        return result;
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
