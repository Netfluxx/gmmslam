#include "gmmslam/gmmap_fitting.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Eigenvalues>
#include <ros/ros.h>

#ifdef GMMSLAM_HAS_GMMAP
#include <gmm_map/cluster.h>
#include <gmm_map/map.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace gmmslam {

bool gmmapFittingAvailable() {
#ifdef GMMSLAM_HAS_GMMAP
    return true;
#else
    return false;
#endif
}

#ifdef GMMSLAM_HAS_GMMAP
namespace {

struct VoxelAccumulator {
    Vector3d sum = Vector3d::Zero();
    Matrix3d sum_outer = Matrix3d::Zero();
    double weight = 0.0;
};

struct FarFillStats {
    int added_components = 0;
    double max_range = 0.0;
};

Eigen::Matrix3d gmmapOpticalToSensorRotation() {
    // GMMap's "tum" pinhole convention is optical: x right, y down, z forward.
    // The Webots RangeFinder PointCloud2 produced by rangefinder_webots.py is:
    // X forward, Y left, Z up.  Convert optical clusters back to that frame so
    // D2D and RViz use the same convention as the original point cloud.
    Eigen::Matrix3d R;
    R << 0.0,  0.0, 1.0,
        -1.0,  0.0, 0.0,
         0.0, -1.0, 0.0;
    return R;
}

std::int64_t voxelKey(const Vector3d& p, double voxel_m) {
    const auto ix = static_cast<std::int64_t>(std::floor(p.x() / voxel_m));
    const auto iy = static_cast<std::int64_t>(std::floor(p.y() / voxel_m));
    const auto iz = static_cast<std::int64_t>(std::floor(p.z() / voxel_m));
    const std::int64_t mask = (1LL << 21) - 1LL;
    return ((ix & mask) << 42) | ((iy & mask) << 21) | (iz & mask);
}

FarFillStats appendFarDepthComponents(const OrganizedDepthImage& depth,
                                      const SogmmConfig& config,
                                      double native_max_range,
                                      GmmModel& model) {
    FarFillStats stats;
    if (!config.gmmap_far_fill_enable ||
        config.gmmap_far_fill_max_components <= 0 ||
        config.gmmap_far_fill_voxel_m <= 0.0) {
        return stats;
    }

    const double max_depth = static_cast<double>(depth.depth.maxCoeff());
    const double skip_depth =
        std::max(0.0, max_depth - config.gmmap_far_fill_skip_max_depth_margin_m);
    const double start_range =
        std::max(config.gmmap_far_fill_start_m, native_max_range + 0.5);
    const double voxel = config.gmmap_far_fill_voxel_m;

    std::unordered_map<std::int64_t, VoxelAccumulator> voxels;
    voxels.reserve(static_cast<std::size_t>(depth.valid_points / 64 + 1));

    for (int v = 0; v < depth.depth.rows(); ++v) {
        for (int u = 0; u < depth.depth.cols(); ++u) {
            const double d = static_cast<double>(depth.depth(v, u));
            if (!(d > 0.0) || !std::isfinite(d) || d >= skip_depth) {
                continue;
            }

            const double right = (static_cast<double>(u) - depth.cx) * d / depth.fx;
            const double down = (static_cast<double>(v) - depth.cy) * d / depth.fy;
            const Vector3d p_sensor(d, -right, -down);
            const double range = p_sensor.norm();
            if (range <= start_range) {
                continue;
            }

            auto& acc = voxels[voxelKey(p_sensor, voxel)];
            acc.sum += p_sensor;
            acc.sum_outer += p_sensor * p_sensor.transpose();
            acc.weight += 1.0;
        }
    }

    std::vector<VoxelAccumulator> sorted;
    sorted.reserve(voxels.size());
    for (const auto& kv : voxels) {
        if (kv.second.weight >= 3.0) {
            sorted.push_back(kv.second);
        }
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const VoxelAccumulator& a, const VoxelAccumulator& b) {
                  return a.weight > b.weight;
              });

    const int limit = std::min(
        static_cast<int>(sorted.size()),
        config.gmmap_far_fill_max_components);
    const double min_var = std::max(1e-4, voxel * voxel * 0.0025);
    const double max_var = std::max(min_var, voxel * voxel);
    for (int i = 0; i < limit; ++i) {
        const Vector3d mean = sorted[i].sum / sorted[i].weight;
        Matrix3d cov =
            sorted[i].sum_outer / sorted[i].weight - mean * mean.transpose();
        cov = 0.5 * (cov + cov.transpose());

        Eigen::SelfAdjointEigenSolver<Matrix3d> eig(cov);
        if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite()) {
            continue;
        }
        Vector3d evals = eig.eigenvalues();
        evals = evals.cwiseMax(min_var).cwiseMin(max_var);
        cov = eig.eigenvectors() * evals.asDiagonal() *
              eig.eigenvectors().transpose();
        cov = 0.5 * (cov + cov.transpose());

        GmmComponent comp;
        comp.mean = mean;
        comp.covariance = cov;
        comp.weight = sorted[i].weight;
        model.components.push_back(std::move(comp));
        stats.max_range = std::max(stats.max_range, model.components.back().mean.norm());
        ++stats.added_components;
    }

    return stats;
}

void normalizeWeights(GmmModel& model) {
    double sum = 0.0;
    for (const auto& comp : model.components) {
        if (std::isfinite(comp.weight) && comp.weight > 0.0) {
            sum += comp.weight;
        }
    }
    if (sum <= 0.0) {
        return;
    }
    for (auto& comp : model.components) {
        comp.weight = (std::isfinite(comp.weight) && comp.weight > 0.0)
            ? comp.weight / sum
            : 0.0;
    }
}

int capComponentsByWeight(GmmModel& model, int max_components) {
    if (max_components <= 0 ||
        static_cast<int>(model.components.size()) <= max_components) {
        return 0;
    }

    auto keep_end = model.components.begin() + max_components;
    std::nth_element(
        model.components.begin(), keep_end, model.components.end(),
        [](const GmmComponent& a, const GmmComponent& b) {
            return a.weight > b.weight;
        });
    model.components.erase(keep_end, model.components.end());
    std::sort(
        model.components.begin(), model.components.end(),
        [](const GmmComponent& a, const GmmComponent& b) {
            return a.weight > b.weight;
        });
    return max_components;
}

class ScopedCoutSilencer {
public:
    explicit ScopedCoutSilencer(bool enabled)
        : enabled_(enabled), old_buf_(enabled ? std::cout.rdbuf(null_.rdbuf()) : nullptr) {}

    ~ScopedCoutSilencer() {
        if (enabled_) {
            std::cout.rdbuf(old_buf_);
        }
    }

private:
    bool enabled_;
    std::streambuf* old_buf_;
    std::ostringstream null_;
};

gmm::map_param makeMapParams(const OrganizedDepthImage& depth,
                             const SogmmConfig& config) {
    gmm::map_param param;
    param.dataset = config.gmmap_dataset;
    param.num_threads = config.gmmap_num_threads;
    if (param.num_threads <= 0) {
#ifdef _OPENMP
        param.num_threads = std::max(1, omp_get_max_threads());
#else
        param.num_threads = 1;
#endif
    }
    param.measure_memory = config.gmmap_measure_memory;
    param.frame_alg_name = config.gmmap_frame_alg_name;
    param.max_depth = static_cast<gmm::FP>(
        config.gmmap_max_depth > 0.0
            ? config.gmmap_max_depth
            : static_cast<double>(depth.depth.maxCoeff()));
    param.hell_thresh_squard_free =
        static_cast<gmm::FP>(config.gmmap_hell_thresh_squared_free);
    param.hell_thresh_squard_obs_scale =
        static_cast<gmm::FP>(config.gmmap_hell_thresh_squared_obs_scale);
    param.hell_thresh_squard_oversized_gau =
        static_cast<gmm::FP>(config.gmmap_hell_thresh_squared_oversized_gau);
    param.hell_thresh_squard_min =
        static_cast<gmm::FP>(config.gmmap_hell_thresh_squared_min);
    param.min_gau_len =
        static_cast<gmm::FP>(config.gmmap_min_gaussian_length);
    param.frame_max_scale =
        static_cast<gmm::FP>(config.gmmap_frame_max_scale);
    param.fusion_max_scale =
        static_cast<gmm::FP>(config.gmmap_fusion_max_scale);
    param.gau_fusion_bd =
        static_cast<gmm::FP>(config.gmmap_fusion_bound);
    param.gau_rtree_bd =
        static_cast<gmm::FP>(config.gmmap_rtree_bound_scale);
    param.depth_scale = static_cast<gmm::FP>(config.gmmap_depth_scale);
    param.track_color = config.gmmap_track_color;
    param.track_intensity = config.gmmap_track_intensity;
    param.cur_debug_frame = config.gmmap_cur_debug_frame;
    param.min_num_neighbor_clusters = config.gmmap_min_num_neighbor_clusters;
    param.max_bbox_len = param.min_gau_len * param.fusion_max_scale;
    param.hell_thresh_squard_obs =
        param.hell_thresh_squard_free * param.hell_thresh_squard_obs_scale;

    gmm::frame_param& frame = param.gmm_frame_param;
    frame.dataset = param.dataset;
    frame.img_width = depth.depth.cols();
    frame.img_height = depth.depth.rows();
    frame.num_threads = param.num_threads;
    frame.measure_memory = param.measure_memory;
    frame.preserve_details_far_objects = true;
    frame.occ_x_t = config.gmmap_occ_x_threshold;
    frame.noise_thresh =
        static_cast<gmm::FP>(config.gmmap_noise_threshold);
    frame.sparse_t = config.gmmap_sparse_threshold;
    frame.ncheck_t = config.gmmap_ncheck_threshold;
    frame.gau_bd_scale =
        static_cast<gmm::FP>(config.gmmap_rtree_bound_scale);
    frame.adaptive_thresh_scale =
        static_cast<gmm::FP>(config.gmmap_adaptive_threshold_scale);
    frame.max_depth = param.max_depth;
    frame.f = static_cast<gmm::FP>(
        std::min(std::abs(depth.fx), std::abs(depth.fy)));
    frame.fx = static_cast<gmm::FP>(depth.fx);
    frame.fy = static_cast<gmm::FP>(depth.fy);
    frame.cx = static_cast<gmm::FP>(depth.cx);
    frame.cy = static_cast<gmm::FP>(depth.cy);
    frame.line_t = static_cast<gmm::FP>(config.gmmap_line_threshold);
    frame.angle_t = static_cast<gmm::FP>(config.gmmap_angle_threshold);
    frame.noise_floor = static_cast<gmm::FP>(config.gmmap_noise_floor);
    frame.num_line_t = config.gmmap_num_line_threshold;
    frame.num_pixels_t = config.gmmap_num_pixels_threshold;
    frame.max_incomplete_clusters = config.gmmap_max_incomplete_clusters;
    frame.free_space_start_len = param.min_gau_len;
    frame.free_space_max_length = param.frame_max_scale * param.min_gau_len;
    frame.free_space_dist_scale =
        static_cast<gmm::FP>(config.gmmap_free_space_dist_scale);
    frame.debug_row_idx = config.gmmap_debug_row_idx;
    // GMMap prints every adaptive depth bin to std::cout; keep ROS logs readable.
    ScopedCoutSilencer silence_compute_depth(true);
    frame.computeDepth(true);

    return param;
}

} // namespace
#endif

GmmModel fitGmmap(const OrganizedDepthImage& depth,
                  const SogmmConfig& config) {
#ifndef GMMSLAM_HAS_GMMAP
    (void)depth;
    (void)config;
    throw std::runtime_error("GMMap fitting backend was not built");
#else
    if (!depth.valid()) {
        return {};
    }

    const auto t0 = std::chrono::steady_clock::now();

    gmm::GMMMap mapper;
    mapper.mapParameters = makeMapParams(depth, config);

    gmm::RowMatrixXf depthmap = depth.depth;
    std::list<gmm::GMMmetadata_c> obs_metadata =
        mapper.extendedSPGFOpt(depthmap);
    std::list<gmm::GMMmetadata_o> free_metadata;
    std::list<gmm::GMMcluster_o*> obs_clusters;
    std::list<gmm::GMMcluster_o*> free_clusters;
    mapper.transferMetadata2ClusterExtended(
        obs_metadata, free_metadata, obs_clusters, free_clusters,
        mapper.mapParameters.gmm_frame_param);

    GmmModel result;
    result.components.reserve(obs_clusters.size());
    const Eigen::Matrix3d R_gmmap_to_sensor = gmmapOpticalToSensorRotation();
    double max_component_range = 0.0;
    for (const auto* cluster : obs_clusters) {
        if (cluster == nullptr) {
            continue;
        }

        const Eigen::Vector3d mean_gmmap = cluster->Mean().cast<double>();
        const Eigen::Matrix3f cov = cluster->Cov();
        if (!mean_gmmap.allFinite() || !cov.allFinite() || cluster->W <= 0.0f) {
            continue;
        }

        Eigen::Matrix3d cov_d =
            R_gmmap_to_sensor * cov.cast<double>() * R_gmmap_to_sensor.transpose();
        cov_d = 0.5 * (cov_d + cov_d.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov_d);
        if (eig.info() != Eigen::Success ||
            (eig.eigenvalues().array() <= 0.0).any()) {
            continue;
        }
        Vector3d evals = eig.eigenvalues();
        // Keep GMMap's anisotropy for visualization and map geometry. The
        // registration-file export applies its own safer covariance clamp, so
        // this path only needs a tiny numerical floor.
        evals = evals.cwiseMax(1e-8);
        cov_d = eig.eigenvectors() * evals.asDiagonal() *
                eig.eigenvectors().transpose();
        cov_d = 0.5 * (cov_d + cov_d.transpose());

        GmmComponent comp;
        comp.mean = R_gmmap_to_sensor * mean_gmmap;
        comp.covariance = cov_d;
        comp.weight = static_cast<double>(cluster->W);
        max_component_range = std::max(max_component_range, comp.mean.norm());
        result.components.push_back(std::move(comp));
    }

    const double native_max_component_range = max_component_range;
    const FarFillStats far_fill =
        appendFarDepthComponents(depth, config, native_max_component_range, result);
    max_component_range =
        std::max(max_component_range, far_fill.max_range);
    const int raw_components = result.numComponents();
    const int component_cap =
        capComponentsByWeight(result, config.gmmap_max_components);
    normalizeWeights(result);

    const auto t1 = std::chrono::steady_clock::now();
    const double dt_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    ROS_INFO("[gmmap] fit depth=%dx%d valid=%d max_depth=%.2f K=%d raw_K=%d "
             "cap=%d native_max_range=%.2f far_fill=%d max_gmm_range=%.2f %.1f ms "
             "fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
             static_cast<int>(depth.depth.cols()),
             static_cast<int>(depth.depth.rows()), depth.valid_points,
             static_cast<double>(depth.depth.maxCoeff()),
             result.numComponents(), raw_components, component_cap,
             native_max_component_range,
             far_fill.added_components, max_component_range,
             dt_ms, depth.fx, depth.fy, depth.cx, depth.cy);

    return result;
#endif
}

} // namespace gmmslam
