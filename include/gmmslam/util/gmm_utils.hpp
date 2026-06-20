#pragma once
#include "gmmslam/config.hpp"
#include "gmmslam/types.hpp"
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gmmslam {

struct PosedGmmInput {
    GmmModel model;
    Matrix4d pose = Matrix4d::Identity();
    int key_idx = -1;
    double pose_uncertainty = std::numeric_limits<double>::infinity();
};

struct PruneDebugRecord {
    Vector3d kept_mean = Vector3d::Zero();
    Vector3d merged_mean = Vector3d::Zero();
    double distance = 0.0;
    int cluster_id = -1;
    int kept_source_key_idx = -1;
    int merged_source_key_idx = -1;
    int kept_component_index = -1;
    int merged_component_index = -1;
    std::string reason;
};

struct PruneResult {
    GmmModel model;
    std::vector<PruneDebugRecord> debug_records;
    int clusters_merged = 0;
    int components_removed = 0;
};

GmmModel filterWellConditioned(const GmmModel& model, double reg = 1e-4);

std::vector<GmmLocalData> precomputeGmmLocalData(const GmmModel& model);

// Concatenate per-keyframe GMMs into a single GMM expressed in T_ref frame
// without any deduplication. Kept for callers that explicitly want the raw
// union of components.
GmmModel mergeGmmsConcatenate(
    const std::vector<std::pair<GmmModel, Matrix4d>>& gmms_with_poses,
    const Matrix4d& T_ref);

GmmModel mergeGmmsConcatenate(
    const std::vector<PosedGmmInput>& gmms_with_poses,
    const Matrix4d& T_ref);

// Concatenate per-keyframe GMMs into T_ref frame and then prune near-duplicate
// components using conservative candidate search plus a Bhattacharyya gate.
// When MapConfig::prune_enable is false this degrades to plain concatenation.
GmmModel mergeGmmsAndPrune(
    const std::vector<std::pair<GmmModel, Matrix4d>>& gmms_with_poses,
    const Matrix4d& T_ref,
    const MapConfig& map_cfg);

GmmModel mergeGmmsAndPrune(
    const std::vector<PosedGmmInput>& gmms_with_poses,
    const Matrix4d& T_ref,
    const MapConfig& map_cfg);

// In-place pruning of an already-merged GMM. Exposed for callers that build
// the pre-pruned model themselves (e.g. tests).
GmmModel pruneSimilarComponents(const GmmModel& in, const MapConfig& map_cfg);
PruneResult pruneSimilarComponentsDetailed(const GmmModel& in,
                                           const MapConfig& map_cfg);

// Prune duplicates only when they come from different source GMM frames.
// When a duplicate pair is found, keep the older source frame and drop the
// duplicate component from the newer measurement.
GmmModel pruneNewerFrameComponents(const GmmModel& in, const MapConfig& map_cfg);
PruneResult pruneNewerFrameComponentsDetailed(const GmmModel& in,
                                              const MapConfig& map_cfg);

// Bhattacharyya distance between two 3D Gaussians. Returns
// std::numeric_limits<double>::infinity() if either covariance is
// numerically singular.
double bhattacharyyaDistance(
    const Vector3d& mu_a, const Matrix3d& cov_a,
    const Vector3d& mu_b, const Matrix3d& cov_b,
    double cov_reg = 1e-6);

void saveGmmToFile(const GmmModel& model, const std::string& filepath);

const std::vector<std::array<double, 3>>& submapColors();

std::array<double, 3> submapColor(int sid);

} // namespace gmmslam
