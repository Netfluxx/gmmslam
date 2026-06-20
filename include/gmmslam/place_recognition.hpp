#pragma once

#include "gmmslam/config.hpp"
#include "gmmslam/solid.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace gmmslam {

// Thread-safe in-memory index of SOLiD descriptors keyed by keyframe index.
// Used as an appearance-based gate on top of the radius loop search and as a
// rescue index when the smoother's pose drifts past the radius. Descriptors
// are searched by exact KD-tree nearest-neighbor lookup over unit-normalized
// range vectors, which preserves the existing cosine ranking.
class PlaceRecognitionIndex {
public:
    explicit PlaceRecognitionIndex(const SolidConfig& cfg);

    bool enabled() const { return cfg_.enable; }
    const SolidConfig& config() const { return cfg_; }
    const SOLiDModule& module() const { return solid_; }

    // Build a descriptor from a preprocessed Nx3 cloud in the sensor frame.
    SolidDescriptor compute(const Eigen::MatrixXf& pts) const;

    // Insert / fetch / remove.
    void insert(int idx, SolidDescriptor desc);
    bool get(int idx, SolidDescriptor& out) const;
    void eraseOlderThan(int min_idx);
    std::size_t size() const;

    // Cosine similarity on the range head (forwards to SOLiDModule).
    double rangeCosine(const SolidDescriptor& q,
                       const SolidDescriptor& c) const {
        return solid_.rangeCosine(q, c);
    }

    // FOV-aware yaw estimate (forwards to SOLiDModule).
    SOLiDModule::YawEstimate yawEstimate(const SolidDescriptor& q,
                                         const SolidDescriptor& c) const {
        return solid_.yawEstimate(q, c);
    }

    // Score a provided candidate list against `q`. For each index the output
    // score is the cosine similarity, or NaN if the index is not in the
    // cache. out_scores is resized to indices.size() on return.
    void scoreMany(const SolidDescriptor& q,
                   const std::vector<int>& indices,
                   std::vector<double>& out_scores) const;

    // Top-K cosine similarity search over the whole cache. Only indices
    // satisfying `accept(idx)` are eligible; a null `accept` accepts all.
    // Results are returned sorted by descending similarity.
    std::vector<std::pair<int, double>> topK(
        const SolidDescriptor& q,
        int K,
        const std::function<bool(int)>& accept = nullptr) const;

private:
    struct KdEntry {
        int idx = -1;
        Eigen::VectorXd unit_range;
    };

    struct KdNode {
        int entry_pos = -1;
        int axis = 0;
        int left = -1;
        int right = -1;
    };

    static Eigen::VectorXd normalizedRangeVector(const SolidDescriptor& desc,
                                                 int num_range);
    static double cosineFromUnitDistanceSq(double dist_sq);

    void markKdDirtyLocked();
    void rebuildKdTreeLocked() const;
    int buildKdTreeLocked(std::vector<int>& positions,
                          int begin, int end,
                          int depth) const;

    SolidConfig cfg_;
    SOLiDModule solid_;
    mutable std::mutex mutex_;
    std::vector<std::pair<int, SolidDescriptor>> by_idx_;
    mutable std::vector<KdEntry> kd_entries_;
    mutable std::vector<KdNode> kd_nodes_;
    mutable int kd_root_ = -1;
    mutable bool kd_dirty_ = true;
};

} // namespace gmmslam
