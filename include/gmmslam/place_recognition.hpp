#pragma once

#include "gmmslam/config.hpp"
#include "gmmslam/solid.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace gmmslam {

// Thread-safe in-memory index of SOLiD descriptors keyed by keyframe index.
// Used as an appearance-based gate on top of the radius loop search and as a
// rescue index when the smoother's pose drifts past the radius. Descriptors
// are tiny (~cfg.num_range + cfg.num_angle doubles) so brute-force kNN scales
// to tens of thousands of frames without any I/O.
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
    SolidConfig cfg_;
    SOLiDModule solid_;
    mutable std::mutex mutex_;
    std::map<int, SolidDescriptor> by_idx_;
};

} // namespace gmmslam
