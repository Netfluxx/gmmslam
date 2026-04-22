#include "gmmslam/place_recognition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gmmslam {

PlaceRecognitionIndex::PlaceRecognitionIndex(const SolidConfig& cfg)
    : cfg_(cfg), solid_(cfg) {}

SolidDescriptor PlaceRecognitionIndex::compute(const Eigen::MatrixXf& pts) const {
    return solid_.makeDescriptor(pts);
}

void PlaceRecognitionIndex::insert(int idx, SolidDescriptor desc) {
    if (desc.empty()) return;
    std::lock_guard<std::mutex> lk(mutex_);
    by_idx_[idx] = std::move(desc);

    // If an absolute cap is configured, trim the oldest entries.
    if (cfg_.keep_descriptors > 0 &&
        static_cast<int>(by_idx_.size()) > cfg_.keep_descriptors) {
        const int to_drop =
            static_cast<int>(by_idx_.size()) - cfg_.keep_descriptors;
        auto it = by_idx_.begin();
        for (int i = 0; i < to_drop && it != by_idx_.end(); ++i) {
            it = by_idx_.erase(it);
        }
    }
}

bool PlaceRecognitionIndex::get(int idx, SolidDescriptor& out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = by_idx_.find(idx);
    if (it == by_idx_.end()) return false;
    out = it->second;
    return true;
}

void PlaceRecognitionIndex::eraseOlderThan(int min_idx) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto it = by_idx_.begin(); it != by_idx_.end();) {
        if (it->first < min_idx) {
            it = by_idx_.erase(it);
        } else {
            break;  // map is ordered
        }
    }
}

std::size_t PlaceRecognitionIndex::size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return by_idx_.size();
}

void PlaceRecognitionIndex::scoreMany(const SolidDescriptor& q,
                                       const std::vector<int>& indices,
                                       std::vector<double>& out_scores) const {
    out_scores.assign(indices.size(),
                      std::numeric_limits<double>::quiet_NaN());
    if (q.empty()) return;

    std::lock_guard<std::mutex> lk(mutex_);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto it = by_idx_.find(indices[i]);
        if (it == by_idx_.end()) continue;
        out_scores[i] = solid_.rangeCosine(q, it->second);
    }
}

std::vector<std::pair<int, double>>
PlaceRecognitionIndex::topK(const SolidDescriptor& q, int K,
                             const std::function<bool(int)>& accept) const {
    std::vector<std::pair<int, double>> results;
    if (q.empty() || K <= 0) return results;

    std::lock_guard<std::mutex> lk(mutex_);
    results.reserve(by_idx_.size());
    for (const auto& [idx, desc] : by_idx_) {
        if (accept && !accept(idx)) continue;
        const double s = solid_.rangeCosine(q, desc);
        if (!std::isfinite(s) || s <= 0.0) continue;
        results.emplace_back(idx, s);
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (static_cast<int>(results.size()) > K) {
        results.resize(static_cast<std::size_t>(K));
    }
    return results;
}

} // namespace gmmslam
