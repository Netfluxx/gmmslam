#include "gmmslam/place_recognition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>

namespace gmmslam {

PlaceRecognitionIndex::PlaceRecognitionIndex(const SolidConfig& cfg)
    : cfg_(cfg), solid_(cfg) {}

SolidDescriptor PlaceRecognitionIndex::compute(const Eigen::MatrixXf& pts) const {
    return solid_.makeDescriptor(pts);
}

void PlaceRecognitionIndex::insert(int idx, SolidDescriptor desc) {
    if (desc.empty()) return;
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = std::lower_bound(
        by_idx_.begin(), by_idx_.end(), idx,
        [](const auto& entry, int value) { return entry.first < value; });
    if (it != by_idx_.end() && it->first == idx) {
        it->second = std::move(desc);
    } else {
        by_idx_.emplace(it, idx, std::move(desc));
    }
    markKdDirtyLocked();

    // If an absolute cap is configured, trim the oldest entries.
    if (cfg_.keep_descriptors > 0 &&
        static_cast<int>(by_idx_.size()) > cfg_.keep_descriptors) {
        const int to_drop =
            static_cast<int>(by_idx_.size()) - cfg_.keep_descriptors;
        by_idx_.erase(by_idx_.begin(),
                      by_idx_.begin() + std::min<int>(
                          to_drop, static_cast<int>(by_idx_.size())));
        markKdDirtyLocked();
    }
}

bool PlaceRecognitionIndex::get(int idx, SolidDescriptor& out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = std::lower_bound(
        by_idx_.begin(), by_idx_.end(), idx,
        [](const auto& entry, int value) { return entry.first < value; });
    if (it == by_idx_.end() || it->first != idx) return false;
    out = it->second;
    return true;
}

void PlaceRecognitionIndex::eraseOlderThan(int min_idx) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto keep_begin = std::lower_bound(
        by_idx_.begin(), by_idx_.end(), min_idx,
        [](const auto& entry, int value) { return entry.first < value; });
    if (keep_begin != by_idx_.begin()) {
        by_idx_.erase(by_idx_.begin(), keep_begin);
        markKdDirtyLocked();
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
        auto it = std::lower_bound(
            by_idx_.begin(), by_idx_.end(), indices[i],
            [](const auto& entry, int value) { return entry.first < value; });
        if (it == by_idx_.end() || it->first != indices[i]) continue;
        out_scores[i] = solid_.rangeCosine(q, it->second);
    }
}

Eigen::VectorXd
PlaceRecognitionIndex::normalizedRangeVector(const SolidDescriptor& desc,
                                             int num_range) {
    if (desc.empty() || num_range <= 0 || desc.vec.size() < num_range ||
        desc.range_norm <= 1e-12) {
        return {};
    }

    Eigen::VectorXd unit = desc.vec.head(num_range);
    unit /= desc.range_norm;
    if (!unit.allFinite()) {
        return {};
    }
    return unit;
}

double PlaceRecognitionIndex::cosineFromUnitDistanceSq(double dist_sq) {
    if (!std::isfinite(dist_sq)) {
        return 0.0;
    }
    return std::clamp(1.0 - 0.5 * dist_sq, 0.0, 1.0);
}

void PlaceRecognitionIndex::markKdDirtyLocked() {
    kd_dirty_ = true;
}

void PlaceRecognitionIndex::rebuildKdTreeLocked() const {
    if (!kd_dirty_) {
        return;
    }

    kd_entries_.clear();
    kd_nodes_.clear();
    kd_root_ = -1;
    kd_entries_.reserve(by_idx_.size());

    for (const auto& [idx, desc] : by_idx_) {
        Eigen::VectorXd unit = normalizedRangeVector(desc, cfg_.num_range);
        if (unit.size() != cfg_.num_range) {
            continue;
        }
        kd_entries_.push_back({idx, std::move(unit)});
    }

    std::vector<int> positions(kd_entries_.size());
    std::iota(positions.begin(), positions.end(), 0);
    kd_nodes_.reserve(positions.size());
    kd_root_ = buildKdTreeLocked(
        positions, 0, static_cast<int>(positions.size()), 0);
    kd_dirty_ = false;
}

int
PlaceRecognitionIndex::buildKdTreeLocked(std::vector<int>& positions,
                                         int begin, int end,
                                         int depth) const {
    if (begin >= end || cfg_.num_range <= 0) {
        return -1;
    }

    const int axis = depth % cfg_.num_range;
    const int mid = begin + (end - begin) / 2;
    std::nth_element(
        positions.begin() + begin,
        positions.begin() + mid,
        positions.begin() + end,
        [&](int a, int b) {
            const auto& ea = kd_entries_[static_cast<std::size_t>(a)];
            const auto& eb = kd_entries_[static_cast<std::size_t>(b)];
            const double va = ea.unit_range(axis);
            const double vb = eb.unit_range(axis);
            if (va == vb) {
                return ea.idx < eb.idx;
            }
            return va < vb;
        });

    const int node_idx = static_cast<int>(kd_nodes_.size());
    kd_nodes_.push_back(KdNode{});
    kd_nodes_[static_cast<std::size_t>(node_idx)].entry_pos =
        positions[static_cast<std::size_t>(mid)];
    kd_nodes_[static_cast<std::size_t>(node_idx)].axis = axis;
    kd_nodes_[static_cast<std::size_t>(node_idx)].left =
        buildKdTreeLocked(positions, begin, mid, depth + 1);
    kd_nodes_[static_cast<std::size_t>(node_idx)].right =
        buildKdTreeLocked(positions, mid + 1, end, depth + 1);
    return node_idx;
}

std::vector<std::pair<int, double>>
PlaceRecognitionIndex::topK(const SolidDescriptor& q, int K,
                             const std::function<bool(int)>& accept) const {
    std::vector<std::pair<int, double>> results;
    if (q.empty() || K <= 0) return results;

    const Eigen::VectorXd q_unit = normalizedRangeVector(q, cfg_.num_range);
    if (q_unit.size() != cfg_.num_range) return results;

    std::lock_guard<std::mutex> lk(mutex_);
    rebuildKdTreeLocked();
    if (kd_root_ < 0) return results;

    using HeapItem = std::pair<double, int>;  // squared distance, idx
    std::priority_queue<HeapItem> best;

    auto maybe_add = [&](int entry_pos) {
        const auto& entry =
            kd_entries_[static_cast<std::size_t>(entry_pos)];
        if (accept && !accept(entry.idx)) return;

        const double dist_sq = (q_unit - entry.unit_range).squaredNorm();
        if (!std::isfinite(dist_sq)) return;

        const HeapItem item{dist_sq, entry.idx};
        if (static_cast<int>(best.size()) < K) {
            best.push(item);
            return;
        }
        if (item < best.top()) {
            best.pop();
            best.push(item);
        }
    };

    std::function<void(int)> search = [&](int node_idx) {
        if (node_idx < 0) return;

        const KdNode& node =
            kd_nodes_[static_cast<std::size_t>(node_idx)];
        const auto& entry =
            kd_entries_[static_cast<std::size_t>(node.entry_pos)];
        const double split_delta = q_unit(node.axis) -
                                   entry.unit_range(node.axis);

        const int near_child =
            split_delta <= 0.0 ? node.left : node.right;
        const int far_child =
            split_delta <= 0.0 ? node.right : node.left;

        search(near_child);
        maybe_add(node.entry_pos);

        const double split_dist_sq = split_delta * split_delta;
        if (static_cast<int>(best.size()) < K ||
            split_dist_sq <= best.top().first + 1e-12) {
            search(far_child);
        }
    };

    search(kd_root_);

    results.reserve(best.size());
    while (!best.empty()) {
        const auto [dist_sq, idx] = best.top();
        best.pop();
        const double s = cosineFromUnitDistanceSq(dist_sq);
        if (std::isfinite(s) && s > 0.0) {
            results.emplace_back(idx, s);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  if (a.second == b.second) {
                      return a.first < b.first;
                  }
                  return a.second > b.second;
              });
    return results;
}

} // namespace gmmslam
