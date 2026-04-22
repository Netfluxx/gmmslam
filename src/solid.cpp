#include "gmmslam/solid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace gmmslam {

namespace {

inline double wrapToPi(double x) {
    while (x >  M_PI) x -= 2.0 * M_PI;
    while (x <= -M_PI) x += 2.0 * M_PI;
    return x;
}

// Simple in-place voxel downsample on an Nx3 float matrix. Returns a new
// matrix with <= pts.rows() rows. leaf_size <= 0 disables.
Eigen::MatrixXf voxelDownsample(const Eigen::MatrixXf& pts, double leaf_size) {
    if (leaf_size <= 0.0 || pts.rows() == 0) return pts;

    const float inv = 1.0f / static_cast<float>(leaf_size);
    // Hash keys by packing (ix, iy, iz) into a 64-bit integer.
    struct KeyAcc { double sx = 0.0, sy = 0.0, sz = 0.0; int n = 0; };
    std::unordered_map<std::int64_t, KeyAcc> bins;
    bins.reserve(static_cast<std::size_t>(pts.rows()));

    auto pack = [](std::int32_t a, std::int32_t b, std::int32_t c) {
        // XOR-shift hashing is sufficient here; collisions only cost accuracy,
        // not correctness (we keep the bin centroid).
        std::int64_t h = static_cast<std::int64_t>(a) * 73856093LL;
        h ^= static_cast<std::int64_t>(b) * 19349663LL;
        h ^= static_cast<std::int64_t>(c) * 83492791LL;
        return h;
    };

    for (int i = 0; i < pts.rows(); ++i) {
        const auto ix = static_cast<std::int32_t>(std::floor(pts(i, 0) * inv));
        const auto iy = static_cast<std::int32_t>(std::floor(pts(i, 1) * inv));
        const auto iz = static_cast<std::int32_t>(std::floor(pts(i, 2) * inv));
        const auto key = pack(ix, iy, iz);
        auto& acc = bins[key];
        acc.sx += pts(i, 0); acc.sy += pts(i, 1); acc.sz += pts(i, 2);
        acc.n  += 1;
    }

    Eigen::MatrixXf out(static_cast<int>(bins.size()), 3);
    int r = 0;
    for (const auto& kv : bins) {
        const auto& a = kv.second;
        out(r, 0) = static_cast<float>(a.sx / a.n);
        out(r, 1) = static_cast<float>(a.sy / a.n);
        out(r, 2) = static_cast<float>(a.sz / a.n);
        ++r;
    }
    return out;
}

} // namespace

SOLiDModule::SOLiDModule(const SolidConfig& cfg) : cfg_(cfg) {
    // Guard against degenerate configurations.
    cfg_.num_angle  = std::max(1, cfg_.num_angle);
    cfg_.num_range  = std::max(1, cfg_.num_range);
    cfg_.num_height = std::max(1, cfg_.num_height);
    if (cfg_.max_distance_m <= 0.0) cfg_.max_distance_m = 1.0;
    if (cfg_.fov_up_deg <= cfg_.fov_down_deg) {
        cfg_.fov_up_deg = cfg_.fov_down_deg + 1.0;
    }

    gap_angle_deg_  = 360.0 / static_cast<double>(cfg_.num_angle);
    gap_range_m_    = cfg_.max_distance_m / static_cast<double>(cfg_.num_range);
    gap_height_deg_ = (cfg_.fov_up_deg - cfg_.fov_down_deg)
                      / static_cast<double>(cfg_.num_height);
}

SolidDescriptor SOLiDModule::makeDescriptor(const Eigen::MatrixXf& raw) const {
    SolidDescriptor desc;
    desc.vec = Eigen::VectorXd::Zero(cfg_.num_range + cfg_.num_angle);

    if (raw.rows() == 0) return desc;

    // Optional coarser voxel for descriptor construction — the gmmslam
    // preprocess voxel (5 cm) is much finer than SOLiD needs. Falls through
    // untouched when voxel_size_m <= 0.
    const Eigen::MatrixXf pts = voxelDownsample(raw, cfg_.voxel_size_m);

    Eigen::MatrixXd range_matrix = Eigen::MatrixXd::Zero(cfg_.num_range,
                                                         cfg_.num_height);
    Eigen::MatrixXd angle_matrix = Eigen::MatrixXd::Zero(cfg_.num_angle,
                                                         cfg_.num_height);

    int accepted = 0;
    for (int i = 0; i < pts.rows(); ++i) {
        const double x = pts(i, 0);
        const double y = pts(i, 1);
        const double z = pts(i, 2);

        const double dist_xy = std::sqrt(x * x + y * y);
        if (dist_xy < cfg_.min_distance_m || dist_xy > cfg_.max_distance_m) {
            continue;
        }

        // atan2 -> [0, 360) degrees, CCW from +X axis.
        double theta_deg = std::atan2(y, x) * 180.0 / M_PI;
        if (theta_deg < 0.0) theta_deg += 360.0;
        if (theta_deg >= 360.0) theta_deg -= 360.0;

        const double phi_deg = std::atan2(z, dist_xy) * 180.0 / M_PI;
        if (phi_deg < cfg_.fov_down_deg || phi_deg > cfg_.fov_up_deg) {
            continue;
        }

        const int idx_range = std::min(
            static_cast<int>(dist_xy / gap_range_m_), cfg_.num_range - 1);
        const int idx_angle = std::min(
            static_cast<int>(theta_deg / gap_angle_deg_), cfg_.num_angle - 1);
        const int idx_height = std::min(
            static_cast<int>((phi_deg - cfg_.fov_down_deg) / gap_height_deg_),
            cfg_.num_height - 1);

        range_matrix(idx_range, idx_height) += 1.0;
        angle_matrix(idx_angle, idx_height) += 1.0;
        ++accepted;
    }
    desc.point_count = accepted;
    if (accepted == 0) return desc;

    // Height-weight vector, normalized to [0, 1]. The paper uses this to give
    // more weight to heights with more points, making the descriptor robust
    // to sparsity at the extremes of the vertical FOV.
    Eigen::VectorXd height_weight(cfg_.num_height);
    for (int c = 0; c < cfg_.num_height; ++c) {
        height_weight(c) = range_matrix.col(c).sum();
    }
    const double min_v = height_weight.minCoeff();
    const double max_v = height_weight.maxCoeff();
    if (max_v - min_v < 1e-12) {
        height_weight.setOnes();  // uniform weighting when all heights equal
    } else {
        height_weight = (height_weight.array() - min_v) / (max_v - min_v);
    }

    const Eigen::VectorXd range_vec = range_matrix * height_weight;
    const Eigen::VectorXd angle_vec = angle_matrix * height_weight;

    desc.vec.head(cfg_.num_range) = range_vec;
    desc.vec.tail(cfg_.num_angle) = angle_vec;
    desc.range_norm = range_vec.norm();
    return desc;
}

double SOLiDModule::rangeCosine(const SolidDescriptor& q,
                                 const SolidDescriptor& c) const {
    if (q.empty() || c.empty()) return 0.0;
    const double denom = q.range_norm * c.range_norm;
    if (denom < 1e-12) return 0.0;
    const auto qh = q.vec.head(cfg_.num_range);
    const auto ch = c.vec.head(cfg_.num_range);
    return std::clamp(qh.dot(ch) / denom, 0.0, 1.0);
}

SOLiDModule::YawEstimate
SOLiDModule::yawEstimate(const SolidDescriptor& q,
                          const SolidDescriptor& c) const {
    YawEstimate est;
    if (q.empty() || c.empty()) return est;

    const int N = cfg_.num_angle;
    const Eigen::VectorXd qa = q.vec.segment(cfg_.num_range, N);
    const Eigen::VectorXd ca = c.vec.segment(cfg_.num_range, N);

    // Non-zero support masks. The paper restricts matching to the overlap of
    // non-zero supports so that shift-minima coming from empty-bin alignment
    // are ruled out — critical for FOV-constrained scans.
    std::vector<uint8_t> qnz(N), cnz(N);
    int q_support = 0, c_support = 0;
    // Slightly looser than 1e-12 so weak bins still count toward overlap
    // (helps marginal loop pairs on sparse depth).
    constexpr double nz_eps = 1e-9;
    for (int i = 0; i < N; ++i) {
        qnz[i] = (qa(i) > nz_eps) ? 1 : 0;
        cnz[i] = (ca(i) > nz_eps) ? 1 : 0;
        q_support += qnz[i];
        c_support += cnz[i];
    }
    if (q_support == 0 || c_support == 0) return est;

    const double max_abs = std::clamp(cfg_.max_abs_yaw_deg, 0.0, 180.0);
    const int max_shift_bins = static_cast<int>(
        std::ceil(max_abs / gap_angle_deg_));

    // Rotate in place into a scratch buffer to avoid reallocations.
    Eigen::VectorXd shifted(N);
    std::vector<uint8_t> shifted_nz(N);

    double best_l1 = std::numeric_limits<double>::infinity();
    double best_overlap = 0.0;
    int    best_k = 0;
    bool   any_valid = false;

    // Iterate signed shift k in [-max_shift_bins, +max_shift_bins].
    for (int k_signed = -max_shift_bins; k_signed <= max_shift_bins; ++k_signed) {
        int k = ((k_signed % N) + N) % N;  // wrap into [0, N)

        int overlap_bins = 0;
        for (int i = 0; i < N; ++i) {
            const int j = (i + k) % N;
            shifted(j)    = qa(i);
            shifted_nz[j] = qnz[i];
        }
        for (int j = 0; j < N; ++j) {
            if (shifted_nz[j] && cnz[j]) ++overlap_bins;
        }

        const double overlap = static_cast<double>(overlap_bins)
                             / static_cast<double>(std::min(q_support, c_support));
        if (overlap < cfg_.overlap_min) continue;

        const double l1 = (ca - shifted).cwiseAbs().sum();
        if (l1 < best_l1) {
            best_l1 = l1;
            best_overlap = overlap;
            best_k = k_signed;
            any_valid = true;
        }
    }

    if (!any_valid) return est;

    const double yaw_deg = static_cast<double>(best_k) * gap_angle_deg_;
    const double signed_yaw = wrapToPi(yaw_deg * M_PI / 180.0
                                       * static_cast<double>(cfg_.yaw_sign));
    est.yaw_rad    = signed_yaw;
    est.l1_distance = best_l1;
    est.overlap    = best_overlap;
    est.valid      = true;
    return est;
}

} // namespace gmmslam
