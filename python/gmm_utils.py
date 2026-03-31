"""GMM projection, filtering, saving, and plotting utilities."""

import os

import numpy as np
import rospy

from imports_setup import HAS_GMM_REGISTRATION, save_gmm_official


def project_gmm_4d_to_3d(gmm_4d):
    """Project a 4D SOGMM to 3D for registration.

    SOGMM produces 4D GMMs (x, y, z, range), but registration expects 3D.
    """
    from sklearn.mixture import GaussianMixture

    K = gmm_4d.n_components_
    gmm_3d = GaussianMixture(n_components=K, covariance_type="full")

    if np.any(np.isnan(gmm_4d.means_)):
        rospy.logerr("[project_gmm] Input 4D GMM has NaN means!")
        raise ValueError("Input GMM contains NaN values")
    if np.any(np.isnan(gmm_4d.covariances_)):
        rospy.logerr("[project_gmm] Input 4D GMM has NaN covariances!")
        raise ValueError("Input GMM contains NaN values")

    gmm_3d.means_ = gmm_4d.means_[:, :3].copy()

    if gmm_4d.covariances_.ndim == 3:
        gmm_3d.covariances_ = gmm_4d.covariances_[:, :3, :3].copy()
    elif gmm_4d.covariances_.ndim == 2:
        D_orig = gmm_4d.means_.shape[1]
        cov_3d = np.zeros((K, 3, 3))
        for k in range(K):
            cov_4d = gmm_4d.covariances_[k].reshape(D_orig, D_orig)
            cov_3d[k] = cov_4d[:3, :3]
        gmm_3d.covariances_ = cov_3d
    else:
        raise ValueError(f"Unexpected covariance shape: {gmm_4d.covariances_.shape}")

    gmm_3d.weights_ = gmm_4d.weights_.copy()
    return filter_well_conditioned_gmm(gmm_3d)


def filter_well_conditioned_gmm(gmm_3d, reg: float = 1e-4):
    """Keep only components with well-conditioned 3x3 covariances.

    Only drops components with NaN/Inf or non-positive eigenvalues after
    regularization.  Planar (near-singular) components are intentionally kept
    because isoplanar_registration specifically relies on them.
    """
    from sklearn.mixture import GaussianMixture

    if hasattr(gmm_3d, "means_"):
        K = int(gmm_3d.means_.shape[0])
    else:
        K = int(gmm_3d.n_components_)
    keep = []
    dropped_bad = []

    for k in range(K):
        cov = gmm_3d.covariances_[k].copy()
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            dropped_bad.append(k)
            continue

        cov = 0.5 * (cov + cov.T) + reg * np.eye(3)
        try:
            eigvals = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            dropped_bad.append(k)
            continue

        if (
            np.any(np.isnan(eigvals))
            or np.any(np.isinf(eigvals))
            or eigvals.min() <= 0.0
        ):
            dropped_bad.append(k)
            continue

        keep.append((k, cov))

    if dropped_bad:
        rospy.logwarn_throttle(
            10.0,
            f"[filter_gmm] dropped {len(dropped_bad)}/{K} bad components (NaN/Inf/non-PD): {dropped_bad[:5]}",
        )

    if not keep:
        raise ValueError("No well-conditioned covariance matrices left after filtering")

    keep_idx = [i for i, _ in keep]
    covs = np.stack([cov for _, cov in keep], axis=0)
    weights = gmm_3d.weights_[keep_idx].astype(np.float64)
    weights_sum = weights.sum()
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        raise ValueError("Invalid weights after filtering GMM components")
    weights /= weights_sum

    filtered = GaussianMixture(n_components=len(keep_idx), covariance_type="full")
    filtered.means_ = gmm_3d.means_[keep_idx].copy()
    filtered.covariances_ = covs
    filtered.weights_ = weights
    filtered.n_components_ = len(keep_idx)
    filtered.converged_ = True
    filtered.n_iter_ = 0
    filtered.lower_bound_ = -np.inf
    try:
        filtered.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
    except np.linalg.LinAlgError:
        covs_safe = covs + 1e-3 * np.eye(3)
        filtered.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs_safe))

    if len(keep_idx) != K:
        rospy.loginfo(
            f"[filter_gmm] keeping {len(keep_idx)}/{K} well-conditioned components"
        )

    return filtered


def precompute_gmm_local_data(gmm):
    """Eigen-decompose each 3D component once (pose-independent).

    Returns a list of (scales, R_local, mu_local) tuples.
    """
    K = gmm.n_components_ if hasattr(gmm, "n_components_") else gmm.n_components
    D = gmm.means_.shape[1]
    covs_raw = gmm.covariances_
    if covs_raw.ndim == 2:
        covs_raw = covs_raw.reshape(K, D, D)

    components = []
    for k in range(K):
        cov3 = 0.5 * (covs_raw[k, :3, :3] + covs_raw[k, :3, :3].T)
        eigvals, eigvecs = np.linalg.eigh(cov3)
        eigvals = np.maximum(eigvals, 1e-9)
        R_local = eigvecs.copy()
        if np.linalg.det(R_local) < 0:
            R_local[:, 0] *= -1
        scales = np.sqrt(eigvals)
        mu_local = gmm.means_[k, :3].copy()
        components.append((scales, R_local, mu_local))
    return components


def merge_gmms_concatenate(gmms_with_poses, T_ref):
    """Concatenate multiple 3D GMMs into one, transforming to a reference frame.

    Parameters
    ----------
    gmms_with_poses : list of (gmm, T_world_4x4)
        Each entry is a keyframe GMM (3D) and its world pose.
    T_ref : np.ndarray (4x4)
        The submap reference pose.  Components are expressed in this frame.

    Returns a sklearn GaussianMixture with concatenated components and
    renormalized weights, or None if no valid components.
    """
    from sklearn.mixture import GaussianMixture

    T_ref_inv = np.linalg.inv(T_ref)
    all_means, all_covs, all_weights = [], [], []

    for gmm, T_w_kf in gmms_with_poses:
        if gmm is None or T_w_kf is None:
            continue
        K = gmm.n_components_ if hasattr(gmm, "n_components_") else gmm.n_components
        D = gmm.means_.shape[1]
        covs_raw = gmm.covariances_
        if covs_raw.ndim == 2:
            covs_raw = covs_raw.reshape(K, D, D)

        T_kf_in_ref = T_ref_inv @ T_w_kf
        R = T_kf_in_ref[:3, :3]
        t = T_kf_in_ref[:3, 3]

        for k in range(K):
            mu_ref = R @ gmm.means_[k, :3] + t
            cov_ref = R @ covs_raw[k, :3, :3] @ R.T
            cov_ref = 0.5 * (cov_ref + cov_ref.T)
            all_means.append(mu_ref)
            all_covs.append(cov_ref)
            all_weights.append(gmm.weights_[k])

    if not all_means:
        return None

    means = np.array(all_means, dtype=np.float64)
    covs = np.array(all_covs, dtype=np.float64)
    weights = np.array(all_weights, dtype=np.float64)
    weights /= weights.sum()
    n = len(all_means)

    merged = GaussianMixture(n_components=n, covariance_type="full")
    merged.means_ = means
    merged.covariances_ = covs
    merged.weights_ = weights
    merged.n_components_ = n
    merged.converged_ = True
    merged.n_iter_ = 0
    merged.lower_bound_ = -np.inf
    try:
        merged.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
    except np.linalg.LinAlgError:
        covs_safe = covs + 1e-4 * np.eye(3)
        merged.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs_safe))
    return merged


# Distinct hues for submap visualization (cycles if > 10 submaps)
SUBMAP_COLORS = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
    (0.98, 0.60, 0.60),
]


def save_gmm_to_file(gmm, filepath: str):
    """Save a GaussianMixture to .gmm format using gira3d's official save utility."""
    if not HAS_GMM_REGISTRATION or save_gmm_official is None:
        raise RuntimeError("save_gmm utility not available")

    if gmm.means_.shape[1] == 4:
        gmm_3d = project_gmm_4d_to_3d(gmm)
    else:
        gmm_3d = filter_well_conditioned_gmm(gmm)

    if gmm_3d is None:
        return None

    nan_count = 0
    inf_count = 0
    for k in range(gmm_3d.n_components_):
        if np.any(np.isnan(gmm_3d.means_[k])):
            nan_count += 1
        if np.any(np.isnan(gmm_3d.covariances_[k])):
            nan_count += 1
        if np.any(np.isinf(gmm_3d.covariances_[k])):
            inf_count += 1

    if nan_count > 0 or inf_count > 0:
        rospy.logerr(
            f"[save_gmm] GMM has {nan_count} NaN and {inf_count} Inf components!"
        )

    save_gmm_official(filepath, gmm_3d)


def plot_gmm_3d(
    gmm, sigma: float = 1.0, max_ellipsoids: int = 50, title: str = "GMM 3D Gaussians"
):
    """Visualize a 3D GMM as ellipsoids in a native matplotlib window."""
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gmm.means_.shape[1] == 4:
        gmm = project_gmm_4d_to_3d(gmm)

    K = gmm.n_components_
    n_u, n_v = 20, 14
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)).ravel(),
            np.outer(np.sin(u), np.sin(v)).ravel(),
            np.outer(np.ones(n_u), np.cos(v)).ravel(),
        ],
        axis=1,
    )

    top_idx = np.argsort(gmm.weights_)[::-1][:max_ellipsoids]
    max_w = gmm.weights_[top_idx[0]]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{title}  (K={K}, showing top-{min(max_ellipsoids, K)})")

    for k in top_idx:
        mu = gmm.means_[k]
        cov = gmm.covariances_[k]
        w = gmm.weights_[k]
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        axes = eigvecs * (np.sqrt(eigvals) * sigma)
        epts = (axes @ sphere.T).T + mu
        ex = epts[:, 0].reshape(n_u, n_v)
        ey = epts[:, 1].reshape(n_u, n_v)
        ez = epts[:, 2].reshape(n_u, n_v)
        alpha = 0.15 + 0.45 * (w / max_w)
        ax.plot_surface(ex, ey, ez, color="steelblue", alpha=alpha, linewidth=0)

    ax.scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        gmm.means_[:, 2],
        c="red",
        s=40,
        zorder=5,
        label="Means",
        depthshade=False,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    return fig
