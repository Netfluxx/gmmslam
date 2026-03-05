
# # Terminal 1 — start the container and roscore
# docker exec -it disal_slam bash -c "source /opt/ros/noetic/setup.bash && roscore"

# # Terminal 2 — run gmmslam node
# docker exec -it disal_slam run_gmmslam

# # Terminal 2 — or run the test script
# docker exec -it disal_slam run_gmm_test



import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation

# scipy < 1.4 uses from_dcm; >= 1.4 uses from_matrix
if not hasattr(Rotation, "from_matrix"):
    Rotation.from_matrix = Rotation.from_dcm

import sys, os, glob

def _inject_venv(venv_root: str):
    """Add site-packages from a venv into sys.path (so that the modules can be imported once the venv is activated)."""
    pattern = os.path.join(venv_root, "lib", "python3*", "site-packages")
    for sp in glob.glob(pattern):
        if sp not in sys.path:
            sys.path.insert(0, sp)

_GIRA_WS = os.environ.get("GIRA_WS", "/root/gira_ws")
_RECONSTRUCTION_VENV = os.path.join(_GIRA_WS, "gira3d-reconstruction", ".venv")
_REGISTRATION_VENV = os.path.join(_GIRA_WS, "gira3d-registration", ".venv")

_inject_venv(_RECONSTRUCTION_VENV)
_inject_venv(_REGISTRATION_VENV)

try:
    from sogmm_py.sogmm import SOGMM
    HAS_SOGMM = True
except ImportError:
    HAS_SOGMM = False
    import logging as _logging
    _logging.warning(
        f"sogmm_py not found - SOGMM fitting disabled. "
        f"Expected venv at {_RECONSTRUCTION_VENV}"
    )

try:
    import gmm_d2d_registration_py
    from utils.save_gmm import save as save_gmm_official
    HAS_GMM_REGISTRATION = True
except ImportError:
    HAS_GMM_REGISTRATION = False
    save_gmm_official = None
    import logging as _logging
    _logging.warning(
        f"gmm_d2d_registration_py not found - D2D registration disabled. "
        f"Expected venv at {_REGISTRATION_VENV}"
    )



def project_gmm_4d_to_3d(gmm_4d):
    """Project a 4D SOGMM to 3D for registration.
    
    SOGMM produces 4D GMMs (x, y, z, range), but registration expects 3D (x, y, z).
    Creates a new sklearn GaussianMixture with only the first 3 dimensions.
    
    Parameters
    ----------
    gmm_4d : sklearn.mixture.GaussianMixture
        4D GMM from SOGMM with shape (N, 4)
    
    Returns
    -------
    gmm_3d : sklearn.mixture.GaussianMixture
        3D GMM with shape (N, 3)
    """
    from sklearn.mixture import GaussianMixture
    
    K = gmm_4d.n_components_
    gmm_3d = GaussianMixture(n_components=K, covariance_type='full')
    
    # Check 4D GMM for NaN/Inf before projection
    if np.any(np.isnan(gmm_4d.means_)):
        rospy.logerr(f"[project_gmm] Input 4D GMM has NaN means!")
        raise ValueError("Input GMM contains NaN values")
    if np.any(np.isnan(gmm_4d.covariances_)):
        rospy.logerr(f"[project_gmm] Input 4D GMM has NaN covariances!")
        raise ValueError("Input GMM contains NaN values")
    
    # Project means: (K, 4) -> (K, 3)
    gmm_3d.means_ = gmm_4d.means_[:, :3].copy()
    
    # Project covariances: (K, 4, 4) -> (K, 3, 3)
    # Handle both 3D array and flattened cases
    if gmm_4d.covariances_.ndim == 3:
        gmm_3d.covariances_ = gmm_4d.covariances_[:, :3, :3].copy()
    elif gmm_4d.covariances_.ndim == 2:
        # Already (K, D*D) flattened - reshape, slice, flatten again
        D_orig = gmm_4d.means_.shape[1]
        cov_3d = np.zeros((K, 3, 3))
        for k in range(K):
            cov_4d = gmm_4d.covariances_[k].reshape(D_orig, D_orig)
            cov_3d[k] = cov_4d[:3, :3]
        gmm_3d.covariances_ = cov_3d
    else:
        raise ValueError(f"Unexpected covariance shape: {gmm_4d.covariances_.shape}")

    
    # Copy weights
    gmm_3d.weights_ = gmm_4d.weights_.copy()
    
    # Set other required attributes for sklearn GaussianMixture
    gmm_3d.n_components_ = K  # This is set after fit() normally
    gmm_3d.converged_ = True
    gmm_3d.n_iter_ = 0
    gmm_3d.lower_bound_ = -np.inf  # Log likelihood lower bound (not critical for saving)
    
    # Compute precision matrices
    try:
        gmm_3d.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_3d.covariances_))
    except np.linalg.LinAlgError as e:
        rospy.logerr(f"[project_gmm] Failed to compute precision matrices: {e}")
        raise
    
    return gmm_3d


def plot_gmm_3d(gmm, pts: np.ndarray = None, sigma: float = 1.0,
                max_ellipsoids: int = 50, title: str = "GMM 3D Gaussians"):
    """Visualize a 3D GMM as ellipsoids using matplotlib (native X11 window)."""
    import matplotlib
    matplotlib.use('TkAgg')   # native window over X11
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if gmm.means_.shape[1] == 4:
        gmm = project_gmm_4d_to_3d(gmm)

    K = gmm.n_components_
    n_u, n_v = 20, 14
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0,     np.pi, n_v)
    sphere = np.stack([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones(n_u), np.cos(v)).ravel(),
    ], axis=1)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f"{title}  (K={K})")

    top_idx = np.argsort(gmm.weights_)[::-1][:max_ellipsoids]
    max_w   = gmm.weights_[top_idx[0]]

    for k in top_idx:
        mu  = gmm.means_[k]
        cov = gmm.covariances_[k]
        w   = gmm.weights_[k]
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        axes = eigvecs * (np.sqrt(eigvals) * sigma)
        epts = (axes @ sphere.T).T + mu
        ex = epts[:, 0].reshape(n_u, n_v)
        ey = epts[:, 1].reshape(n_u, n_v)
        ez = epts[:, 2].reshape(n_u, n_v)
        alpha = 0.15 + 0.45 * (w / max_w)
        ax.plot_surface(ex, ey, ez, color='steelblue', alpha=alpha, linewidth=0)

    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], gmm.means_[:, 2],
               c='red', s=40, zorder=5, label='Means')

    if pts is not None and len(pts) > 0:
        max_pts = 10_000
        dp = pts if len(pts) <= max_pts else pts[np.random.choice(len(pts), max_pts, replace=False)]
        ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], c='gray', s=1, alpha=0.3)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig




# ===========================================================================
# Helpers
# ===========================================================================

def make_gmm(means: np.ndarray, covariances: np.ndarray, weights: np.ndarray):
    """Build a fitted sklearn GaussianMixture directly from parameters (no EM)."""
    from sklearn.mixture import GaussianMixture
    means       = np.asarray(means,       dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    weights     = np.asarray(weights,     dtype=np.float64)
    weights    /= weights.sum()
    K = len(means)
    gmm = GaussianMixture(n_components=K, covariance_type='full')
    gmm.means_       = means
    gmm.covariances_ = covariances + 1e-6 * np.eye(3)[None]   # ensure PD
    gmm.weights_     = weights
    gmm.n_components_ = K
    gmm.converged_   = True
    gmm.n_iter_      = 0
    gmm.lower_bound_ = -np.inf
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    return gmm


def sample_from_gmm(gmm, n_samples: int) -> np.ndarray:
    """Draw n_samples points from a manually-built GaussianMixture."""
    counts = np.random.multinomial(n_samples, gmm.weights_)
    parts  = []
    for k, nk in enumerate(counts):
        if nk > 0:
            parts.append(
                np.random.multivariate_normal(gmm.means_[k], gmm.covariances_[k], nk)
            )
    return np.vstack(parts).astype(np.float32)


def save_gmm_to_file(gmm, filepath: str):
    """Save a 3D GaussianMixture to .gmm format (gira3d CSV)."""
    if not HAS_GMM_REGISTRATION or save_gmm_official is None:
        raise RuntimeError("save_gmm utility not available")
    if gmm.means_.shape[1] == 4:
        gmm = project_gmm_4d_to_3d(gmm)
    save_gmm_official(filepath, gmm)
    print(f"[save_gmm] Saved {gmm.n_components_} components → {os.path.basename(filepath)}")


def plot_two_gmms_3d(gmm_a, gmm_b,
                     pts_a: np.ndarray = None,
                     pts_b: np.ndarray = None,
                     sigma: float = 2.0,
                     max_ellipsoids: int = 50,
                     title: str = "GMM Registration — target (red) / source (blue)"):
    """Plot two 3D GMMs as ellipsoids in a native matplotlib window."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_u, n_v = 20, 14
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0,     np.pi, n_v)
    sphere = np.stack([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones(n_u), np.cos(v)).ravel(),
    ], axis=1)

    def _draw_gmm(ax, gmm, color, label, max_els):
        top_idx = np.argsort(gmm.weights_)[::-1][:max_els]
        max_w   = gmm.weights_[top_idx[0]]
        for i, k in enumerate(top_idx):
            mu  = gmm.means_[k]
            cov = gmm.covariances_[k]
            w   = gmm.weights_[k]
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            axes = eigvecs * (np.sqrt(eigvals) * sigma)
            epts = (axes @ sphere.T).T + mu
            ex = epts[:, 0].reshape(n_u, n_v)
            ey = epts[:, 1].reshape(n_u, n_v)
            ez = epts[:, 2].reshape(n_u, n_v)
            alpha = 0.12 + 0.38 * (w / max_w)
            ax.plot_surface(ex, ey, ez, color=color, alpha=alpha, linewidth=0)
        # Means (only label the first for the legend)
        ax.scatter(*gmm.means_.T, c=color, s=50, zorder=5,
                   label=label, depthshade=False)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    _draw_gmm(ax, gmm_a, 'tab:red',  'Target (A)', max_ellipsoids)
    _draw_gmm(ax, gmm_b, 'tab:blue', 'Source (B)', max_ellipsoids)

    for pts, color, name in [(pts_a, 'salmon', 'Cloud A'), (pts_b, 'royalblue', 'Cloud B')]:
        if pts is not None and len(pts) > 0:
            max_pts = 10_000
            dp = pts if len(pts) <= max_pts else pts[
                np.random.choice(len(pts), max_pts, replace=False)]
            ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2],
                       c=color, s=1, alpha=0.25, label=name)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


# ===========================================================================
# Test class
# ===========================================================================

class TestSimpleGMMReg:
    def __init__(self):

        # SOGMM parameters
        self.sogmm_bandwidth = rospy.get_param("~sogmm_bandwidth", 0.2)  # ??????
        self.sogmm_compute   = rospy.get_param("~sogmm_compute",   "CPU")  # "CPU" or "GPU"

        if not HAS_SOGMM:
            raise ImportError("sogmm_py not found - cannot run tests. "
                              f"Expected venv at {_RECONSTRUCTION_VENV}")
        else:
            rospy.loginfo(f"[gmmslam] SOGMM bandwidth : {self.sogmm_bandwidth}")
            rospy.loginfo(f"[gmmslam] SOGMM compute   : {self.sogmm_compute}")
    
        if not HAS_GMM_REGISTRATION:
            raise ImportError("gmm_d2d_registration_py not found - cannot run tests. "
                              f"Expected venv at {_REGISTRATION_VENV}")

        self.sg = SOGMM(bandwidth=self.sogmm_bandwidth, compute=self.sogmm_compute)
        self.local_gmms = []  # list[(stamp, GMMf4)]
        # D2D registration state
        self.prev_gmm_path = None   # path to previous frame's .gmm file
        self.gmm_dir = rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms")
        os.makedirs(self.gmm_dir, exist_ok=True)

    def _register_gmm(self, source_path: str, target_path: str) -> np.ndarray:
        """Register source GMM to target GMM using D2D registration.
        
        Performs two-stage registration:
          1. isoplanar_registration - fast alignment
          2. anisotropic_registration - refinement
        
        Parameters
        ----------
        source_path : str
            Path to current frame's .gmm file
        target_path : str
            Path to previous frame's .gmm file
        
        Returns
        -------
        T : (4,4) np.ndarray or None
            Transformation T_source_target (from target to source frame)
            Returns None if registration failed.
        """

        # Validate input files exist
        if not os.path.exists(source_path):
            rospy.logerr(f"[gmmslam] Source GMM file not found: {source_path}")
            return None
        if not os.path.exists(target_path):
            rospy.logerr(f"[gmmslam] Target GMM file not found: {target_path}")
            return None
        
        try:
            T_init = np.eye(4, dtype=np.float64)

            # Stage 1: isoplanar registration
            result_iso = gmm_d2d_registration_py.isoplanar_registration(
                T_init, source_path, target_path
            )
            T_iso = result_iso[0]
            score_iso = result_iso[1]
            
            # Check for NaN scores (indicates ill-conditioned matrices)
            if np.isnan(score_iso) or np.isinf(score_iso):
                rospy.logwarn(f"[gmmslam] D2D registration failed - ill-conditioned covariance matrices (iso score: nan)")
                return None
            
            # Check for NaN/Inf in transformation
            if np.any(np.isnan(T_iso)) or np.any(np.isinf(T_iso)):
                rospy.logwarn(f"[gmmslam] D2D registration failed - invalid transformation (ill-conditioned matrices)")
                return None
            
            # Stage 2: anisotropic refinement
            result_aniso = gmm_d2d_registration_py.anisotropic_registration(
                T_iso, source_path, target_path
            )
            T_final = result_aniso[0]
            score_final = result_aniso[1]
            
            # Check for NaN scores (indicates ill-conditioned matrices)
            if np.isnan(score_final) or np.isinf(score_final):
                rospy.logwarn(f"[gmmslam] D2D registration failed - ill-conditioned covariance matrices (aniso score: nan)")
                return None
            
            # Check for NaN/Inf in final transformation
            if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
                rospy.logwarn(f"[gmmslam] D2D registration failed - invalid transformation (ill-conditioned matrices)")
                return None
            
            rospy.loginfo(
                f"[gmmslam] D2D registration | iso score: {score_iso:.4f} | aniso score: {score_final:.4f}")
            
            # Log transformation for debugging
            rospy.logdebug(f"[gmmslam] T_final translation: [{T_final[0,3]:.3f}, {T_final[1,3]:.3f}, {T_final[2,3]:.3f}]")
            
            return T_final
        except Exception as e:
            rospy.logerr(f"[gmmslam] D2D registration failed cuz: {e}")
            import traceback; rospy.logerr(traceback.format_exc())
            return None
        


if __name__ == "__main__":
    rospy.init_node("test_simple_gmm_reg", anonymous=True)
    test = TestSimpleGMMReg()

    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Define two 3-D GMMs with a known rigid offset between them
    # ------------------------------------------------------------------
    means_a = np.array([
        [0.0,  0.0,  0.0],
        [1.2,  0.0,  0.3],
        [0.6,  1.0,  0.1],
        [0.2,  0.7, -0.4],
    ], dtype=np.float64)

    covs_a = np.array([
        np.diag([0.10, 0.08, 0.05]),
        np.diag([0.07, 0.12, 0.06]),
        np.diag([0.09, 0.07, 0.04]),
        np.diag([0.11, 0.09, 0.07]),
    ], dtype=np.float64)

    weights_a = np.array([0.35, 0.30, 0.20, 0.15])

    # GMM-B = GMM-A shifted by a known offset
    offset = np.array([0.30, 0.15, 0.05])   # ground-truth translation B→A
    means_b = means_a + offset

    gmm_a = make_gmm(means_a, covs_a,       weights_a)
    gmm_b = make_gmm(means_b, covs_a.copy(), weights_a.copy())

    print(f"[test] GMM-A — {gmm_a.n_components_} components")
    print(f"[test] GMM-B — {gmm_b.n_components_} components  (offset = {offset})")

    # ------------------------------------------------------------------
    # 2. Sample point clouds for visual overlay
    # ------------------------------------------------------------------
    pts_a = sample_from_gmm(gmm_a, 600)
    pts_b = sample_from_gmm(gmm_b, 600)

    # ------------------------------------------------------------------
    # 3. Plot both GMMs before registration
    # ------------------------------------------------------------------
    print("[test] Plotting GMMs before registration …")
    plot_two_gmms_3d(gmm_a, gmm_b, pts_a=pts_a, pts_b=pts_b, sigma=2.0,
                     title="Before registration — target (red) / source (blue)")

    # ------------------------------------------------------------------
    # 4. Save to .gmm files
    # ------------------------------------------------------------------
    gmm_dir = rospy.get_param("~gmm_dir", "/tmp/gmmslam_gmms")
    os.makedirs(gmm_dir, exist_ok=True)
    path_a = os.path.join(gmm_dir, "gmm_target.gmm")
    path_b = os.path.join(gmm_dir, "gmm_source.gmm")
    save_gmm_to_file(gmm_a, path_a)
    save_gmm_to_file(gmm_b, path_b)

    # ------------------------------------------------------------------
    # 5. D2D registration: align source (B) onto target (A)
    # ------------------------------------------------------------------
    print("[test] Running D2D registration (source B → target A) …")
    T_est = test._register_gmm(source_path=path_b, target_path=path_a)

    if T_est is not None:
        t_est = T_est[:3, 3]
        # T_est is T_source_target, i.e. maps A→B frame, so translation = -offset
        gt_translation = -offset
        err = np.linalg.norm(t_est - gt_translation)
        print(f"[test] Estimated translation : {np.round(t_est, 4)}")
        print(f"[test] Ground-truth          : {gt_translation}")
        print(f"[test] Translation error     : {err:.4f} m")
        print(f"[test] Full T_estimated:\n{np.round(T_est, 4)}")

        # Plot registered source on top of target
        means_b_aligned = (T_est[:3, :3] @ gmm_b.means_.T).T + T_est[:3, 3]
        gmm_b_aligned   = make_gmm(means_b_aligned, covs_a.copy(), weights_a.copy())
        pts_b_aligned   = (T_est[:3, :3] @ pts_b.T).T + T_est[:3, 3]

        print("[test] Plotting GMMs after registration …")
        plot_two_gmms_3d(gmm_a, gmm_b_aligned,
                         pts_a=pts_a, pts_b=pts_b_aligned,
                         sigma=2.0,
                         title=f"After registration (err={err:.4f} m) — target (red) / aligned source (blue)")
    else:
        print("[test] Registration returned None — check GMM validity")

    rospy.spin()