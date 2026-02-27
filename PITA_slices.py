# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 20:17:49 2026

@author: Usama
"""



"""
PITA: Piecewise Interpolation and Tensor Approximation for Hessian Reconstruction

Reconstructs full Hessian tensors along an intrinsic reaction coordinate (IRC)
from a sparse set of sampled Hessian matrices using tensor decomposition,
piecewise polynomial fitting, and conjugate-gradient refinement.

Generalised to work with an arbitrary number of sampled slices.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorly as tl
from scipy.io import savemat

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
#  Eigenvalue extraction
# ---------------------------------------------------------------------------

def get_eigenvalue_matrix(df, tag="All"):
    """
    Extract eigenvalue rows and energies from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Eigenvalues', 'Energy', and optionally 'tag'.
    tag : str
        Filter: 'All', 'R', 'P', 'T', 'Rhalf', or 'Phalf'.

    Returns
    -------
    evals : list of lists
    energies : list
    """
    evals, energies = [], []
    for index, row in df.iterrows():
        include = False
        if tag == "All":
            include = True
        elif tag in ("R", "P", "T"):
            include = row.get("tag") == tag
        elif tag == "Rhalf":
            include = index <= 0
        elif tag == "Phalf":
            include = index >= 0
        if include:
            evals.append(list(row.Eigenvalues))
            energies.append(row.Energy)
    return evals, energies


# ---------------------------------------------------------------------------
#  Tensor construction helpers
# ---------------------------------------------------------------------------

def build_hessian_tensor(df, n_points):
    """
    Stack Hessians from *df* into a 3-D tensor of shape (m, n_points, m).

    Parameters
    ----------
    df : pd.DataFrame  – must contain a 'Hessian' column.
    n_points : int      – number of IRC points to stack.

    Returns
    -------
    tensor : ndarray of shape (m, n_points, m)
    """
    m = df["Hessian"].iloc[0].shape[0]
    slices = [df["Hessian"].iloc[i] for i in range(n_points)]
    stacked = np.hstack(slices)
    return tl.tensor(stacked.reshape(m, n_points, m))


def build_sampled_tensor(df, sample_index):
    """
    Build a tensor from only the sampled slice indices.

    Returns
    -------
    tensor : ndarray of shape (m, n_samples, m)
    """
    m = df["Hessian"].iloc[0].shape[0]
    slices = [df["Hessian"].iloc[i] for i in sample_index]
    stacked = np.hstack(slices)
    return tl.tensor(stacked.reshape(m, len(sample_index), m))


# ---------------------------------------------------------------------------
#  Interpolation (linear, weighted by s)
# ---------------------------------------------------------------------------

def interpolation_s(sample_index, tensor, s_values):
    """
    Linear interpolation along mode-1, weighted by the reaction coordinate *s*.

    Parameters
    ----------
    sample_index : list[int]   – sorted indices of known slices.
    tensor       : ndarray     – full (m, n_points, m) tensor.
    s_values     : array-like  – reaction-coordinate values for every point.

    Returns
    -------
    interp_tensor : ndarray of same shape as *tensor*.
    """
    s = np.asarray(s_values, dtype=float) - np.min(s_values)
    s_sampled = [s[i] for i in sample_index]

    slices = [tensor[:, 0, :]]  # start with the first slice

    for j in range(len(sample_index) - 1):
        lo, hi = sample_index[j], sample_index[j + 1]
        s_lo, s_hi = s_sampled[j], s_sampled[j + 1]
        # first segment: skip index lo (already appended); others: include lo
        start = 1 if j == 0 else 0
        for k in range(start, hi - lo):
            frac = (s[lo + k] - s_lo) / (s_hi - s_lo)
            interp_slice = tensor[:, lo, :] + frac * (tensor[:, hi, :] - tensor[:, lo, :])
            slices.append(interp_slice)

    slices.append(tensor[:, sample_index[-1], :])  # last known slice

    stacked = np.hstack(slices)
    interp_tensor = tl.tensor(stacked.reshape(tensor.shape))

    rel_err = np.linalg.norm(tensor - interp_tensor) / np.linalg.norm(tensor)
    print(f"Interpolation error: {rel_err:.6e}")
    return interp_tensor


# ---------------------------------------------------------------------------
#  Piecewise polynomial approximation (quadratic, 3-point windows)
# ---------------------------------------------------------------------------

def _vandermonde(degree, n_cols, x):
    """
    Build a Vandermonde-like matrix S of shape (degree+1, n_cols).
    S[i, j] = x[j]^(degree - i).
    """
    S = np.zeros((degree + 1, n_cols))
    for i in range(degree + 1):
        for j in range(n_cols):
            S[i, j] = x[j] ** (degree - i)
    return S


def polynomial_approx(tensor, triplet, x_values):
    """
    Fit a quadratic through three sampled slices of *tensor*.

    Parameters
    ----------
    tensor  : ndarray (m, n_points, m) – full tensor.
    triplet : list[int] of length 3   – indices of the three samples.
    x_values: list/array               – x-axis for polynomial evaluation.

    Returns
    -------
    estimated : ndarray (m, span, m) where span = triplet[-1] - triplet[0] + 1.
    """
    assert len(triplet) == 3
    m = tensor[:, 0, :].shape[0]
    n_samples = 3
    span = triplet[-1] - triplet[0] + 1

    # Build sampled unfolding
    sampled = np.hstack([tensor[:, idx, :] for idx in triplet])
    sampled_tensor = tl.tensor(sampled.reshape(m, n_samples, m))
    unfold_C = tl.unfold(sampled_tensor, mode=1)

    # Sampling operator
    sampling = np.zeros((n_samples, span))
    for k, idx in enumerate(triplet):
        sampling[k, idx - triplet[0]] = 1.0

    S = _vandermonde(n_samples, span, np.array(x_values))
    L = np.linalg.pinv(sampling @ S.T) @ unfold_C
    estimated = tl.fold(S.T @ L, mode=1, shape=(m, span, m))
    return estimated


def piecewise_polynomial(tensor, sample_index, x_values):
    """
    Build a full-tensor estimate by stitching overlapping quadratic segments
    fitted to consecutive triplets of sampled slices.

    Works with *any* number of samples >= 3.

    Parameters
    ----------
    tensor       : ndarray (m, n_points, m)
    sample_index : list[int] – sorted sample indices (length >= 3).
    x_values     : list/array – polynomial x-axis values (length = n_points).

    Returns
    -------
    estimated : ndarray of same shape as *tensor*.
    """
    n_samples = len(sample_index)
    assert n_samples >= 3, "Need at least 3 sampled slices for quadratic fitting."

    # Fit overlapping triplets: (0,1,2), (1,2,3), ..., (k-3,k-2,k-1)
    segments = []
    for i in range(n_samples - 2):
        triplet = [sample_index[i], sample_index[i + 1], sample_index[i + 2]]
        seg = polynomial_approx(tensor, triplet, x_values)
        segments.append(seg)

    # Stitch: first segment covers [si[0], si[2]], last covers [si[-3], si[-1]]
    estimated = np.zeros_like(tensor, dtype=float)
    # First segment
    estimated[:, sample_index[0]: sample_index[2] + 1, :] = segments[0]
    # Last segment (overwrites overlapping region – last segment wins)
    estimated[:, sample_index[-3]:, :] = segments[-1]
    # Middle segments
    for i in range(1, len(segments) - 1):
        lo = sample_index[i]
        hi = sample_index[i + 2] + 1
        estimated[:, lo:hi, :] = segments[i]

    rel_err = np.linalg.norm(tensor - estimated) / np.linalg.norm(tensor)
    print(f"Polynomial tensor error: {rel_err:.6e}")
    return estimated


# ---------------------------------------------------------------------------
#  Conjugate-gradient refinement
# ---------------------------------------------------------------------------

def compute_conjugate(
    unfold_C,
    unfold_QS,
    unfold_M,
    tensor_H,
    lam,
    tensor_shape,
    sample_index,
    tol=1e-6,
    maxit=500,
    reg=0.0,
):
    """
    Solve  X · C ≈ (1-λ) QS + λ M  via preconditioned CG (row-by-row),
    then fold, symmetrise, and pin known slices.

    Parameters
    ----------
    unfold_C     : ndarray – mode-1 unfolding of the sampled tensor.
    unfold_QS    : ndarray – mode-1 unfolding of the polynomial estimate.
    unfold_M     : ndarray – mode-1 unfolding of the interpolation estimate.
    tensor_H     : ndarray – ground-truth tensor (for pinning & error).
    lam          : float   – blending parameter in [0, 1].
    tensor_shape : tuple   – (m, n_points, m).
    sample_index : list[int] – indices of known slices.
    tol, maxit, reg : CG parameters.

    Returns
    -------
    X1      : ndarray – solution matrix.
    recon   : float   – reconstruction NMSE.
    est_H   : ndarray – reconstructed, symmetrised, pinned tensor.
    """
    A = unfold_C
    B = (1.0 - lam) * unfold_QS + lam * unfold_M
    k, n = A.shape
    m_rows = B.shape[0]

    C = B @ A.T

    diag_P = (A * A).sum(axis=1) + reg
    inv_diag_P = 1.0 / diag_P

    def Mv(v):
        return A @ (A.T @ v) + reg * v

    def cg_solve(c):
        y = np.zeros_like(c)
        r = c - Mv(y)
        z = inv_diag_P * r
        p = z.copy()
        rz = np.dot(r, z)
        rhs_norm = np.linalg.norm(c)
        if rhs_norm == 0:
            return y
        for _ in range(maxit):
            q = Mv(p)
            alpha = rz / np.dot(p, q)
            y += alpha * p
            r -= alpha * q
            if np.linalg.norm(r) <= tol * rhs_norm:
                break
            z_new = inv_diag_P * r
            rz_new = np.dot(r, z_new)
            beta = rz_new / rz
            p = z_new + beta * p
            rz = rz_new
        return y

    X1 = np.zeros((m_rows, k))
    for i in range(m_rows):
        X1[i, :] = cg_solve(C[i, :])

    # Fold, symmetrise, and pin known slices
    est_H_tmp = tl.fold(X1 @ unfold_C, mode=1, shape=tensor_shape)
    est_H = 0.5 * (np.transpose(est_H_tmp, (2, 1, 0)) + est_H_tmp)
    for idx in sample_index:
        est_H[:, idx, :] = tensor_H[:, idx, :]

    recon = np.linalg.norm(tensor_H - est_H) / np.linalg.norm(tensor_H)
    print(f"CG reconstruction NMSE: {recon:.6e}")
    return X1, recon, est_H


# ---------------------------------------------------------------------------
#  Post-processing: projection & eigendecomposition
# ---------------------------------------------------------------------------

def project_and_diagonalise(est_H, df, n_points, sample_index, n_drop=7):
    """
    For each IRC point:
      1. Project reconstructed Hessian perpendicular to the gradient.
      2. Diagonalise and sort eigenvalues.
      3. For sampled points, use ground-truth eigenvalues/vectors.
      4. For reconstructed points, drop the first *n_drop* modes.

    Parameters
    ----------
    est_H        : ndarray (m, n_points, m) – reconstructed Hessian tensor.
    df           : pd.DataFrame             – original data with Gradient, Hessian, etc.
    n_points     : int                      – number of IRC points.
    sample_index : list[int]                – indices of known slices.
    n_drop       : int                      – number of trivial modes to discard.

    Returns
    -------
    eig_vals_clean : list of arrays
    eig_vecs_clean : list of arrays
    hess_list      : list of (m, m) arrays
    """
    eig_vals_clean = []
    eig_vecs_clean = []
    hess_list = []

    for n in range(n_points):
        H = est_H[:, n, :]
        hess_list.append(H)

        if n in sample_index:
            # Use ground-truth for sampled points
            eig_vals_clean.append(np.asarray(df["Eigenvalues"].iloc[n]))
            eig_vecs_clean.append(np.asarray(df["Eigenvectors"].iloc[n]))
            continue

        # Project onto space perpendicular to gradient
        grad = df["Gradient"].iloc[n]
        P = np.eye(len(H)) - np.outer(grad, grad)
        H_proj = P.T @ H @ P
        H_proj = 0.5 * (H_proj + H_proj.T)

        evals, evecs = np.linalg.eigh(H_proj)  # eigh → real, sorted
        # Drop the first n_drop trivial modes
        eig_vals_clean.append(evals[n_drop:])
        eig_vecs_clean.append(evecs[:, n_drop:].T)

    return eig_vals_clean, eig_vecs_clean, hess_list


# ---------------------------------------------------------------------------
#  Sample-index selection
# ---------------------------------------------------------------------------

def select_sample_indices(s_values, n_samples=5):
    """
    Automatically choose *n_samples* indices along the reaction coordinate.

    Strategy:
        - Always include first (0) and last (n-1) points.
        - Always include the transition state (s ≈ 0).
        - Distribute remaining samples evenly between endpoints and TS.

    Parameters
    ----------
    s_values  : array-like – reaction-coordinate values.
    n_samples : int        – desired number of samples (>= 3).

    Returns
    -------
    sample_index : sorted list[int] of length n_samples.
    """
    s = np.asarray(s_values)
    n_points = len(s)
    ts_idx = int(np.where(s == 0)[0][0]) if 0 in s else int(np.argmin(np.abs(s)))

    # Mandatory anchors
    anchors = {0, ts_idx, n_points - 1}
    remaining = n_samples - len(anchors)
    assert remaining >= 0, f"n_samples={n_samples} must be >= {len(anchors)} (anchors)."

    # Split remaining samples between left (0→TS) and right (TS→end) segments
    n_left = remaining // 2
    n_right = remaining - n_left

    def pick_between(s_arr, lo_idx, hi_idx, n_picks):
        """Pick n_picks indices evenly spaced (in s) between lo_idx and hi_idx."""
        if n_picks == 0:
            return []
        s_lo, s_hi = s_arr[lo_idx], s_arr[hi_idx]
        targets = np.linspace(s_lo, s_hi, n_picks + 2)[1:-1]  # exclude endpoints
        indices = []
        for t in targets:
            idx = int(np.argmin(np.abs(s_arr - t)))
            # Avoid duplicates with anchors
            while idx in anchors or idx in indices:
                idx += 1
                if idx >= n_points:
                    idx = int(np.argmin(np.abs(s_arr - t))) - 1
            indices.append(idx)
        return indices

    left_picks = pick_between(s, 0, ts_idx, n_left)
    right_picks = pick_between(s, ts_idx, n_points - 1, n_right)

    sample_index = sorted(anchors | set(left_picks) | set(right_picks))
    return sample_index


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(matrix, s, title, filename=None):
    """Plot reconstructed (or true) eigenvalue profiles along the IRC."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i in range(matrix.shape[0]):
        ax.plot(s, matrix[i, :], linestyle="solid", lw=1.2, marker="o", markersize=3)
    ax.set_xlabel("s", fontsize=20)
    ax.set_ylabel(r"$\omega_i^2$", fontsize=18)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    return fig


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------

def reconstruct_system(
    system_name,
    base_dir="..",
    n_samples=5,
    n_drop=7,
    lam=0.75,
    eta=0.001,
    cg_tol=1e-50,
    cg_maxit=2000,
    cg_reg=0.0,
):
    """
    Full PITA reconstruction pipeline for one chemical system.

    Parameters
    ----------
    system_name : str   – directory / system identifier.
    base_dir    : str   – parent directory containing system folders.
    n_samples   : int   – number of Hessian slices to sample (>= 3).
    n_drop      : int   – number of trivial eigenvalue modes to discard.
    lam         : float – blending weight between polynomial (0) and interpolation (1).
    eta         : float – gradient-descent step size (post-CG refinement).
    cg_tol, cg_maxit, cg_reg : CG solver parameters.
    """
    # ---- Load ----
    pkl_path = os.path.join(base_dir, system_name, "MatPKL", f"{system_name}.pkl")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)
    s_values = np.array(df["s"])
    n_points = len(df)

    # ---- Sample selection ----
    sample_index = select_sample_indices(s_values, n_samples)
    print(f"System: {system_name}")
    print(f"Sample indices: {sample_index}  (n={len(sample_index)})")
    print(f"Sampled s-values: {s_values[sample_index]}")

    # ---- Build tensors ----
    tensor_H = build_hessian_tensor(df, n_points)
    sampled_tensor = build_sampled_tensor(df, sample_index)
    m = tensor_H.shape[0]

    # ---- Polynomial x-axis (normalised) ----
    x_values = [1 + 0.01 * i for i in range(1, n_points + 1)]

    # ---- Piecewise polynomial estimate ----
    poly_estimate = piecewise_polynomial(tensor_H, sample_index, x_values)

    # ---- Interpolation estimate ----
    interp_estimate = interpolation_s(sample_index, tensor_H, s_values)

    # ---- Mode-1 unfoldings ----
    unfold_QS = tl.unfold(poly_estimate, mode=1)
    unfold_M = tl.unfold(interp_estimate, mode=1)
    unfold_C = tl.unfold(sampled_tensor, mode=1)

    # ---- Initial X via pseudoinverse ----
    X1 = unfold_QS @ np.linalg.pinv(unfold_C)

    # ---- Gradient (diagnostic) ----
    QS_residual = (X1 @ unfold_C) - unfold_QS
    M_residual = (X1 @ unfold_C) - unfold_M
    grad = (1 - lam) * (QS_residual @ unfold_C.T) + lam * (M_residual @ unfold_C.T)
    print(f"Initial gradient norm: {np.linalg.norm(grad):.6e}")

    # ---- Conjugate-gradient solve ----
    tensor_shape = (m, n_points, m)
    X1, recon_cg, est_H = compute_conjugate(
        unfold_C, unfold_QS, unfold_M, tensor_H, lam,
        tensor_shape, sample_index,
        tol=cg_tol, maxit=cg_maxit, reg=cg_reg,
    )

    # ---- Optional gradient-descent step ----
    #X1 = X1 - eta * grad
    est_H_tmp = tl.fold(X1 @ unfold_C, mode=1, shape=tensor_shape)
    est_H = 0.5 * (np.transpose(est_H_tmp, (2, 1, 0)) + est_H_tmp)
    for idx in sample_index:
        est_H[:, idx, :] = tensor_H[:, idx, :]

    recon_final = np.linalg.norm(tensor_H - est_H) / np.linalg.norm(tensor_H)
    print(f"Final reconstruction NMSE: {recon_final:.6e}")

    # ---- Eigendecomposition ----
    eig_vals_clean, eig_vecs_clean, hess_list = project_and_diagonalise(
        est_H, df, n_points, sample_index, n_drop=n_drop,
    )

    # ---- Eigenvalue matrix for plotting / saving ----
    matrix = np.column_stack(eig_vals_clean)
    s = np.array(df["s"])

    Xtrue, _ = get_eigenvalue_matrix(df, "All")
    Xtrue = np.array(Xtrue).T

    overall_err = np.linalg.norm(matrix - Xtrue, "fro") / np.linalg.norm(Xtrue, "fro")
    print(f"Eigenvalue reconstruction RMSE: {overall_err:.6e}")

    # ---- Save results ----
    out_df = pd.DataFrame({
        "Distance": df["Eigenvalues"].index,
        "Eigenvectors": eig_vecs_clean,
        "Eigenvalues": eig_vals_clean,
        "Hessian": hess_list,
    }).set_index("Distance")

    pkl_out = os.path.join(base_dir, system_name, f"{system_name}_Hesscg_indexbased.pkl")
    out_df.to_pickle(pkl_out)

    mat_out = os.path.join(base_dir, system_name, f"result_{system_name}_Hesscg_indexbased.mat")
    savemat(mat_out, {
        "Xvmc2": matrix,
        "s": s,
        "I": [x + 1 for x in sample_index],  # 1-indexed for MATLAB
        "Xtrue": Xtrue,
    })

    # ---- Plots ----
    plot_eigenvalue_profiles(matrix, s, f"{system_name} PITA (n_samples={len(sample_index)})")
    plot_eigenvalue_profiles(Xtrue, s, f"{system_name} DFT (ground truth)")
    plt.show()

    return {
        "est_H": est_H,
        "matrix": matrix,
        "Xtrue": Xtrue,
        "sample_index": sample_index,
        "recon_nmse": recon_final,
        "eigenvalue_rmse": overall_err,
    }


# ---------------------------------------------------------------------------
#  Entry point — edit the config dict below to change parameters
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # =====================  CONFIGURATION  =====================
    config = {
        "base_dir":   "",       # Parent directory containing system folders
        "systems":    ["Rad"],       # List of system names, e.g. ["Rad", "Oxo3"]
                                  #   None → auto-detect folders with 'Rad' or 'Oxo'
        "n_samples":  12,          # Number of Hessian slices to sample (>= 3)
        "n_drop":     7,          # Trivial eigenvalue modes to discard
        "lam":        0.75,       # Blending weight: 0 = polynomial, 1 = interpolation
        "eta":        0.001,      # Gradient-descent step size (post-CG)
        "cg_tol":     1e-50,      # Conjugate-gradient tolerance
        "cg_maxit":   2000,       # Conjugate-gradient max iterations
        "cg_reg":     0.0,        # Conjugate-gradient regularisation
    }
    # ===========================================================

    # Auto-detect systems if none specified
    if config["systems"] is None:
        all_dirs = os.listdir(config["base_dir"])
        systems = sorted(d for d in all_dirs if "Rad" in d or "Oxo" in d)
    else:
        systems = config["systems"]

    print(f"Processing {len(systems)} system(s): {systems}\n")

    results = {}
    for system_name in systems:
        print("=" * 60)
        res = reconstruct_system(
            system_name,
            base_dir=config["base_dir"],
            n_samples=config["n_samples"],
            n_drop=config["n_drop"],
            lam=config["lam"],
            eta=config["eta"],
            cg_tol=config["cg_tol"],
            cg_maxit=config["cg_maxit"],
            cg_reg=config["cg_reg"],
        )
        results[system_name] = res
        print()

    print("Done.")