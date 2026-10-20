"""
reduced_dynamics.py
====================
Numerical validation of the reduced models in SML_final_audited11.tex,
Section IV ("Theoretical Analysis"). Two independent things are checked:

1. The PAIRWISE SPHERICAL model (Theorem, pairwise cooperation-diversity
   balance) via actual projected (Riemannian) gradient descent directly on
   S^{D-1} -- matching the manuscript's derivation on the sphere, not an
   unconstrained R^D flow. We check that gradient descent from generic
   initial conditions converges to d_star = sqrt(d0^2 - w/eta) when
   eta*d0^2 > w, and collapses to d=0 when eta*d0^2 < w -- exactly the two
   regimes claimed by the theorem.

2. The AUXILIARY population-level Euclidean cooperation-only model (kept
   deliberately separate, per the manuscript: "This exact exponential law
   ... is an auxiliary result and is not the dynamics of normalized SML
   representations"), which has the exact closed form
   E(t) = E(0) * exp(-2wKt).

Equation map
------------
Eq. (angular-gradient)   dV/dtheta = [w+eta*(d^2-d0^2)] sin(theta)
Eq. (angular-stability)  d^2V/dtheta^2 |_star = 2*eta*sin^2(theta_star) > 0
Eq. (effective-target)   d_star^2 = d0^2 - w/eta   (noncollapse: eta*d0^2 > w)
Eq. (equilibrium-feasibility)  d_star^2 <= 2K/(K-1)  (K-peer feasibility)
Eq. (euclidean-flow)         u_dot_k = -w sum_{j!=k}(u_k-u_j)
Eq. (euclidean-contraction)  E(t) = E(0) exp(-2wKt)

NOTE ON A FIXED CODE/PAPER MISMATCH: a previous revision of this file
hardcoded eta=1 everywhere and dropped it from every formula, based on an
earlier internal draft (SML_final_audited9.tex) that had folded eta into
lambda_div. The manuscript actually being implemented here
(SML_final_audited11.tex) reinstates eta as an independent parameter (see
the Theorem statement and Eq. rho = eta*d0^2/w), so this module restores
eta as an explicit argument throughout.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# 1. Pairwise spherical model (Theorem) -- projected gradient descent
# ----------------------------------------------------------------------
def _riemannian_grad(r_i, r_j, w, d0, eta):
    d2 = np.sum((r_i - r_j) ** 2)
    coeff = w + eta * (d2 - d0 ** 2)           # Eq. (angular-gradient) scalar
    euclid_grad_i = coeff * (r_i - r_j)
    euclid_grad_j = coeff * (r_j - r_i)
    grad_i = euclid_grad_i - (r_i @ euclid_grad_i) * r_i
    grad_j = euclid_grad_j - (r_j @ euclid_grad_j) * r_j
    return grad_i, grad_j


def simulate_spherical_pair(r_i0, r_j0, w, d0, eta=1.0, lr=0.05, steps=4000):
    """
    Riemannian (projected) gradient descent of V(r_i, r_j) directly on
    S^{D-1} x S^{D-1}. Returns the trajectory of the pairwise distance
    d(t) = ||r_i(t)-r_j(t)||.
    """
    r_i = r_i0 / np.linalg.norm(r_i0)
    r_j = r_j0 / np.linalg.norm(r_j0)
    d_hist = np.zeros(steps + 1)
    d_hist[0] = np.linalg.norm(r_i - r_j)
    for t in range(steps):
        g_i, g_j = _riemannian_grad(r_i, r_j, w, d0, eta)
        r_i = r_i - lr * g_i
        r_j = r_j - lr * g_j
        r_i = r_i / np.linalg.norm(r_i)
        r_j = r_j / np.linalg.norm(r_j)
        d_hist[t + 1] = np.linalg.norm(r_i - r_j)
    return d_hist


def theorem_equilibrium(w, d0, eta=1.0):
    """d_star = sqrt(d0^2 - w/eta) if eta*d0^2>w else None (Eq. effective-target)."""
    val = d0 ** 2 - w / eta
    return float(np.sqrt(val)) if val > 0 else None


# ----------------------------------------------------------------------
# 2. Auxiliary population-level Euclidean cooperation-only model
# ----------------------------------------------------------------------
def simulate_euclidean_population(u0, w, T=5.0, steps=5000):
    """
    Integrates Eq. (euclidean-flow):
        u_dot_k = -w * sum_{j!=k} (u_k - u_j)
    for K unconstrained points u_k in R^D (explicit Euler). Returns
    (t, E_numeric(t), E_closed_form(t)) matching Eq.
    (euclidean-contraction): E(t) = E(0) * exp(-2wKt).
    NOTE: this is the auxiliary Euclidean model only -- it does not involve
    eta or the diversity potential, and is not claimed to describe the
    normalized spherical dynamics of SML itself.
    """
    K, D = u0.shape
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    u = u0.copy()

    def dispersion(u):
        ubar = u.mean(axis=0)
        return np.mean(np.sum((u - ubar) ** 2, axis=1))

    E_numeric = np.zeros(steps + 1)
    E_numeric[0] = dispersion(u)
    for i in range(steps):
        ubar = u.mean(axis=0)
        du = -w * K * (u - ubar)   # sum_{j!=k}(u_k-u_j) = K(u_k-ubar)
        u = u + dt * du
        E_numeric[i + 1] = dispersion(u)

    E0 = E_numeric[0]
    E_closed = E0 * np.exp(-2 * w * K * t)
    return t, E_numeric, E_closed


# ----------------------------------------------------------------------
# 3. Population-level feasibility (Gram matrix PSD condition)
# ----------------------------------------------------------------------
def gram_feasible(d_star, K):
    """Eq. (equilibrium-feasibility): d_star^2 <= 2K/(K-1)."""
    if K < 2:
        return True
    return d_star ** 2 <= (2.0 * K) / (K - 1)


def gram_eigenvalues(c_star, K):
    """
    Eigenvalues of G_star = (1-c_star) I_K + c_star * 1 1^T, used to
    directly verify the PSD feasibility condition c_star >= -1/(K-1)
    (Eq. equilibrium-cosine / equilibrium-feasibility).
    Returns (eig_symmetric_direction, eig_orthogonal_complement, multiplicity).
    """
    eig_1 = 1 + c_star * (K - 1)     # along the all-ones direction
    eig_perp = 1 - c_star             # multiplicity K-1
    return eig_1, eig_perp, K - 1


# ----------------------------------------------------------------------
# Full validation report
# ----------------------------------------------------------------------
def validate_all(seed=0, D=8, verbose=True):
    rng = np.random.default_rng(seed)
    report = {}

    # --- Theorem, super-threshold regime with eta != 1: eta*d0^2 > w ---
    # Chosen so eta actually matters: w=0.6, d0=1.0, eta=2.0 gives
    # eta*d0^2=2.0 > w=0.6 (noncollapse), whereas eta=1.0 with the same
    # w, d0 would give eta*d0^2=1.0 > w=0.6 too -- so we also include a
    # case where eta changes which regime applies (see below) to make the
    # eta-dependence of the theorem an actual test, not just a passthrough.
    w, d0, eta = 0.6, 1.0, 2.0   # eta*d0^2 = 2.0 > w = 0.6
    d_star_theory = theorem_equilibrium(w, d0, eta)
    errs, finals = [], []
    for trial in range(5):
        r_i0 = rng.normal(size=D)
        r_j0 = rng.normal(size=D)
        d_hist = simulate_spherical_pair(r_i0, r_j0, w, d0, eta=eta, lr=0.05, steps=4000)
        errs.append(abs(d_hist[-1] - d_star_theory))
        finals.append(d_hist[-1])
    report["thm_superthreshold_eta"] = eta
    report["thm_superthreshold_d_star_theory"] = d_star_theory
    report["thm_superthreshold_d_final_mean"] = float(np.mean(finals))
    report["thm_superthreshold_max_abs_error"] = float(np.max(errs))

    # --- Theorem, sub-threshold regime purely because eta is SMALL ---
    # Same w, d0 as above (w=0.6, d0=1.0) but eta=0.5 flips the regime:
    # eta*d0^2 = 0.5 < w = 0.6, so collapse is now predicted. This
    # isolates eta's own effect on the transition, holding w and d0 fixed.
    w2, d02, eta2 = 0.6, 1.0, 0.5   # eta*d0^2 = 0.5 < w = 0.6
    d_star_theory_sub = theorem_equilibrium(w2, d02, eta2)  # None expected
    finals_sub = []
    for trial in range(5):
        r_i0 = rng.normal(size=D)
        r_j0 = rng.normal(size=D)
        d_hist = simulate_spherical_pair(r_i0, r_j0, w2, d02, eta=eta2, lr=0.05, steps=4000)
        finals_sub.append(d_hist[-1])
    report["thm_subthreshold_eta"] = eta2
    report["thm_subthreshold_d_star_theory"] = d_star_theory_sub  # should be None
    report["thm_subthreshold_d_final_mean"] = float(np.mean(finals_sub))  # should be near 0

    # --- Population-level feasibility (Gram matrix) ---
    K = 5
    c_star = -1.0 / (K - 1)          # boundary case: exactly feasible (simplex)
    eig1, eig_perp, mult = gram_eigenvalues(c_star, K)
    report["gram_boundary_eig_along_ones"] = eig1     # should be ~0
    report["gram_boundary_eig_orthogonal"] = eig_perp  # should be > 0
    c_star_infeasible = -1.0 / (K - 1) - 0.1
    eig1_bad, _, _ = gram_eigenvalues(c_star_infeasible, K)
    report["gram_infeasible_eig_along_ones"] = eig1_bad  # should be < 0

    # --- Auxiliary Euclidean population model (Eq. euclidean-contraction) ---
    K, Dpop = 6, 16
    u0 = rng.normal(size=(K, Dpop))
    w = 0.4
    t, E_num, E_closed = simulate_euclidean_population(u0, w, T=3.0, steps=6000)
    report["euclidean_pop_max_abs_error"] = float(np.max(np.abs(E_num - E_closed)))
    report["euclidean_pop_final_dispersion"] = float(E_num[-1])

    if verbose:
        print("=== Reduced-dynamics validation (SML_final_audited11, Section IV) ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    validate_all()
