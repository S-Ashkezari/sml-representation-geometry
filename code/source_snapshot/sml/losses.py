"""
losses.py
=========
Implements every loss term in SML_final_audited11.tex Section III
("Synergistic Mutual Learning") and the practical<->theoretical parameter
mapping of Section IV ("Theoretical Analysis").

Equation map (SML_final_audited11)
-----------------------------------
Eq. (normalized-representation)
    r_k(x) = h_k(x) / ||h_k(x)||_2                       in S^{D-1}

Eq. (representation-cooperation)
    L_rep^(k) = (1/2) sum_{j!=k} a_kj || r_k - r_j ||_2^2,   a_kj = a_jk
    -- the 1/2 factor means sum_k L_rep^(k) = sum_{k<j} d_kj^2 exactly,
       with NO double counting (verified by direct expansion; see
       `representation_loss` docstring below).

Eq. (temperature) / (output-loss)
    p_k^(T) = softmax(z_k / T)
    L_out^(k) = sum_{j!=k} b_kj * KL( sg[p_j^(T)] || p_k^(T) )

Eq. (diversity-loss)
    L_div = (eta/4) * sum_{k<j} (d_kj^2 - d0^2)^2,   d_kj = ||r_k - r_j||_2,
    eta >= 0.

    NOTE ON A FIXED CODE/PAPER MISMATCH: `eta` is a genuine, independent
    parameter of the final manuscript (SML_final_audited11.tex). It appears
    explicitly in the Theorem ("Pairwise cooperation-diversity balance"),
    in the equilibrium d_star^2 = d0^2 - w/eta, and in the dimensionless
    ratio rho = eta*d0^2/w (Eq. rho:
    "rho = eta*d0^2/w = eta*d0^2*lambda_div/(2*lambda_rep)"). An earlier
    internal draft (SML_final_audited9.tex) had removed eta from L_div,
    folding it entirely into lambda_div; that simplification was reverted
    in the manuscript actually being implemented here (audited11), but a
    previous revision of this code base was never updated to match and
    silently hardcoded eta=1 throughout. This module restores eta as a
    first-class argument everywhere it affects the theory, matching
    audited11 exactly.

Eq. (total-objective)
    L_total = sum_k [ L_task^(k) + lambda_out*L_out^(k) + lambda_rep*L_rep^(k) ]
              + lambda_div * L_div

Eq. (effective-cooperation)   (uniform interaction, derived in Section IV):
dividing the pairwise part of L_total by lambda_div gives
V(d) = (w/2) d^2 + (eta/4)(d^2-d0^2)^2 with

    w = 2 * lambda_rep / lambda_div

(eta survives this division unchanged: eta multiplies L_div, which is
itself already scaled by lambda_div in L_total, so dividing by lambda_div
cancels the lambda_div factor but leaves eta in place -- exactly as
derived in the paper immediately before Eq. (effective-cooperation)).

Eq. (rho)
    rho = eta * d0^2 / w = eta * d0^2 * lambda_div / (2 * lambda_rep)

Eq. (effective-target) / Theorem (pairwise cooperation-diversity balance)
    d_star^2 = d0^2 - w/eta   (exists iff eta*d0^2 > w, i.e. rho > 1)
"""
from __future__ import annotations
import itertools

import torch
import torch.nn.functional as F

from .models import normalize_representation


# ----------------------------------------------------------------------
# Eq. (representation-cooperation): symmetric spherical cooperation
# ----------------------------------------------------------------------
def representation_loss(h_list, a_kj=1.0):
    """
    Returns L_rep^(k) for every k (list of length K), each computed exactly
    as Eq. (representation-cooperation):
        L_rep^(k) = (1/2) sum_{j!=k} a_kj || r_k - r_j ||_2^2
    where r_k = h_k / ||h_k|| (representations are normalized here, not by
    the caller, so this function is the single source of truth for the
    normalization used in the cooperation term).

    Verification that the 1/2 factor removes double counting: summing over
    k, sum_k L_rep^(k) = (1/2) sum_k sum_{j!=k} a_kj d_kj^2. With a_kj
    symmetric, sum_k sum_{j!=k} d_kj^2 counts every unordered pair {k,j}
    twice, so sum_k L_rep^(k) = (1/2)*2*sum_{k<j} d_kj^2 = sum_{k<j} d_kj^2,
    matching the manuscript's claim exactly (no residual factor).

    No stop-gradient is applied to r_j: both peers receive gradient from
    every pairwise term, matching "both representations are updated,
    making the interaction symmetric."
    """
    K = len(h_list)
    r_list = [normalize_representation(h) for h in h_list]
    losses = []
    for k in range(K):
        acc = h_list[k].new_zeros(())
        for j in range(K):
            if j == k:
                continue
            d2 = (r_list[k] - r_list[j]).pow(2).sum(dim=-1).mean()
            acc = acc + a_kj * d2
        losses.append(0.5 * acc)
    return losses


# ----------------------------------------------------------------------
# Eq. (temperature)-(output-loss): output-level mutual distillation
# ----------------------------------------------------------------------
def output_mutual_loss(z_list, T=4.0, b_kj=1.0):
    """
    Returns L_out^(k) for every k, exactly Eq. (output-loss):
        L_out^(k) = sum_{j!=k} b_kj * KL( sg[p_j^(T)] || p_k^(T) )
    Uses F.kl_div(log q, p) = KL(p || q) with p = sg[p_j^(T)] (detached
    teacher) and q = p_k^(T) (student), matching the direction written in
    the manuscript.
    """
    K = len(z_list)
    p_list = [F.softmax(z / T, dim=-1) for z in z_list]
    losses = []
    for k in range(K):
        log_pk = F.log_softmax(z_list[k] / T, dim=-1)
        acc = z_list[k].new_zeros(())
        for j in range(K):
            if j == k:
                continue
            p_j_sg = p_list[j].detach()
            kl = F.kl_div(log_pk, p_j_sg, reduction="batchmean")
            acc = acc + b_kj * kl
        losses.append(acc)
    return losses


# ----------------------------------------------------------------------
# Eq. (diversity-loss): finite-target diversity potential (evaluated once)
# ----------------------------------------------------------------------
def diversity_loss(h_list, d0, eta=1.0):
    """
    Eq. (diversity-loss): L_div = (eta/4) * sum_{k<j} (d_kj^2 - d0^2)^2,
    with d_kj = ||r_k - r_j||_2, r_k the l2-normalized representation, and
    eta >= 0 the diversity strength (restored as an explicit, independent
    parameter -- see module docstring for why this was previously wrong).

    Evaluated ONCE over the model population ("The diversity objective is
    evaluated once over each unordered pair. It is not separately
    duplicated in the loss of every model."), i.e. each unordered pair
    contributes exactly one term, not one term per model.
    """
    r_list = [normalize_representation(h) for h in h_list]
    K = len(r_list)
    total = h_list[0].new_zeros(())
    for k, j in itertools.combinations(range(K), 2):
        d2 = (r_list[k] - r_list[j]).pow(2).sum(dim=-1).mean()
        total = total + (eta / 4.0) * (d2 - d0 ** 2) ** 2
    return total


# ----------------------------------------------------------------------
# Eq. (total-objective): total training objective
# ----------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class SMLLossOutput:
    total: torch.Tensor
    task: list
    out: list
    rep: list
    div: torch.Tensor


def total_objective(h_list, z_list, y,
                     lambda_out=1.0, lambda_rep=1.0, lambda_div=1.0,
                     d0=1.0, eta=1.0, T=4.0,
                     use_out=True, use_rep=True, use_div=True):
    """
    Eq. (total-objective):
        L_total = sum_k [ L_task^(k) + lambda_out*L_out^(k) + lambda_rep*L_rep^(k) ]
                  + lambda_div * L_div
    where L_div = (eta/4) sum_{k<j} (d_kj^2-d0^2)^2 (Eq. diversity-loss).
    Toggle flags (use_out/use_rep/use_div) reproduce every ablation listed
    in the Experimental Evaluation section (DML-only, rep-only,
    cooperation-without-diversity, full SML, etc.).
    """
    K = len(h_list)
    task_losses = [F.cross_entropy(z_list[k], y) for k in range(K)]

    if use_out:
        out_losses = output_mutual_loss(z_list, T=T)
    else:
        out_losses = [z_list[0].new_zeros(()) for _ in range(K)]

    if use_rep:
        rep_losses = representation_loss(h_list)
    else:
        rep_losses = [h_list[0].new_zeros(()) for _ in range(K)]

    if use_div:
        div = diversity_loss(h_list, d0, eta=eta)
    else:
        div = h_list[0].new_zeros(())

    total = lambda_div * div
    for k in range(K):
        total = total + task_losses[k] + lambda_out * out_losses[k] + lambda_rep * rep_losses[k]

    return SMLLossOutput(total=total, task=task_losses, out=out_losses,
                          rep=rep_losses, div=div)


# ----------------------------------------------------------------------
# Practical <-> theoretical mapping (Section IV), matching audited11
# ----------------------------------------------------------------------
def coop_strength(lambda_rep: float, lambda_div: float) -> float:
    """
    Eq. (effective-cooperation), uniform interaction (a_kj=1):
        w = 2 * lambda_rep / lambda_div
    Derivation: dividing the pairwise part of L_total,
        lambda_rep * d^2 + lambda_div * (eta/4)(d^2-d0^2)^2   (K=2, one pair)
    by lambda_div gives
        (lambda_rep/lambda_div) d^2 + (eta/4)(d^2-d0^2)^2
    which matches V(d) = (w/2) d^2 + (eta/4)(d^2-d0^2)^2 iff w/2 = lambda_rep/lambda_div.
    Note eta cancels out of this particular ratio (it does not appear in w
    itself); it re-enters through rho() and d_star_squared() below.
    Requires lambda_div > 0.
    """
    if lambda_div <= 0:
        raise ValueError("lambda_div must be > 0 to define the reduced cooperation strength w.")
    return 2.0 * lambda_rep / lambda_div


def rho(d0: float, w: float, eta: float = 1.0) -> float:
    """Eq. (rho): rho = eta * d0^2 / w.

    Edge case w=0 (lambda_rep=0, i.e. no representation cooperation at
    all): the noncollapse condition eta*d0^2 > w holds trivially for any
    eta>0, so rho is +inf by convention (this also fixes a latent
    division-by-zero that existed in earlier revisions whenever an
    ablation swept lambda_rep down to 0, e.g. experiments/exp4_ablation.py).
    """
    if w == 0:
        return float("inf")
    return eta * d0 ** 2 / w


def d_star_squared(d0: float, w: float, eta: float = 1.0):
    """
    Eq. (effective-target) / Theorem (pairwise cooperation-diversity
    balance): d_star^2 = d0^2 - w/eta when eta*d0^2 > w (noncollapse
    condition), else None (no positive interior equilibrium; the collapsed
    state d=0 is the locally stable configuration instead).
    Requires eta > 0.

    Edge case w=0 (no cooperation): the equilibrium is exactly d0^2, since
    the diversity potential alone determines the stationary point.
    """
    if eta <= 0:
        raise ValueError("eta must be > 0 to define a finite-target diversity equilibrium.")
    if w == 0:
        return d0 ** 2
    val = d0 ** 2 - w / eta
    return val if val > 0 else None


def gram_feasible(d_star: float, K: int) -> bool:
    """
    Eq. (equilibrium-feasibility): a common pairwise equilibrium distance
    d_star is simultaneously realizable for all K(K-1)/2 pairs (as an exact
    equidistant spherical configuration) iff
        d_star^2 <= 2K / (K-1).
    """
    if K < 2:
        return True
    return d_star ** 2 <= (2.0 * K) / (K - 1)
