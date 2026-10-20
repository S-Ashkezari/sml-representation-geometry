"""
metrics.py
==========
Empirical measurements needed by the Experimental Protocol and "Evaluating
Synergistic Gain" sections of SML_final_audited11.tex.

Equation map
------------
Eq. (empirical-representation-diversity)
    D_rep = [2/(K(K-1))] * sum_{k<j} E_x[ || r_k(x)-r_j(x) ||_2^2 ]

Eq. (target-deviation)
    E_d0  = [2/(K(K-1))] * sum_{k<j} E_x[ (|| r_k(x)-r_j(x) ||_2^2 - d0^2)^2 ]

Eq. (ensemble):      p_ens = sum_k omega_k p_k(x)
Eq. (equilibrium-feasibility): d_star^2 <= 2K/(K-1)  (see sml.losses.gram_feasible)
"Experimental Evaluation": G_int = P_SML - P_Ind (matched interaction gain)
"""
from __future__ import annotations
import itertools

import torch
import torch.nn.functional as F

from .models import normalize_representation


@torch.no_grad()
def mean_pairwise_squared_distance(h_list):
    """
    D_rep (Eq. empirical-representation-diversity): mean pairwise squared
    distance between normalized representations, averaged over all
    K(K-1)/2 pairs and the batch -- the primary representation-diversity
    statistic, computed in exactly the geometry (l2-normalized, per-sample)
    used by L_rep and L_div. The manuscript's explicit prefactor
    2/(K(K-1)) is exactly the average-over-pairs computed below.
    """
    K = len(h_list)
    r_list = [normalize_representation(h) for h in h_list]
    vals = []
    for k, j in itertools.combinations(range(K), 2):
        vals.append((r_list[k] - r_list[j]).pow(2).sum(dim=-1).mean())
    return (sum(vals) / len(vals)).item()


@torch.no_grad()
def target_deviation(h_list, d0):
    """
    E_d0 (Eq. target-deviation): complementary statistic measuring how far
    the realized pairwise distances sit from the nominal target d0,
    averaged over all pairs and the batch:
        E_d0 = [2/(K(K-1))] * sum_{k<j} E_x[(||r_k-r_j||^2 - d0^2)^2]
    """
    K = len(h_list)
    r_list = [normalize_representation(h) for h in h_list]
    vals = []
    for k, j in itertools.combinations(range(K), 2):
        d2 = (r_list[k] - r_list[j]).pow(2).sum(dim=-1)
        vals.append(((d2 - d0 ** 2) ** 2).mean())
    return (sum(vals) / len(vals)).item()


@torch.no_grad()
def mean_pairwise_distance(h_list):
    """Mean of d_kj = ||r_k-r_j||_2 (not squared) over all pairs and the batch."""
    K = len(h_list)
    r_list = [normalize_representation(h) for h in h_list]
    vals = []
    for k, j in itertools.combinations(range(K), 2):
        vals.append((r_list[k] - r_list[j]).norm(dim=-1).mean().item())
    return sum(vals) / len(vals)


@torch.no_grad()
def batch_averaged_dispersion(h_list):
    """
    Secondary population-dispersion statistic: (1/K) sum_k || rbar_k - rbar ||^2
    where rbar_k is the batch-averaged normalized representation of model k.
    Complements D_rep (which is per-example) with a population-level summary.
    """
    K = len(h_list)
    r_list = [normalize_representation(h) for h in h_list]
    rbar_k = [r.mean(dim=0) for r in r_list]
    rbar = torch.stack(rbar_k, dim=0).mean(dim=0)
    disp = sum((rk - rbar).pow(2).sum() for rk in rbar_k) / K
    return disp.item()


@torch.no_grad()
def prediction_disagreement(z_list):
    """
    Prediction-level diversity, measured separately from representation
    diversity ("Representation diversity should be measured separately from
    prediction diversity"). Returns (mean pairwise 0/1 disagreement rate,
    mean pairwise symmetric KL).
    """
    K = len(z_list)
    preds = [z.argmax(dim=-1) for z in z_list]
    probs = [F.softmax(z, dim=-1) for z in z_list]
    disagree, skl = [], []
    for k, j in itertools.combinations(range(K), 2):
        disagree.append((preds[k] != preds[j]).float().mean().item())
        log_pk = torch.log(probs[k].clamp_min(1e-8))
        log_pj = torch.log(probs[j].clamp_min(1e-8))
        kl_kj = (probs[k] * (log_pk - log_pj)).sum(-1).mean()
        kl_jk = (probs[j] * (log_pj - log_pk)).sum(-1).mean()
        skl.append((0.5 * (kl_kj + kl_jk)).item())
    return sum(disagree) / len(disagree), sum(skl) / len(skl)


@torch.no_grad()
def ensemble_accuracy(z_list, y, weights=None):
    """Eq. (ensemble): p_ens = sum_k omega_k p_k(x); accuracy of the ensemble vote."""
    K = len(z_list)
    if weights is None:
        weights = [1.0 / K] * K
    probs = sum(w * F.softmax(z, dim=-1) for w, z in zip(weights, z_list))
    pred = probs.argmax(dim=-1)
    return (pred == y).float().mean().item()


def interaction_gain(p_sml: float, p_ind: float) -> float:
    """G_int = P_SML - P_Ind (matched inference-time ensembles)."""
    return p_sml - p_ind
