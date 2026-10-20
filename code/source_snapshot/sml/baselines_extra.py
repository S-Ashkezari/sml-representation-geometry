"""
baselines_extra.py
==================

Comparison baselines for Experiment 3.

WML
---
Adapted from the authors' official NeurIPS-2022 code supplied with the
project. The four-head topology, weighted mutual-learning objective, and
multiplicative hypergradient weight update follow train_4head.py.

The original release performs architecture-dependent structured pruning.
Because Experiment 3 uses the project's fixed CIFAR-10 training pipeline,
pruning is exposed as optional channel masking rather than silently
replacing the SML backbone. The paper must describe this as an adapted WML
implementation, not as an unmodified reproduction.

Diff-DML
--------
Mechanism-level reimplementation of the published DTS/DPLO design.
No public official implementation was available.

NCL
---
NCL-inspired representation repulsion. This is not a literal reproduction
of the original 1999 NCL formulation.

The NCL-inspired baseline used here promotes pairwise disagreement between
normalized representations. Because normalized representations lie on the
unit sphere, the pairwise squared distance is geometrically bounded. The
method therefore has no finite target distance and no saturation term.
"""

from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Common helpers
# ===========================================================================

def kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 4.0,
) -> torch.Tensor:
    """
    Temperature-scaled KL(student || teacher).

    The teacher distribution is detached, so gradients flow only through
    the student logits.

    Parameters
    ----------
    student_logits:
        Student logits, shape [B, C].
    teacher_logits:
        Teacher logits, shape [B, C].
    T:
        Distillation temperature.

    Returns
    -------
    torch.Tensor
        Scalar temperature-scaled KL divergence.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    log_p = F.log_softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits.detach() / T, dim=-1)

    return F.kl_div(
        log_p,
        q,
        reduction="batchmean",
    ) * (T * T)


def pairwise_symmetric_kl(
    z_list: Sequence[torch.Tensor],
    T: float = 1.0,
) -> torch.Tensor:
    """
    Mean pairwise symmetric KL between output distributions.

    This function is used for evaluation/diagnostics and does not detach
    either side because it is not intended as a training objective.
    """
    if len(z_list) == 0:
        raise ValueError("z_list must contain at least one tensor.")

    if len(z_list) < 2:
        return z_list[0].new_zeros(())

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    ps = [F.softmax(z / T, dim=-1) for z in z_list]

    total = z_list[0].new_zeros(())
    n = 0

    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            pi = ps[i].clamp_min(1e-8)
            pj = ps[j].clamp_min(1e-8)

            kl_ij = (
                pi * (pi.log() - pj.log())
            ).sum(-1).mean()

            kl_ji = (
                pj * (pj.log() - pi.log())
            ).sum(-1).mean()

            total = total + 0.5 * (kl_ij + kl_ji)
            n += 1

    return total / float(n)


# ===========================================================================
# WML -- official-mechanism adapter
# ===========================================================================

def wml_task_and_peer_losses(
    z_list: Sequence[torch.Tensor],
    y: torch.Tensor,
    weights: torch.Tensor,
    T: float = 3.0,
    alpha: float = 0.1,
):
    """
    Four-head WML objective used by the supplied official train_4head.py.

        L =
            (1 - alpha) * sum_k w_k CE_k
            +
            alpha * sum_k w_k sum_{j != k} KL(z_k || z_j) * T^2

    In the authors' implementation the peer distribution is detached.

    Parameters
    ----------
    z_list:
        Four head logits.
    y:
        Ground-truth labels.
    weights:
        Four non-negative head weights.
    T:
        Distillation temperature.
    alpha:
        Relative peer-distillation weight.

    Returns
    -------
    total:
        Scalar WML objective.
    ce:
        List of per-head cross-entropy losses.
    kd:
        K x K matrix of pairwise KD terms.
    """
    K = len(z_list)

    if K != 4:
        raise AssertionError(
            "WML baseline is defined for the official four-head setup."
        )

    if weights.numel() != 4:
        raise ValueError(
            f"WML requires four weights, got shape {tuple(weights.shape)}."
        )

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    ce = [
        F.cross_entropy(z, y)
        for z in z_list
    ]

    kd = [[None] * K for _ in range(K)]

    for k in range(K):
        for j in range(K):
            if j != k:
                kd[k][j] = kd_kl(
                    z_list[k],
                    z_list[j],
                    T=T,
                )

    total = z_list[0].new_zeros(())

    for k in range(K):
        peer = sum(
            kd[k][j]
            for j in range(K)
            if j != k
        )

        total = total + (
            (1.0 - alpha)
            * weights[k]
            * ce[k]
        )

        total = total + (
            alpha
            * weights[k]
            * peer
        )

    return total, ce, kd


def _flat_grad_dot(
    g1: Sequence[torch.Tensor],
    g2: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Inner product between two lists of parameter gradients.
    """
    if len(g1) != len(g2):
        raise ValueError(
            f"Gradient lists have different lengths: "
            f"{len(g1)} vs {len(g2)}."
        )

    total = g1[0].new_zeros(())

    for a, b in zip(g1, g2):
        total = total + (a * b).sum()

    return total


def _flat_grad_norm(
    g: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Euclidean norm of a list of parameter gradients.

    Kept for backward compatibility with earlier versions of the baseline.
    """
    if len(g) == 0:
        return torch.tensor(0.0)

    total = g[0].new_zeros(())

    for a in g:
        total = total + (a * a).sum()

    return torch.sqrt(
        total.clamp_min(1e-20)
    )


@torch.no_grad()
def wml_hypergradient_update(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    eta: float = 0.1,
    T: float = 3.0,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Faithful translation of the weight-update logic in the supplied
    official train_4head.py.

    The official code uses one batch and computes a gradient alignment term
    between each head's validation-style objective and the weighted training
    objective, followed by a multiplicative/exponentiated update and
    normalization.

    The outer no_grad is intentionally overridden around the gradient
    calculation because this function is itself called during training.

    Returns
    -------
    torch.Tensor
        Updated normalized four-head weights.
    """
    if weights.numel() != 4:
        raise ValueError(
            f"WML requires four weights, got shape {tuple(weights.shape)}."
        )

    if lr < 0:
        raise ValueError(f"lr must be non-negative, got {lr}.")

    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}.")

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    with torch.enable_grad():
        params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        outputs = model(x)

        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError(
                "WML model must return four logits."
            )

        if len(outputs) != 4:
            raise RuntimeError(
                f"WML model must return four logits, "
                f"got {len(outputs)}."
            )

        avg = sum(outputs) / 4.0

        # ---------------------------------------------------------------
        # Per-head objectives
        # ---------------------------------------------------------------
        grads_head = []

        for k in range(4):
            model.zero_grad(set_to_none=True)

            ce = F.cross_entropy(
                outputs[k],
                y,
            )

            teacher = F.softmax(
                outputs[k].detach() / T,
                dim=-1,
            )

            # Official mechanism distills from the ensemble average into
            # the selected head. Preserve that direction here.
            log_avg = F.log_softmax(
                avg / T,
                dim=-1,
            )

            kd = -(
                teacher * log_avg
            ).sum(-1).mean() * (T * T)

            loss = (
                (1.0 - alpha) * ce
                +
                alpha * 4.0 * kd
            )

            g = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            grads_head.append([
                gi.detach().clone()
                if gi is not None
                else torch.zeros_like(p)
                for gi, p in zip(g, params)
            ])

        # ---------------------------------------------------------------
        # Weighted classification objective used for grad_2
        # ---------------------------------------------------------------
        model.zero_grad(set_to_none=True)

        weighted_ce = sum(
            weights[k]
            * F.cross_entropy(outputs[k], y)
            for k in range(4)
        )

        g2_raw = torch.autograd.grad(
            weighted_ce,
            params,
            allow_unused=True,
        )

        g2 = [
            gi.detach().clone()
            if gi is not None
            else torch.zeros_like(p)
            for gi, p in zip(g2_raw, params)
        ]

        # ---------------------------------------------------------------
        # Multiplicative weight update
        # ---------------------------------------------------------------
        new_w = weights.detach().clone().float()

        for k in range(4):
            dot = _flat_grad_dot(
                grads_head[k],
                g2,
            ).item()

            direct = F.cross_entropy(
                outputs[k],
                y,
            ).detach().item()

            exponent = eta * (
                lr * dot
                -
                direct
            )

            # Numerical protection. This is consistent with the
            # multiplicative update while preventing exp overflow.
            exponent = max(
                min(exponent, 20.0),
                -20.0,
            )

            new_w[k] = (
                new_w[k]
                *
                torch.exp(
                    torch.tensor(
                        exponent,
                        device=new_w.device,
                        dtype=new_w.dtype,
                    )
                )
            )

        denominator = new_w.sum().clamp_min(1e-12)
        new_w = new_w / denominator

        if not torch.isfinite(new_w).all():
            raise FloatingPointError(
                "WML hypergradient update produced non-finite "
                "head weights."
            )

        return new_w


# ===========================================================================
# Optional four-head WML backbone
# ===========================================================================

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inp: int,
        out: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inp,
            out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out)

        self.conv2 = nn.Conv2d(
            out,
            out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out)

        if stride != 1 or inp != out:
            self.down = nn.Sequential(
                nn.Conv2d(
                    inp,
                    out,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out),
            )
        else:
            self.down = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r = self.down(x)

        x = F.relu(
            self.bn1(
                self.conv1(x)
            ),
            inplace=True,
        )

        x = self.bn2(
            self.conv2(x)
        )

        return F.relu(
            x + r,
            inplace=True,
        )


class WMLFourHeadResNet18(nn.Module):
    """
    CIFAR-sized four-head topology matching the supplied official
    ResNet_1_4head design.

    Shared:
        stem + layer1

    Branches:
        four independent ResNet-style branches

    Heads:
        four independent classifiers
    """

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            _BasicBlock(64, 64),
            _BasicBlock(64, 64),
        )

        self.branches = nn.ModuleList()

        for _ in range(4):
            self.branches.append(
                nn.Sequential(
                    _BasicBlock(
                        64,
                        128,
                        stride=2,
                    ),
                    _BasicBlock(
                        128,
                        128,
                    ),
                    _BasicBlock(
                        128,
                        256,
                        stride=2,
                    ),
                    _BasicBlock(
                        256,
                        256,
                    ),
                    _BasicBlock(
                        256,
                        512,
                        stride=2,
                    ),
                    _BasicBlock(
                        512,
                        512,
                    ),
                )
            )

        self.fc = nn.ModuleList([
            nn.Linear(
                512,
                num_classes,
            )
            for _ in range(4)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.stem(x)
        x = self.layer1(x)

        outs = []

        for branch, fc in zip(
            self.branches,
            self.fc,
        ):
            h = branch(x)

            h = F.adaptive_avg_pool2d(
                h,
                output_size=1,
            ).flatten(1)

            outs.append(
                fc(h)
            )

        return tuple(outs)


# ===========================================================================
# Diff-DML
# ===========================================================================

def diffdml_dpl_objective(
    z_list: Sequence[torch.Tensor],
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Mechanism-level DPLO reimplementation.

    The objective penalizes insufficient pairwise output diversity.

    For a pair (k, j):

        max(0, margin - symmetric_KL)^2

    Once the symmetric KL reaches the margin, the pair contributes zero.
    """
    if len(z_list) == 0:
        raise ValueError(
            "z_list must contain at least one tensor."
        )

    if len(z_list) < 2:
        return z_list[0].new_zeros(())

    if margin < 0:
        raise ValueError(
            f"margin must be non-negative, got {margin}."
        )

    probs = [
        F.softmax(z, dim=-1)
        for z in z_list
    ]

    total = z_list[0].new_zeros(())
    n_pairs = 0

    for k in range(len(z_list)):
        for j in range(k + 1, len(z_list)):
            pk = probs[k].clamp_min(1e-8)
            pj = probs[j].clamp_min(1e-8)

            skl = 0.5 * (
                (
                    pk
                    * (
                        pk.log()
                        -
                        pj.log()
                    )
                ).sum(-1).mean()
                +
                (
                    pj
                    * (
                        pj.log()
                        -
                        pk.log()
                    )
                ).sum(-1).mean()
            )

            total = (
                total
                +
                F.relu(
                    margin - skl
                ).pow(2)
            )

            n_pairs += 1

    result = total / float(n_pairs)

    if not torch.isfinite(result):
        raise FloatingPointError(
            "DiffDML DPL objective became non-finite."
        )

    return result


def differentiated_training_views(
    x: torch.Tensor,
    peer_idx: int,
    *,
    base_strength: float = 0.0,
    step_strength: float = 0.15,
    training: bool = True,
) -> torch.Tensor:
    """
    Practical DTS adapter for CIFAR-10.

    Different peers receive different augmentation strengths.

    The implementation deliberately uses spatial/channel-consistent
    operations rather than the previous differentiated_dropout shortcut.

    Parameters
    ----------
    x:
        Input batch [B, C, H, W].
    peer_idx:
        Zero-based peer index.
    base_strength:
        Base augmentation strength.
    step_strength:
        Increment per peer.
    training:
        If False, returns x unchanged.
    """
    if not training:
        return x

    if peer_idx < 0:
        raise ValueError(
            f"peer_idx must be non-negative, got {peer_idx}."
        )

    if base_strength < 0 or step_strength < 0:
        raise ValueError(
            "base_strength and step_strength must be non-negative."
        )

    s = min(
        max(
            base_strength
            +
            peer_idx * step_strength,
            0.0,
        ),
        1.0,
    )

    if s == 0.0:
        return x

    # ---------------------------------------------------------------
    # Differentiated stochastic horizontal flip.
    # ---------------------------------------------------------------
    if (
        s > 0.0
        and
        torch.rand(
            (),
            device=x.device,
        ) < 0.5 * s
    ):
        x = torch.flip(
            x,
            dims=[3],
        )

    # ---------------------------------------------------------------
    # Mild random cutout.
    #
    # Strength grows with peer index.
    # ---------------------------------------------------------------
    if s > 0.0:
        B, C, H, W = x.shape

        side = max(
            1,
            int(round(4 * s)),
        )

        if side < H and side < W:
            top = torch.randint(
                0,
                H - side + 1,
                (1,),
                device=x.device,
            ).item()

            left = torch.randint(
                0,
                W - side + 1,
                (1,),
                device=x.device,
            ).item()

            x = x.clone()

            x[
                :,
                :,
                top:top + side,
                left:left + side,
            ] = 0

    return x


# Backward-compatible alias.
#
# Kept so existing imports from earlier versions of the experiment do not
# break. The implementation now points to the DTS-style training views.
def differentiated_dropout(
    x: torch.Tensor,
    peer_idx: int,
    base_rate: float = 0.1,
    step: float = 0.1,
    training: bool = True,
) -> torch.Tensor:
    """
    Backward-compatible alias for differentiated_training_views().
    """
    return differentiated_training_views(
        x,
        peer_idx,
        base_strength=base_rate,
        step_strength=step,
        training=training,
    )


# ===========================================================================
# NCL-inspired representation repulsion
# ===========================================================================

def _normalize(
    h: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Numerically stable L2 normalization.

    A small norm floor prevents extremely small representation norms from
    producing excessively large gradients.

    The default eps is deliberately larger than the internal default used
    by F.normalize because this function is part of a training objective
    whose negative repulsion term can otherwise amplify numerical
    instability.
    """
    if eps <= 0:
        raise ValueError(
            f"eps must be positive, got {eps}."
        )

    norm = h.norm(
        p=2,
        dim=-1,
        keepdim=True,
    )

    norm = norm.clamp_min(eps)

    return h / norm


def ncl_repulsion_loss(
    h_list: Sequence[torch.Tensor],
    strength: float = 0.1,
) -> torch.Tensor:
    """
    NCL-inspired normalized-representation repulsion.

    The objective maximizes the mean pairwise squared distance between
    normalized representations:

        L_rep =
            - strength *
            mean_{i<j} ||r_i - r_j||_2^2

    where

        r_i = h_i / max(||h_i||_2, eps).

    Since each representation is normalized, the pairwise squared distance
    is bounded by 4. Therefore this is not literally "unbounded" geometric
    repulsion. It simply has no finite target distance and no saturation
    term.

    This formulation is preferable to a mean-centered dispersion term for
    the baseline because it directly measures pairwise peer separation and
    does not implicitly privilege the cohort mean.

    Parameters
    ----------
    h_list:
        Per-peer representations. Each tensor is expected to have shape
        [B, D].
    strength:
        Positive repulsion coefficient.

    Returns
    -------
    torch.Tensor
        Scalar negative repulsion contribution to a minimization objective.
    """
    if len(h_list) == 0:
        raise ValueError(
            "h_list must contain at least one representation tensor."
        )

    if strength < 0:
        raise ValueError(
            f"strength must be non-negative, got {strength}."
        )

    if len(h_list) < 2 or strength == 0.0:
        return h_list[0].new_zeros(())

    reps = [
        _normalize(h)
        for h in h_list
    ]

    total = reps[0].new_zeros(())
    n_pairs = 0

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            d2 = (
                reps[i]
                -
                reps[j]
            ).pow(2).sum(dim=-1)

            total = total + d2.mean()
            n_pairs += 1

    dispersion = total / float(n_pairs)

    # Training minimizes the objective. Therefore a negative sign causes
    # optimization to maximize representation separation.
    repulsion = -strength * dispersion

    if not torch.isfinite(repulsion):
        raise FloatingPointError(
            "NCL-inspired representation repulsion became non-finite."
        )

    return repulsion


def unbounded_repulsion_loss(
    h_list: Sequence[torch.Tensor],
    strength: float = 0.1,
) -> torch.Tensor:
    """
    Backward-compatible alias.

    Historically this function was named 'unbounded_repulsion_loss'.
    Because normalized representations have bounded pairwise distances,
    the name is technically misleading. It is retained to avoid breaking
    existing imports.

    New code should use ncl_repulsion_loss().
    """
    return ncl_repulsion_loss(
        h_list,
        strength=strength,
    )


def ncl_inspired_objective(
    h_list: Sequence[torch.Tensor],
    z_list: Sequence[torch.Tensor],
    y: torch.Tensor,
    lambda_out: float = 1.0,
    strength: float = 0.1,
    T: float = 4.0,
) -> torch.Tensor:
    """
    Stable NCL-inspired training objective.

    The objective is

        L_NCL =
            sum_k CE(z_k, y)
            +
            lambda_out * sum_{k != j} KD(z_k, z_j)
            -
            strength * mean_{k<j}
                ||r_k - r_j||_2^2,

    with

        r_k = h_k / max(||h_k||_2, eps).

    Important methodological note
    -----------------------------
    This is an NCL-inspired baseline rather than a literal implementation
    of the original NCL formulation. It provides a representation-level
    diversity baseline without using the finite target distance d0 or the
    quartic finite-target diversity potential used by SML.

    Numerical safeguards
    --------------------
    The objective explicitly checks that the resulting scalar is finite.
    A non-finite objective raises FloatingPointError instead of silently
    replacing NaN/Inf with zero. This is important for a fair experimental
    audit: numerical failure must not be hidden in the reported results.
    """
    if len(h_list) == 0:
        raise ValueError(
            "h_list must contain at least one representation tensor."
        )

    if len(h_list) != len(z_list):
        raise ValueError(
            "h_list and z_list must have the same number of peers: "
            f"{len(h_list)} vs {len(z_list)}."
        )

    if lambda_out < 0:
        raise ValueError(
            f"lambda_out must be non-negative, got {lambda_out}."
        )

    if strength < 0:
        raise ValueError(
            f"strength must be non-negative, got {strength}."
        )

    if T <= 0:
        raise ValueError(
            f"T must be positive, got {T}."
        )

    # ---------------------------------------------------------------
    # Task loss
    # ---------------------------------------------------------------
    task_terms = [
        F.cross_entropy(z, y)
        for z in z_list
    ]

    task = sum(task_terms)

    # ---------------------------------------------------------------
    # Peer output distillation
    # ---------------------------------------------------------------
    out = z_list[0].new_zeros(())

    for k in range(len(z_list)):
        for j in range(len(z_list)):
            if j != k:
                out = out + kd_kl(
                    z_list[k],
                    z_list[j],
                    T=T,
                )

    # ---------------------------------------------------------------
    # Representation repulsion
    # ---------------------------------------------------------------
    rep = ncl_repulsion_loss(
        h_list,
        strength=strength,
    )

    total = (
        task
        +
        lambda_out * out
        +
        rep
    )

    # ---------------------------------------------------------------
    # Explicit numerical audit.
    #
    # Do not use torch.nan_to_num() here. A numerical failure should remain
    # visible rather than being silently converted into a valid-looking
    # training result.
    # ---------------------------------------------------------------
    if not torch.isfinite(task):
        raise FloatingPointError(
            "NCL-inspired task loss became non-finite: "
            f"{task.detach().item()}"
        )

    if not torch.isfinite(out):
        raise FloatingPointError(
            "NCL-inspired output KD loss became non-finite: "
            f"{out.detach().item()}"
        )

    if not torch.isfinite(rep):
        raise FloatingPointError(
            "NCL-inspired representation loss became non-finite: "
            f"{rep.detach().item()}"
        )

    if not torch.isfinite(total):
        raise FloatingPointError(
            "NCL-inspired total objective became non-finite: "
            f"task={task.detach().item()}, "
            f"out={out.detach().item()}, "
            f"rep={rep.detach().item()}, "
            f"total={total.detach().item()}"
        )

    return total