"""
models.py
=========
Peer network backbone, updated to match SML_final_audited11.tex.

Key change from the earlier draft: pairwise projections P_kj have been
REMOVED from the manuscript ("Summary of the Final Formulation...":
"peer representations are mapped directly to a common unit sphere through
l2 normalization. No pair-specific projection P_kj is introduced."). All
peers therefore share the same D-dimensional space and are compared after
normalization onto S^{D-1}; no learnable projection module is needed.

    h_k(x; theta_k) in R^D        -- raw intermediate representation
    r_k(x) = h_k(x) / ||h_k(x)||  -- Eq. (related-normalized-representation)
    p_k(x) = f_k(x; theta_k)      -- predictive distribution
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class PeerNet(nn.Module):
    """
    Small CNN classifier exposing a raw representation h_k in R^D. `width`
    lets peers have different capacities while emitting representations of
    the same shared dimension `repr_dim`, so K heterogeneous peers can be
    built without requiring identical feature extractors.
    """

    def __init__(self, in_channels=3, num_classes=10, repr_dim=128, width=32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, width, pool=True),
            ConvBlock(width, width * 2, pool=True),
            ConvBlock(width * 2, width * 4, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.repr_head = nn.Linear(width * 4, repr_dim)
        self.classifier = nn.Linear(repr_dim, num_classes)
        self.repr_dim = repr_dim

    def forward(self, x):
        feat = self.features(x)
        feat = self.gap(feat).flatten(1)
        h = self.repr_head(feat)      # raw representation h_k
        z = self.classifier(h)        # logits z_k (classifier acts on the
                                       # raw h_k, keeping the task pathway
                                       # separate from the geometric
                                       # interaction pathway, which operates
                                       # on the normalized r_k)
        return h, z


def build_cohort(K, num_classes=10, repr_dim=128, widths=None, in_channels=3):
    """Builds K peer networks (f_k(x; theta_k), k = 1..K)."""
    if widths is None:
        widths = [32] * K
    assert len(widths) == K
    return nn.ModuleList([
        PeerNet(in_channels=in_channels, num_classes=num_classes,
                repr_dim=repr_dim, width=w)
        for w in widths
    ])


def normalize_representation(h, eps=1e-8):
    """
    Eq. (related-normalized-representation): r_k(x) = h_k(x) / ||h_k(x)||_2.
    F.normalize divides by max(||h||, eps), matching r_k(x) for h_k(x) != 0
    while avoiding division by zero.
    """
    return F.normalize(h, dim=-1, eps=eps)
