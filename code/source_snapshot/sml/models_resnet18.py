"""
models_resnet18.py
==================
Backbones for SML experiments.

This module preserves the original small PeerNet and adds a genuinely
different CIFAR-adapted ResNet-18 peer. Both expose:

    h_k(x) in R^D    -- raw representation used by the classifier
    z_k(x)           -- logits

The SML geometric pathway continues to normalize h_k externally, so the
paper formulation is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Original small CNN
# ---------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x)), inplace=True))


class PeerNet(nn.Module):
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
        h = self.repr_head(feat)
        z = self.classifier(h)
        return h, z


# ---------------------------------------------------------------------
# CIFAR ResNet-18
# ---------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet18Peer(nn.Module):
    """
    ResNet-18 adapted to 32x32 CIFAR images:
      - 3x3 stride-1 stem
      - no initial max-pool
      - stages [64, 128, 256, 512]
      - learned projection from pooled 512-D feature to shared repr_dim
    """

    def __init__(self, in_channels=3, num_classes=10, repr_dim=64):
        super().__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 64, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.repr_head = nn.Linear(512, repr_dim)
        self.classifier = nn.Linear(repr_dim, num_classes)
        self.repr_dim = repr_dim

        self._reset_parameters()

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        h = self.repr_head(x)
        z = self.classifier(h)
        return h, z


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_cohort(
    K,
    num_classes=10,
    repr_dim=128,
    widths=None,
    in_channels=3,
    backbone="small_cnn",
):
    name = str(backbone).lower()

    if name in {"small_cnn", "peer", "peernet", "default"}:
        if widths is None:
            widths = [32] * K
        if len(widths) != K:
            raise ValueError("len(widths) must equal K.")
        return nn.ModuleList([
            PeerNet(
                in_channels=in_channels,
                num_classes=num_classes,
                repr_dim=repr_dim,
                width=w,
            )
            for w in widths
        ])

    if name in {"resnet18", "cifar_resnet18", "resnet18_cifar"}:
        return nn.ModuleList([
            CIFARResNet18Peer(
                in_channels=in_channels,
                num_classes=num_classes,
                repr_dim=repr_dim,
            )
            for _ in range(K)
        ])

    raise ValueError(f"Unknown backbone: {backbone!r}")


def normalize_representation(h, eps=1e-8):
    return F.normalize(h, dim=-1, eps=eps)
