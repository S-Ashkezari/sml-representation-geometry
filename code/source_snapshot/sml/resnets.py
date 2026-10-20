# sml/resnets.py — optional ResNet-style peer with the same (h, z) interface
import torch, torch.nn as nn, torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.short(x))

class ResNetPeer(nn.Module):
    """Basic ResNet for 32x32 inputs emitting (h, z) like PeerNet."""
    def __init__(self, num_classes=10, repr_dim=64, width=16):
        super().__init__()
        w = width
        self.conv = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
            BasicBlock(w, 2*w), BasicBlock(2*w, 2*w), nn.MaxPool2d(2),
            BasicBlock(2*w, 4*w), BasicBlock(4*w, 4*w), nn.MaxPool2d(2),
            BasicBlock(4*w, 8*w), BasicBlock(8*w, 8*w))
        self.fc_rep = nn.Linear(8*w, repr_dim); self.fc_cls = nn.Linear(repr_dim, num_classes)
        self.repr_dim = repr_dim
    def forward(self, x):
        f = self.conv(x).mean(dim=(2, 3))
        h = self.fc_rep(f); return h, self.fc_cls(h)