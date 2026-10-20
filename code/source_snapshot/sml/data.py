"""
data.py
=======
Two data sources:

1. `get_cifar_loaders` -- the real benchmark used in the Experimental
   Protocol section. Requires internet access to download CIFAR-10/100 via
   torchvision (not available in this sandbox, but this is the function to
   call on a machine with normal internet/GPU access).

2. `SyntheticImageDataset` -- a small, fully offline synthetic classification
   task (random-but-fixed class-conditional Gaussian blobs rendered as
   textured images) used ONLY to smoke-test that the training/loss code
   executes correctly end-to-end in this sandbox. It is NOT a substitute for
   a real benchmark and should not be used to draw scientific conclusions.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader


def get_cifar_loaders(root="./data", dataset="cifar10", batch_size=128,
                       num_workers=2):
    """Real experimental-protocol data loader (needs internet access)."""
    import torchvision
    import torchvision.transforms as T

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        # Standard CIFAR-100 channel statistics. CIFAR-10 statistics are
        # kept unchanged so existing CIFAR-10 results remain comparable.
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(
            f"Unsupported CIFAR dataset '{dataset}'. "
            "Expected 'cifar10' or 'cifar100'."
        )
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    num_classes = 10 if dataset == "cifar10" else 100

    train_set = cls(root=root, train=True, download=True, transform=train_tf)
    test_set = cls(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, test_loader, num_classes


class SyntheticImageDataset(Dataset):
    """
    Offline stand-in dataset for smoke-testing only. Each class is a fixed
    random 3x32x32 template plus per-sample Gaussian noise, so the
    classification task is easy and learnable in a handful of iterations,
    letting us verify the training loop, losses, and metrics run without
    errors and produce sane numbers (accuracy > chance, D_h finite, etc.).
    """

    def __init__(self, num_classes=10, n_per_class=200, size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.templates = torch.randn(num_classes, 3, size, size, generator=g)
        self.num_classes = num_classes
        self.n_per_class = n_per_class
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.num_classes * self.n_per_class

    def __getitem__(self, idx):
        c = idx % self.num_classes
        g = torch.Generator().manual_seed(self.seed * 100000 + idx)
        noise = 0.5 * torch.randn(3, self.size, self.size, generator=g)
        x = self.templates[c] + noise
        return x, c


def get_synthetic_loaders(num_classes=10, n_per_class_train=200,
                           n_per_class_test=40, batch_size=32, seed=0):
    train_set = SyntheticImageDataset(num_classes, n_per_class_train, seed=seed)
    test_set = SyntheticImageDataset(num_classes, n_per_class_test, seed=seed + 1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes
