"""
data.py
=======

Data loaders for:
1. CIFAR-10
2. CIFAR-100
3. SyntheticImageDataset (offline smoke testing only)

The CIFAR loaders are intended for the real experimental protocol.
They require torchvision and internet access on the first run in order
to download the datasets.

CIFAR-10 and CIFAR-100 use their respective dataset-specific channel
statistics for normalization.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """
    Initialize NumPy and Python RNGs inside each DataLoader worker.

    torch.initial_seed() is derived from the DataLoader generator,
    so workers receive deterministic but distinct seeds.
    """
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------

def get_cifar_loaders(
    root="./data",
    dataset="cifar10",
    batch_size=128,
    num_workers=2,
    seed=0,
):
    """
    Load CIFAR-10 or CIFAR-100 with dataset-specific normalization.

    Parameters
    ----------
    root : str
        Root directory in which the dataset is stored/downloaded.

    dataset : str
        Either "cifar10" or "cifar100".

    batch_size : int
        Batch size for both training and test loaders.

    num_workers : int
        Number of DataLoader worker processes.

    seed : int
        Seed used for deterministic training-set shuffling.

    Returns
    -------
    train_loader : DataLoader
    test_loader : DataLoader
    num_classes : int
        10 for CIFAR-10 and 100 for CIFAR-100.
    """

    import torchvision
    import torchvision.transforms as T

    # Normalize dataset name to make the interface robust to
    # accidental capitalization or surrounding whitespace.
    dataset = str(dataset).strip().lower()

    # -------------------------------------------------------------
    # Dataset-specific configuration
    # -------------------------------------------------------------

    if dataset == "cifar10":

        cls = torchvision.datasets.CIFAR10

        num_classes = 10

        # Standard CIFAR-10 channel statistics.
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

    elif dataset == "cifar100":

        cls = torchvision.datasets.CIFAR100

        num_classes = 100

        # Standard CIFAR-100 channel statistics.
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Expected 'cifar10' or 'cifar100'."
        )

    # -------------------------------------------------------------
    # Data augmentation
    # -------------------------------------------------------------

    # The same standard CIFAR augmentation is used for both
    # datasets so that the experimental protocol remains comparable.
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # No stochastic augmentation is applied to the test set.
    # The test images are normalized using the SAME statistics
    # as the corresponding training dataset.
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # -------------------------------------------------------------
    # Dataset objects
    # -------------------------------------------------------------

    train_set = cls(
        root=root,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_set = cls(
        root=root,
        train=False,
        download=True,
        transform=test_tf,
    )

    # -------------------------------------------------------------
    # Deterministic training-set shuffling
    # -------------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )

    # Test set must not be shuffled.
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, test_loader, num_classes


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------

class SyntheticImageDataset(Dataset):
    """
    Offline synthetic classification dataset for smoke testing only.

    Each class is represented by a fixed random 3x32x32 template.
    Individual samples are generated by adding Gaussian noise to the
    corresponding class template.

    This dataset is NOT a substitute for CIFAR-10/100 and should not
    be used for scientific conclusions.
    """

    def __init__(
        self,
        num_classes=10,
        n_per_class=200,
        size=32,
        seed=0,
        templates=None,
    ):
        g = torch.Generator().manual_seed(seed)

        self.templates = (
            templates.clone()
            if templates is not None
            else torch.randn(
                num_classes,
                3,
                size,
                size,
                generator=g,
            )
        )

        self.num_classes = num_classes
        self.n_per_class = n_per_class
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.num_classes * self.n_per_class

    def __getitem__(self, idx):
        c = idx % self.num_classes

        g = torch.Generator().manual_seed(
            self.seed * 100000 + idx
        )

        noise = 0.5 * torch.randn(
            3,
            self.size,
            self.size,
            generator=g,
        )

        x = self.templates[c] + noise

        return x, c


def get_synthetic_loaders(
    num_classes=10,
    n_per_class_train=200,
    n_per_class_test=40,
    batch_size=32,
    seed=0,
):
    """
    Return synthetic train/test loaders for offline smoke testing.
    """

    # Same class templates define the underlying distribution.
    # Train and test differ only in sample noise and RNG seed.
    g = torch.Generator().manual_seed(seed)

    templates = torch.randn(
        num_classes,
        3,
        32,
        32,
        generator=g,
    )

    train_set = SyntheticImageDataset(
        num_classes=num_classes,
        n_per_class=n_per_class_train,
        seed=seed,
        templates=templates,
    )

    test_set = SyntheticImageDataset(
        num_classes=num_classes,
        n_per_class=n_per_class_test,
        seed=seed + 1,
        templates=templates,
    )

    train_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_gen,
        worker_init_fn=_worker_init_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, test_loader, num_classes

