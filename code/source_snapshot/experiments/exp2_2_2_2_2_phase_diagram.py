"""
exp2_2_2_phase_diagram.py
=========================

Publication-grade SML phase-diagram experiment.

Protocol:
- CIFAR-10/100 or synthetic smoke-test data.
- K=2 SML models.
- One exact initial model state per seed, shared across all parameter
  configurations belonging to that seed.
- Fresh optimizer state for every configuration.
- Incremental raw CSV writing after every completed run.
- Safe resume using a protocol-aware key.
- Existing legacy/recovered rows are accepted only when their scientific
  configuration matches the current experiment; unknown legacy execution
  metadata is reported explicitly.

The reduced pairwise theory predicts

    d_*^2 = d_0^2 - w / eta,

    w = 2 lambda_rep / lambda_div,

    rho = eta d_0^2 / w,

with a non-zero equilibrium only for rho > 1.

Important:
D_rep is the empirical mean pairwise SQUARED distance between the
L2-normalized representations. Therefore D_rep is directly comparable to
the theoretical quantity d_*^2 (same squared-distance geometry).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sml.train import SMLConfig, SMLTrainer
from sml.data import get_cifar_loaders, get_synthetic_loaders


# ============================================================================
# Experiment constants
# ============================================================================

EXPERIMENT_VERSION = "2.0-publication-audit"

LAMBDA_DIV = 1.0

FULL_LAMBDA_REPS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4]
FULL_D0S = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
FULL_ETAS = [0.5, 1.0, 2.0]

SMOKE_LAMBDA_REPS = [0.2]
SMOKE_D0S = [0.5, 1.0]
SMOKE_ETAS = [0.5, 2.0]


RAW_RESULT_FIELDS = [
    "run_index",
    "seed",
    "epochs",
    "lambda_rep",
    "lambda_div",
    "d0",
    "eta",
    "w",
    "rho",
    "D_rep",
    "E_d0",
    "ensemble_acc",
    "d_star_squared_theory",
    "initial_digest",
    "final_digest",
    "initial_state_file",
    "dataset",
    "synthetic",
    "smoke_cifar",
    "repr_dim",
    "common_initialization",
    "deterministic",
    "train_samples",
    "test_samples",
    "experiment_version",
    "initial_state_digest",
    "recovery_status",
]

RAW_RESULT_TYPES = {
    "run_index": int,
    "seed": int,
    "epochs": int,
    "lambda_rep": float,
    "lambda_div": float,
    "d0": float,
    "eta": float,
    "w": float,
    "rho": float,
    "D_rep": float,
    "E_d0": float,
    "ensemble_acc": float,
    "d_star_squared_theory": float,
}


# ============================================================================
# Theory
# ============================================================================

def compute_w(lambda_rep: float, lambda_div: float) -> float:
    if lambda_div <= 0:
        raise ValueError("lambda_div must be > 0.")
    return 2.0 * lambda_rep / lambda_div


def compute_rho(
    d0: float,
    lambda_rep: float,
    lambda_div: float,
    eta: float,
) -> float:
    w = compute_w(lambda_rep, lambda_div)
    if w == 0:
        return float("inf")
    return eta * d0**2 / w


def theoretical_d_star_squared(
    d0: float,
    lambda_rep: float,
    lambda_div: float,
    eta: float,
) -> float | None:
    if eta <= 0:
        raise ValueError("eta must be > 0.")
    w = compute_w(lambda_rep, lambda_div)
    if w == 0:
        return d0**2
    value = d0**2 - w / eta
    return value if value > 0 else None


# ============================================================================
# Determinism
# ============================================================================

def enable_deterministic_mode() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        print(
            "[WARNING] torch.use_deterministic_algorithms(True) failed: "
            f"{type(exc).__name__}: {exc}"
        )

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


# ============================================================================
# Digests
# ============================================================================

def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def state_digest(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()

    for name, tensor in module.state_dict().items():
        hasher.update(name.encode("utf-8"))
        array = tensor.detach().to("cpu").contiguous().numpy()
        hasher.update(array.tobytes())

    return hasher.hexdigest()[:16]


# ============================================================================
# Runtime metadata
# ============================================================================

def runtime_metadata(device: str) -> dict:
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }

    try:
        import torchvision
        meta["torchvision"] = torchvision.__version__
    except Exception:
        meta["torchvision"] = None

    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            meta["tf32_matmul"] = bool(
                torch.backends.cuda.matmul.allow_tf32
            )
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            meta["tf32_cudnn"] = bool(
                torch.backends.cudnn.allow_tf32
            )
    else:
        meta["gpu"] = None
        meta["tf32_matmul"] = None
        meta["tf32_cudnn"] = None

    return meta


# ============================================================================
# Data
# ============================================================================

def subset_loader(
    loader,
    n_samples: int | None,
    seed: int,
) -> DataLoader:
    """
    Preserve the original smoke-test semantics.

    This function intentionally does not alter the full-CIFAR protocol.
    """
    if n_samples is None:
        return loader

    n = min(int(n_samples), len(loader.dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(loader.dataset),
        size=n,
        replace=False,
    ).tolist()

    subset = Subset(loader.dataset, indices)

    batch_size = (
        loader.batch_size
        if loader.batch_size is not None
        else 32
    )

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False,
    )


def create_loaders(args, seed: int):
    if args.synthetic:
        return get_synthetic_loaders(
            num_classes=10,
            n_per_class_train=64,
            n_per_class_test=32,
            batch_size=32,
            seed=seed,
        )

    train_loader, test_loader, num_classes = get_cifar_loaders(
        dataset=args.dataset,
        seed=seed,
    )

    if args.smoke_cifar:
        train_loader = subset_loader(
            train_loader,
            args.train_samples,
            seed=seed,
        )
        test_loader = subset_loader(
            test_loader,
            args.test_samples,
            seed=seed + 10000,
        )

        print(
            f"[smoke-cifar] train samples = "
            f"{len(train_loader.dataset)}"
        )
        print(
            f"[smoke-cifar] test samples  = "
            f"{len(test_loader.dataset)}"
        )

    return train_loader, test_loader, num_classes


# ============================================================================
# Trainer
# ============================================================================

def create_trainer(
    num_classes: int,
    device: str,
    repr_dim: int,
    seed: int,
    lambda_rep: float,
    lambda_div: float,
    d0: float,
    eta: float,
) -> SMLTrainer:
    """
    Construct a fresh trainer.

    The model RNG is seeded before construction. The optimizer is also fresh
    for every configuration. The common-initialization state, when enabled,
    overwrites only the model state; it never reuses optimizer state.
    """
    from sml.data import seed_everything

    seed_everything(seed)

    config = SMLConfig(
        K=2,
        num_classes=num_classes,
        repr_dim=repr_dim,
        lambda_out=1.0,
        lambda_rep=lambda_rep,
        lambda_div=lambda_div,
        d0=d0,
        eta=eta,
        use_out=True,
        use_rep=True,
        use_div=True,
        device=device,
    )

    return SMLTrainer(
        config,
        in_channels=3,
    )


# ============================================================================
# Initial-state management
# ============================================================================

def initial_state_path(
    output_dir: str,
    seed: int,
) -> str:
    return os.path.join(
        output_dir,
        f"common_initial_state_seed{seed}.pt",
    )


def save_initial_state(
    path: str,
    trainer: SMLTrainer,
    seed: int,
) -> str:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    digest = state_digest(trainer.models)

    payload = {
        "format_version": 2,
        "seed": int(seed),
        "model_state": trainer.models.state_dict(),
        "K": int(trainer.cfg.K),
        "repr_dim": int(trainer.cfg.repr_dim),
        "num_classes": int(trainer.cfg.num_classes),
        "initial_digest": digest,
    }

    torch.save(payload, path)

    print(f"[INIT] Saved seed-specific initial state: {path}")
    print(f"[INIT] initial digest = {digest}")

    return digest


def load_initial_state(
    path: str,
    trainer: SMLTrainer,
    expected_seed: int | None = None,
) -> str:
    payload = torch.load(
        path,
        map_location="cpu",
    )

    required = [
        "model_state",
        "K",
        "repr_dim",
        "num_classes",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(
            f"Initial-state file is missing required metadata: {missing}"
        )

    if expected_seed is not None:
        saved_seed = payload.get("seed")
        if saved_seed is not None and int(saved_seed) != int(expected_seed):
            raise RuntimeError(
                f"Initial-state seed mismatch: file={saved_seed}, "
                f"requested={expected_seed}"
            )

    if int(payload["K"]) != int(trainer.cfg.K):
        raise RuntimeError(
            f"Initial-state K mismatch: file={payload['K']}, "
            f"trainer={trainer.cfg.K}"
        )

    if int(payload["repr_dim"]) != int(trainer.cfg.repr_dim):
        raise RuntimeError(
            f"Initial-state repr_dim mismatch: file={payload['repr_dim']}, "
            f"trainer={trainer.cfg.repr_dim}"
        )

    if int(payload["num_classes"]) != int(trainer.cfg.num_classes):
        raise RuntimeError(
            f"Initial-state num_classes mismatch: file={payload['num_classes']}, "
            f"trainer={trainer.cfg.num_classes}"
        )

    trainer.models.load_state_dict(
        payload["model_state"],
        strict=True,
    )

    digest = state_digest(trainer.models)
    expected = payload.get("initial_digest")

    if expected is not None and digest != expected:
        raise RuntimeError(
            "Initial-state digest mismatch: "
            f"expected={expected}, actual={digest}"
        )

    print(f"[INIT] Loaded initial state: {path}")
    print(f"[INIT] loaded model digest = {digest}")

    return digest


def get_or_create_initial_state(
    output_dir: str,
    seed: int,
    num_classes: int,
    device: str,
    repr_dim: int,
) -> tuple[str, str]:
    path = initial_state_path(output_dir, seed)

    trainer = create_trainer(
        num_classes=num_classes,
        device=device,
        repr_dim=repr_dim,
        seed=seed,
        lambda_rep=0.0,
        lambda_div=1.0,
        d0=0.5,
        eta=0.5,
    )

    if os.path.exists(path):
        print(
            f"[INIT] Existing initial state found for seed={seed}"
        )
        digest = load_initial_state(
            path,
            trainer,
            expected_seed=seed,
        )
        return path, digest

    print(
        f"[INIT] Creating independent initial state for seed={seed}"
    )

    digest = save_initial_state(
        path,
        trainer,
        seed,
    )
    return path, digest


# ============================================================================
# Protocol metadata / resume
# ============================================================================

def protocol_metadata(args, lambda_reps, d0s, etas, configurations) -> dict:
    """
    Metadata describing the scientific protocol.

    `deterministic` is intentionally recorded but not used as part of the
    scientific (config, seed) resume key, because legacy terminal-recovered
    rows may not expose whether that flag was enabled. A warning is emitted
    when legacy rows have unknown execution metadata.
    """
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": args.dataset,
        "synthetic": bool(args.synthetic),
        "smoke_cifar": bool(args.smoke_cifar),
        "epochs": int(args.epochs),
        "seeds": int(args.seeds),
        "repr_dim": int(args.repr_dim),
        "common_initialization": bool(args.common_initialization),
        "deterministic": bool(args.deterministic),
        "train_samples": int(args.train_samples),
        "test_samples": int(args.test_samples),
        "lambda_div": float(LAMBDA_DIV),
        "lambda_reps": list(lambda_reps),
        "d0s": list(d0s),
        "etas": list(etas),
        "n_configurations": len(configurations),
        "n_runs_expected": len(configurations) * args.seeds,
    }


def raw_result_key(result: dict) -> tuple:
    """
    Scientific identity of one completed run.

    This intentionally includes all SML control parameters and the seed and
    epoch budget. Dataset/protocol identity is validated separately.
    """
    return (
        float(result["lambda_rep"]),
        float(result["lambda_div"]),
        float(result["d0"]),
        float(result["eta"]),
        int(result["seed"]),
        int(result["epochs"]),
    )


def _coerce_value(value, caster):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "None", "null"}:
        return None
    return caster(value)


def coerce_raw_row(row: dict) -> dict:
    out = dict(row)

    for key, caster in RAW_RESULT_TYPES.items():
        if key in out:
            out[key] = _coerce_value(out[key], caster)

    if out.get("d_star_squared_theory") is not None:
        out["d_star_squared_theory"] = float(
            out["d_star_squared_theory"]
        )

    return out


def validate_existing_row(
    row: dict,
    args,
    expected_protocol: dict,
) -> None:
    """
    Reject rows that cannot belong to the requested scientific experiment.

    Recovered legacy rows may omit protocol metadata; those are accepted only
    after their numerical run identity is checked by the main loop.
    """
    if int(row["epochs"]) != int(args.epochs):
        raise RuntimeError(
            f"Existing row has epochs={row['epochs']}, "
            f"current experiment requires {args.epochs}."
        )

    if int(row["seed"]) < 0 or int(row["seed"]) >= int(args.seeds):
        raise RuntimeError(
            f"Existing row has seed={row['seed']}, outside current seed range."
        )

    for key in ["dataset", "repr_dim", "common_initialization"]:
        if key not in row:
            continue

        expected = expected_protocol[key]
        actual = row[key]

        if key == "repr_dim":
            if int(actual) != int(expected):
                raise RuntimeError(
                    f"Existing row protocol mismatch for {key}: "
                    f"{actual} != {expected}"
                )
        elif key == "common_initialization":
            if str(actual).lower() != str(bool(expected)).lower():
                raise RuntimeError(
                    f"Existing row protocol mismatch for {key}: "
                    f"{actual} != {expected}"
                )
        elif str(actual) != str(expected):
            raise RuntimeError(
                f"Existing row protocol mismatch for {key}: "
                f"{actual} != {expected}"
            )


def load_existing_raw_results(
    csv_path: str,
    args,
    expected_protocol: dict,
) -> tuple[list[dict], set, bool]:
    """
    Load incremental CSV.

    Returns:
        rows
        completed keys
        legacy_unknown_metadata_present
    """
    if not os.path.exists(csv_path):
        return [], set(), False

    rows = []
    legacy_unknown = False

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"Existing CSV has no header: {csv_path}"
            )

        for raw in reader:
            if not raw:
                continue

            row = coerce_raw_row(raw)

            for required in [
                "seed",
                "epochs",
                "lambda_rep",
                "lambda_div",
                "d0",
                "eta",
                "D_rep",
                "E_d0",
                "ensemble_acc",
            ]:
                if required not in row or row[required] is None:
                    raise RuntimeError(
                        f"Existing CSV contains incomplete row: {raw}"
                    )

            validate_existing_row(
                row,
                args,
                expected_protocol,
            )

            if row.get("deterministic") in {
                None,
                "",
                "UNKNOWN_FROM_LOG",
                "unknown",
                "UNKNOWN",
            }:
                legacy_unknown = True

            rows.append(row)

    keys = {raw_result_key(row) for row in rows}

    print(
        f"[RESUME] Existing raw results: {csv_path}"
    )
    print(
        f"[RESUME] Completed rows found: {len(rows)}"
    )

    if legacy_unknown:
        print(
            "[RESUME] WARNING: at least one recovered/legacy row does not "
            "record whether deterministic mode was enabled."
        )

    return rows, keys, legacy_unknown


def append_raw_result(
    csv_path: str,
    result: dict,
) -> None:
    """
    Append one completed run immediately.

    The file is flushed and fsynced before returning, reducing the chance that
    a power loss loses the just-completed row.
    """
    os.makedirs(
        os.path.dirname(csv_path) or ".",
        exist_ok=True,
    )

    file_is_new = (
        not os.path.exists(csv_path)
        or os.path.getsize(csv_path) == 0
    )

    fieldnames = RAW_RESULT_FIELDS

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if file_is_new:
            writer.writeheader()

        writer.writerow(result)
        handle.flush()
        os.fsync(handle.fileno())


# ============================================================================
# One run
# ============================================================================

def run_one(
    train_loader,
    test_loader,
    num_classes: int,
    device: str,
    repr_dim: int,
    seed: int,
    lambda_rep: float,
    lambda_div: float,
    d0: float,
    eta: float,
    epochs: int,
    initial_state_file: str | None,
) -> dict:
    trainer = create_trainer(
        num_classes=num_classes,
        device=device,
        repr_dim=repr_dim,
        seed=seed,
        lambda_rep=lambda_rep,
        lambda_div=lambda_div,
        d0=d0,
        eta=eta,
    )

    if initial_state_file is not None:
        initial_digest = load_initial_state(
            initial_state_file,
            trainer,
            expected_seed=seed,
        )
    else:
        initial_digest = state_digest(trainer.models)

    for _ in range(epochs):
        for x, y in train_loader:
            trainer.step(x, y)

    metrics = trainer.evaluate(test_loader)

    w = compute_w(lambda_rep, lambda_div)
    rho = compute_rho(
        d0=d0,
        lambda_rep=lambda_rep,
        lambda_div=lambda_div,
        eta=eta,
    )
    d_star_sq = theoretical_d_star_squared(
        d0=d0,
        lambda_rep=lambda_rep,
        lambda_div=lambda_div,
        eta=eta,
    )

    final_digest = state_digest(trainer.models)

    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "lambda_rep": float(lambda_rep),
        "lambda_div": float(lambda_div),
        "d0": float(d0),
        "eta": float(eta),
        "w": float(w),
        "rho": float(rho),
        "D_rep": float(metrics["D_rep"]),
        "E_d0": float(metrics["E_d0"]),
        "ensemble_acc": float(metrics["ensemble_acc"]),
        "d_star_squared_theory": (
            None
            if d_star_sq is None
            else float(d_star_sq)
        ),
        "initial_digest": initial_digest,
        "final_digest": final_digest,
        "initial_state_file": initial_state_file,
    }


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_results(results: list[dict]) -> list[dict]:
    grouped = {}

    for result in results:
        key = (
            result["lambda_rep"],
            result["lambda_div"],
            result["d0"],
            result["eta"],
            result["epochs"],
        )
        grouped.setdefault(key, []).append(result)

    summary = []

    for key, rows in sorted(grouped.items()):
        lambda_rep, lambda_div, d0, eta, epochs = key

        def values(name):
            return np.asarray(
                [r[name] for r in rows],
                dtype=float,
            )

        D = values("D_rep")
        E = values("E_d0")
        A = values("ensemble_acc")

        summary.append(
            {
                "lambda_rep": lambda_rep,
                "lambda_div": lambda_div,
                "d0": d0,
                "eta": eta,
                "epochs": epochs,
                "w": rows[0]["w"],
                "rho": rows[0]["rho"],
                "d_star_squared_theory": rows[0][
                    "d_star_squared_theory"
                ],
                "n_seeds": len(rows),
                "D_rep_mean": float(D.mean()),
                "D_rep_std": float(
                    D.std(ddof=1) if len(D) > 1 else 0.0
                ),
                "E_d0_mean": float(E.mean()),
                "E_d0_std": float(
                    E.std(ddof=1) if len(E) > 1 else 0.0
                ),
                "ensemble_acc_mean": float(A.mean()),
                "ensemble_acc_std": float(
                    A.std(ddof=1) if len(A) > 1 else 0.0
                ),
            }
        )

    return summary


# ============================================================================
# Saving
# ============================================================================

def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )


def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    tmp_path = path + ".tmp"

    fieldnames = RAW_RESULT_FIELDS

    with open(
        tmp_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, path)


# ============================================================================
# Plot
# ============================================================================

def plot_phase_diagram(
    summary: list[dict],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    d0_values = sorted(
        {float(row["d0"]) for row in summary}
    )

    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(
        figsize=(8.2, 5.9)
    )

    for index, d0 in enumerate(d0_values):
        rows = [
            row
            for row in summary
            if float(row["d0"]) == d0
        ]
        rows.sort(key=lambda row: row["rho"])

        rho = np.asarray(
            [r["rho"] for r in rows],
            dtype=float,
        )
        measured = np.asarray(
            [r["D_rep_mean"] for r in rows],
            dtype=float,
        )
        std = np.asarray(
            [r["D_rep_std"] for r in rows],
            dtype=float,
        )

        ax.errorbar(
            rho,
            measured,
            yerr=std,
            marker=markers[index % len(markers)],
            linestyle="None",
            capsize=3,
            markersize=6,
            label=rf"measured, $d_0={d0:g}$",
        )

        theory_rows = [
            r
            for r in rows
            if r["d_star_squared_theory"] is not None
        ]

        if theory_rows:
            theory_rho = np.asarray(
                [r["rho"] for r in theory_rows],
                dtype=float,
            )
            theory = np.asarray(
                [
                    r["d_star_squared_theory"]
                    for r in theory_rows
                ],
                dtype=float,
            )

            ax.plot(
                theory_rho,
                theory,
                marker="x",
                linestyle="-",
                linewidth=1.2,
                markersize=6,
                label=rf"theory, $d_0={d0:g}$",
            )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label=r"predicted transition, $\rho=1$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"$\rho=\eta d_0^2/w$"
        r",  $w=2\lambda_{\mathrm{rep}}/"
        r"\lambda_{\mathrm{div}}$"
    )
    ax.set_ylabel(
        r"empirical representation diversity $D_{\mathrm{rep}}$"
        r" (mean pairwise squared distance)"
    )
    ax.set_title("SML Phase Diagram")
    ax.grid(
        True,
        which="both",
        alpha=0.2,
    )
    ax.legend(
        fontsize=7.5,
        loc="best",
    )
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[PLOT] Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Publication-grade SML phase diagram with "
            "seed-specific common initialization and safe resume."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-cifar",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repr_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--common-initialization",
        action="store_true",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--out-dir",
        default="results",
    )
    parser.add_argument(
        "--out-name",
        default="exp2_2_2_phase_diagram",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete the existing raw CSV and start a new experiment. "
            "Initial-state files are retained but validated before use."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1.")

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.repr_dim < 1:
        raise ValueError("--repr_dim must be >= 1.")

    if args.train_samples < 1 or args.test_samples < 1:
        raise ValueError("Smoke-test sample counts must be >= 1.")

    if args.synthetic and args.smoke_cifar:
        raise ValueError(
            "--synthetic and --smoke-cifar cannot be used together."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    if args.deterministic:
        enable_deterministic_mode()

    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(
        args.out_dir,
        args.out_name + ".json",
    )
    csv_path = os.path.join(
        args.out_dir,
        args.out_name + ".csv",
    )
    summary_path = os.path.join(
        args.out_dir,
        args.out_name + "_summary.csv",
    )
    plot_path = os.path.join(
        args.out_dir,
        args.out_name + ".png",
    )

    if args.fresh and os.path.exists(csv_path):
        print(
            f"[FRESH] Removing existing raw CSV: {csv_path}"
        )
        os.remove(csv_path)

    # ------------------------------------------------------------------
    # Parameter grid
    # ------------------------------------------------------------------

    if args.synthetic or args.smoke_cifar:
        lambda_reps = SMOKE_LAMBDA_REPS
        d0s = SMOKE_D0S
        etas = SMOKE_ETAS
    else:
        lambda_reps = FULL_LAMBDA_REPS
        d0s = FULL_D0S
        etas = FULL_ETAS

    configurations = [
        (
            lambda_rep,
            LAMBDA_DIV,
            d0,
            eta,
        )
        for lambda_rep in lambda_reps
        for d0 in d0s
        for eta in etas
    ]

    total_runs = len(configurations) * args.seeds

    print("=" * 70)
    print("SML PHASE DIAGRAM EXPERIMENT")
    print("=" * 70)
    print(f"Dataset      : {args.dataset}")
    print(f"Synthetic    : {args.synthetic}")
    print(f"Smoke CIFAR  : {args.smoke_cifar}")
    print(f"Device       : {args.device}")
    print(f"Epochs       : {args.epochs}")
    print(f"Seeds        : {args.seeds}")
    print(f"repr_dim     : {args.repr_dim}")
    print(
        f"Common init  : "
        f"{args.common_initialization}"
    )
    print(
        f"Deterministic: "
        f"{args.deterministic}"
    )
    print(
        f"Configurations per seed: "
        f"{len(configurations)}"
    )
    print(
        f"Total runs: {total_runs}"
    )
    print("=" * 70)

    runtime = runtime_metadata(args.device)

    print("\nRUNTIME")
    for key, value in runtime.items():
        print(
            f"{key.replace('_', ' ').title():<22}: {value}"
        )

    # ------------------------------------------------------------------
    # Prepare common initial states
    # ------------------------------------------------------------------

    seed_initial_states = {}

    if args.common_initialization:
        print("\n" + "=" * 70)
        print("SEED-SPECIFIC COMMON INITIALIZATION")
        print("=" * 70)

        _, _, num_classes_for_init = create_loaders(
            args,
            seed=0,
        )

        for seed in range(args.seeds):
            path, digest = get_or_create_initial_state(
                output_dir=args.out_dir,
                seed=seed,
                num_classes=num_classes_for_init,
                device=args.device,
                repr_dim=args.repr_dim,
            )

            seed_initial_states[seed] = (
                path,
                digest,
            )

        print("=" * 70)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    expected_protocol = protocol_metadata(
        args,
        lambda_reps,
        d0s,
        etas,
        configurations,
    )

    results, done_keys, legacy_unknown = (
        load_existing_raw_results(
            csv_path,
            args,
            expected_protocol,
        )
    )

    # ------------------------------------------------------------------
    # Main grid
    # ------------------------------------------------------------------

    run_counter = 0
    new_run_counter = 0

    for seed in range(args.seeds):
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        # IMPORTANT:
        # This intentionally preserves the original experimental semantics:
        # one loader per seed, reused across configurations. We do NOT change
        # this now, because doing so would make the already completed runs
        # numerically non-comparable to the remaining runs.
        train_loader, test_loader, num_classes = create_loaders(
            args,
            seed=seed,
        )

        initial_state_file = None
        initial_state_digest = None

        if args.common_initialization:
            initial_state_file = seed_initial_states[seed][0]
            initial_state_digest = seed_initial_states[seed][1]

        for (
            lambda_rep,
            lambda_div,
            d0,
            eta,
        ) in configurations:

            run_counter += 1

            rho = compute_rho(
                d0=d0,
                lambda_rep=lambda_rep,
                lambda_div=lambda_div,
                eta=eta,
            )

            key = (
                float(lambda_rep),
                float(lambda_div),
                float(d0),
                float(eta),
                int(seed),
                int(args.epochs),
            )

            if key in done_keys:
                print(
                    f"[{run_counter}/{total_runs}] "
                    f"seed={seed} "
                    f"lambda_rep={lambda_rep} "
                    f"d0={d0} "
                    f"eta={eta} "
                    f"rho={rho:.4f} "
                    f"-- SKIPPED"
                )
                continue

            print(
                f"\n[{run_counter}/{total_runs}] "
                f"seed={seed} "
                f"lambda_rep={lambda_rep} "
                f"lambda_div={lambda_div} "
                f"d0={d0} "
                f"eta={eta} "
                f"rho={rho:.4f}"
            )

            result = run_one(
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                device=args.device,
                repr_dim=args.repr_dim,
                seed=seed,
                lambda_rep=lambda_rep,
                lambda_div=lambda_div,
                d0=d0,
                eta=eta,
                epochs=args.epochs,
                initial_state_file=initial_state_file,
            )

            result["run_index"] = int(run_counter)
            result["dataset"] = args.dataset
            result["synthetic"] = bool(args.synthetic)
            result["smoke_cifar"] = bool(args.smoke_cifar)
            result["repr_dim"] = int(args.repr_dim)
            result["common_initialization"] = bool(
                args.common_initialization
            )
            result["deterministic"] = bool(args.deterministic)
            result["train_samples"] = int(args.train_samples)
            result["test_samples"] = int(args.test_samples)
            result["experiment_version"] = EXPERIMENT_VERSION

            # Keep the original scientific values, while recording the
            # protocol that generated the row.
            result["initial_state_digest"] = (
                initial_state_digest
            )

            print(
                "    -> "
                f"D_rep={result['D_rep']:.6f}  "
                f"E_d0={result['E_d0']:.6f}  "
                f"d*²(theory)="
                f"{result['d_star_squared_theory']}  "
                f"acc={result['ensemble_acc']:.6f}  "
                f"initial={result['initial_digest']}  "
                f"final={result['final_digest']}"
            )

            append_raw_result(
                csv_path,
                result,
            )

            results.append(result)
            done_keys.add(key)
            new_run_counter += 1

    print(
        f"\n[RESUME] Existing/completed: "
        f"{len(results) - new_run_counter}"
    )
    print(
        f"[RESUME] New this session: "
        f"{new_run_counter}"
    )
    print(
        f"[RESUME] Total completed on disk: "
        f"{len(results)} / {total_runs}"
    )

    if legacy_unknown:
        print(
            "\n[WARNING] Some recovered rows do not record the original "
            "deterministic flag. Do not describe bitwise reproducibility "
            "for those rows unless the original command is independently "
            "confirmed."
        )

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    expected_keys = set()

    for seed in range(args.seeds):
        for (
            lambda_rep,
            lambda_div,
            d0,
            eta,
        ) in configurations:
            expected_keys.add(
                (
                    float(lambda_rep),
                    float(lambda_div),
                    float(d0),
                    float(eta),
                    int(seed),
                    int(args.epochs),
                )
            )

    actual_keys = {
        raw_result_key(row)
        for row in results
    }

    missing_keys = sorted(
        expected_keys - actual_keys
    )

    unexpected_keys = sorted(
        actual_keys - expected_keys
    )

    if unexpected_keys:
        raise RuntimeError(
            f"Raw CSV contains {len(unexpected_keys)} rows that do not "
            "belong to the current parameter grid."
        )

    print(
        f"[CHECK] Expected runs : {len(expected_keys)}"
    )
    print(
        f"[CHECK] Completed runs: {len(actual_keys)}"
    )
    print(
        f"[CHECK] Missing runs  : {len(missing_keys)}"
    )

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------

    summary = aggregate_results(results)

    experiment_metadata = {
        **expected_protocol,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "n_runs_completed": len(results),
        "n_runs_missing": len(missing_keys),
        "missing_run_keys": [
            list(key)
            for key in missing_keys
        ],
        "runtime": runtime,
        "initial_states": {
            str(seed): {
                "path": seed_initial_states[seed][0],
                "digest": seed_initial_states[seed][1],
            }
            for seed in seed_initial_states
        },
        "legacy_unknown_deterministic_metadata": (
            legacy_unknown
        ),
    }

    payload = {
        "metadata": experiment_metadata,
        "results": results,
        "summary": summary,
    }

    save_json(
        json_path,
        payload,
    )

    # Rewriting the complete raw CSV is atomic. This is safe because every
    # run was already persisted incrementally; the incremental file is the
    # recovery source, while this final rewrite normalizes the schema.
    save_csv(
        csv_path,
        results,
    )

    save_csv(
        summary_path,
        summary,
    )

    print(f"\n[SAVE] Raw JSON    : {json_path}")
    print(f"[SAVE] Raw CSV     : {csv_path}")
    print(f"[SAVE] Summary CSV : {summary_path}")

    if results:
        try:
            plot_phase_diagram(
                summary,
                plot_path,
            )
        except Exception as exc:
            print(
                "[WARNING] Plotting failed: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)
    print(
        f"Completed : {len(actual_keys)} / {len(expected_keys)}"
    )
    print(
        f"Missing   : {len(missing_keys)}"
    )
    print(
        f"Summary configurations: {len(summary)}"
    )

    if missing_keys:
        print(
            "\n[STATUS] Experiment is incomplete. "
            "Re-run the same command to resume."
        )
    else:
        print(
            "\n[STATUS] Experiment complete."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()