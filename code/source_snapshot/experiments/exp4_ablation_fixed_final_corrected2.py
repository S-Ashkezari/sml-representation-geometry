"""
exp4_ablation.py
================

Main ablation study for SML.

Purpose
-------
Evaluate the sensitivity of SML to the four principal loss/control
parameters:

    lambda_out
    lambda_rep
    lambda_div
    d0

One parameter is varied at a time while all other parameters are kept
at the common operating point.

The operating point is aligned with Exp3-A:

    lambda_out = 0.5
    lambda_rep = 0.1
    lambda_div = 0.1
    d0         = 1.0
    eta        = 1.0

The ablation is primarily intended for CIFAR-10. Synthetic data can
still be used for a quick reproducibility/debugging run.

Recommended paper run
---------------------
    python -m experiments.exp4_ablation_fixed \
        --dataset cifar \
        --epochs 50 \
        --seeds 3 \
        --device cuda

Quick smoke test
----------------
    python -m experiments.exp4_ablation_fixed \
        --dataset synthetic \
        --epochs 2 \
        --seeds 1 \
        --device cpu

Output
------
    results/exp4_ablation_cifar10_compact.json

The JSON contains:
    - experiment configuration
    - parameter grids
    - individual runs
    - aggregated mean/std results
    - operating-point information
    - reproducibility information
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from sml.train import SMLConfig, SMLTrainer
from sml.data import (
    seed_everything,
    get_cifar_loaders,
    get_synthetic_loaders,
)


# ---------------------------------------------------------------------
# Common operating point
# ---------------------------------------------------------------------

DEFAULTS: Dict[str, float] = {
    "lambda_out": 0.5,
    "lambda_rep": 0.1,
    "lambda_div": 0.1,
    "d0": 1.0,
    "eta": 1.0,
}

# Matched optimization protocol from Exp3. These are fixed throughout Exp4.
OPTIMIZATION: Dict[str, float] = {
    "T": 4.0,
    "lr": 0.1,
    "weight_decay": 5e-4,
}


# ---------------------------------------------------------------------
# Ablation grids
# ---------------------------------------------------------------------
#
# The default value is included in every sweep.
#
# The grids are deliberately compact because this is a supporting
# ablation rather than the main mechanistic experiment. With three values
# per parameter and three seeds, the study contains 36 training runs.
# ---------------------------------------------------------------------

GRIDS: Dict[str, List[float]] = {
    # Compact one-factor-at-a-time grids. The operating point is retained
    # in every sweep. Three seeds are used because Exp4 is a supporting
    # sensitivity study; Exp3 remains the main matched comparison.
    "lambda_out": [0.0, 0.5, 1.0],
    "lambda_rep": [0.0, 0.1, 0.2],
    "lambda_div": [0.05, 0.1, 0.2],
    "d0": [0.5, 1.0, 1.5],
}


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """Convert common non-JSON-native objects to JSON-safe values."""

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(data),
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp_path.replace(path)


def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    """Load an existing JSON file if available."""

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------

def enable_deterministic_mode() -> None:
    """
    Enable deterministic PyTorch behavior as far as supported.

    This is intentionally aligned with the reproducibility settings
    used in the other experiments.
    """

    torch.use_deterministic_algorithms(True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

def make_loaders(
    dataset: str,
    seed: int,
    batch_size: int,
    workers: int,
    train_samples: int | None,
    test_samples: int | None,
) -> Tuple[Any, Any, int]:
    """
    Construct loaders using the current project data API.

    Supported datasets:
        synthetic
        cifar
    """

    if dataset == "synthetic":

        kwargs: Dict[str, Any] = {
            "num_classes": 10,
            "n_per_class_train": 64,
            "n_per_class_test": 32,
            "batch_size": batch_size,
            "seed": seed,
        }

        # Keep this branch compatible with the current synthetic
        # loader. train_samples/test_samples are intentionally not
        # forced here because the synthetic loader is already
        # controlled through n_per_class_*.

        return get_synthetic_loaders(**kwargs)

    if dataset == "cifar":
        # Use the same CIFAR-10 train/test loader API as Exp3 so that
        # Exp4 differs from the matched comparison only in the ablated
        # SML parameter. This avoids confounding the ablation with a
        # different augmentation, split, or preprocessing pipeline.
        return get_cifar_loaders(
            root="./data",
            dataset="cifar10",
            batch_size=batch_size,
            num_workers=workers,
        )

    raise ValueError(
        f"Unsupported dataset '{dataset}'. "
        f"Expected 'cifar' or 'synthetic'."
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def build_config(
    params: Dict[str, float],
    num_classes: int,
    repr_dim: int,
    device: str,
    K: int,
    epochs: int,
) -> SMLConfig:
    """Build the SML configuration for one ablation setting."""

    return SMLConfig(
        K=K,
        num_classes=num_classes,
        repr_dim=repr_dim,

        use_out=True,
        use_rep=True,
        use_div=True,

        device=device,

        # Exactly match Exp3 optimization settings.
        T=OPTIMIZATION["T"],
        lr=OPTIMIZATION["lr"],
        momentum=0.9,
        weight_decay=OPTIMIZATION["weight_decay"],
        epochs=epochs,

        lambda_out=params["lambda_out"],
        lambda_rep=params["lambda_rep"],
        lambda_div=params["lambda_div"],
        d0=params["d0"],
        eta=params["eta"],
    )


# ---------------------------------------------------------------------
# Model digest
# ---------------------------------------------------------------------

def model_digest(trainer: SMLTrainer) -> str:
    """
    Compute a SHA-256 digest of the complete model state.

    This is used only as a reproducibility fingerprint.
    """

    import hashlib

    state = trainer.models.state_dict()

    h = hashlib.sha256()

    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous()

        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())

    return h.hexdigest()


# ---------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------

def run_one(
    train_loader: Any,
    test_loader: Any,
    num_classes: int,
    params: Dict[str, float],
    epochs: int,
    device: str,
    seed: int,
    repr_dim: int,
    K: int,
    scheduler: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate one ablation configuration.

    Important:
    The loaders are supplied from outside so that, for a fixed seed,
    all parameter settings use the same data split/loaders.
    """

    seed_everything(seed)

    cfg = build_config(
        params=params,
        num_classes=num_classes,
        repr_dim=repr_dim,
        device=device,
        K=K,
        epochs=epochs,
    )

    trainer = SMLTrainer(
        cfg,
        in_channels=3,
    )

    initial_digest = model_digest(trainer)

    for epoch in range(epochs):

        for x, y in train_loader:
            trainer.step(x, y)

        # Exp3 uses the trainer's cosine scheduler after every epoch.
        if scheduler and hasattr(trainer, "scheduler"):
            trainer.scheduler.step()

    metrics = trainer.evaluate(test_loader)

    result: Dict[str, Any] = {
        "seed": int(seed),
        "epochs": int(epochs),

        "lambda_out": float(params["lambda_out"]),
        "lambda_rep": float(params["lambda_rep"]),
        "lambda_div": float(params["lambda_div"]),
        "d0": float(params["d0"]),
        "eta": float(params["eta"]),

        "initial_digest": initial_digest,
        "final_digest": model_digest(trainer),
    }

    # Keep all metrics returned by the current evaluator.
    for key, value in metrics.items():
        result[key] = json_safe(value)

    return result


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return sample mean and sample standard deviation."""

    clean = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not clean:
        return float("nan"), float("nan")

    mean = sum(clean) / len(clean)

    if len(clean) == 1:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)

    return mean, math.sqrt(variance)


def aggregate_one_group(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate all seeds belonging to one parameter setting.
    """

    if not runs:
        raise ValueError("Cannot aggregate an empty group.")

    metrics = [
        "ensemble_acc",
        "best_single_acc",
        "D_rep",
        "E_d0",
        "representation_dispersion",
        "prediction_disagreement_rate",
        "prediction_symmetric_kl",
    ]

    first = runs[0]

    result: Dict[str, Any] = {
        "lambda_out": first["lambda_out"],
        "lambda_rep": first["lambda_rep"],
        "lambda_div": first["lambda_div"],
        "d0": first["d0"],
        "eta": first["eta"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
    }

    for metric in metrics:

        values = [
            r[metric]
            for r in runs
            if metric in r and r[metric] is not None
        ]

        if not values:
            continue

        mean, std = mean_std(values)

        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std

    return result


def aggregate_results(
    runs: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate results separately for each swept parameter.
    """

    grouped: Dict[str, Dict[Tuple[float, ...], List[Dict[str, Any]]]] = {}

    for run in runs:

        swept_param = run["swept_param"]

        if swept_param not in grouped:
            grouped[swept_param] = {}

        key = (
            float(run["lambda_out"]),
            float(run["lambda_rep"]),
            float(run["lambda_div"]),
            float(run["d0"]),
            float(run["eta"]),
        )

        grouped[swept_param].setdefault(key, []).append(run)

    summary: Dict[str, List[Dict[str, Any]]] = {}

    for swept_param, groups in grouped.items():

        rows = [
            aggregate_one_group(group)
            for group in groups.values()
        ]

        # Sort according to the swept parameter.
        rows.sort(
            key=lambda r: float(r[swept_param])
        )

        summary[swept_param] = rows

    return summary


# ---------------------------------------------------------------------
# Completed-run handling
# ---------------------------------------------------------------------

def run_key(
    swept_param: str,
    value: float,
    seed: int,
) -> str:
    """
    Unique identifier for one ablation run.
    """

    return (
        f"{swept_param}="
        f"{float(value):.12g}|seed={int(seed)}"
    )


def build_completed_keys(
    runs: List[Dict[str, Any]],
) -> set[str]:
    """Build a set of completed run identifiers."""

    completed: set[str] = set()

    for run in runs:

        if "swept_param" not in run:
            continue

        if "swept_value" not in run:
            continue

        if "seed" not in run:
            continue

        key = run_key(
            run["swept_param"],
            float(run["swept_value"]),
            int(run["seed"]),
        )

        completed.add(key)

    return completed


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_grids() -> None:
    """Validate the ablation design."""

    required = {
        "lambda_out",
        "lambda_rep",
        "lambda_div",
        "d0",
    }

    if set(GRIDS.keys()) != required:
        raise ValueError(
            "GRIDS must contain exactly: "
            "lambda_out, lambda_rep, lambda_div, d0."
        )

    for parameter, default_value in DEFAULTS.items():

        if parameter == "eta":
            continue

        if parameter not in GRIDS:
            raise ValueError(
                f"Missing grid for parameter '{parameter}'."
            )

        if default_value not in GRIDS[parameter]:
            raise ValueError(
                f"Default value {default_value} for '{parameter}' "
                f"is not included in its ablation grid."
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="SML one-at-a-time ablation study."
    )

    parser.add_argument(
        "--dataset",
        default="cifar",
        choices=["cifar", "synthetic"],
        help="Dataset. Use 'cifar' for CIFAR-10.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of independent seeds.",
    )

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--train-samples",
        "--train_samples",
        dest="train_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--test-samples",
        "--test_samples",
        dest="test_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--K",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default="results/exp4_ablation_cifar10_compact.json",
    )

    args = parser.parse_args()

    validate_grids()

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1.")

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.deterministic:
        enable_deterministic_mode()

    output_path = Path(args.output)

    # ---------------------------------------------------------------
    # Load previous results if available.
    # ---------------------------------------------------------------

    existing = load_json_if_exists(output_path)

    if existing is not None:

        runs: List[Dict[str, Any]] = existing.get(
            "runs",
            [],
        )

        print(
            f"[resume] Found {len(runs)} existing run(s) "
            f"in {output_path}"
        )

    else:

        runs = []

    completed = build_completed_keys(runs)

    # ---------------------------------------------------------------
    # Experiment metadata
    # ---------------------------------------------------------------

    experiment_metadata = {
        "experiment": "Exp4-Ablation",
        "description": (
            "One-at-a-time ablation of the four principal "
            "SML control parameters."
        ),

        "dataset": args.dataset,

        "epochs": args.epochs,
        "seeds": args.seeds,

        "batch_size": args.batch_size,
        "workers": args.workers,

        "K": args.K,
        "repr_dim": args.repr_dim,

        "device": args.device,
        "deterministic": args.deterministic,

        "defaults": DEFAULTS,
        "optimization": OPTIMIZATION,

        "grids": GRIDS,

        # eta is intentionally fixed.
        "eta_fixed": DEFAULTS["eta"],

        "protocol": "one-factor-at-a-time",
    }

    # ---------------------------------------------------------------
    # Main ablation loop
    # ---------------------------------------------------------------

    total_runs = sum(
        len(grid) * args.seeds
        for grid in GRIDS.values()
    )

    completed_before = len(completed)

    print()
    print("=" * 72)
    print("Exp4 — SML Ablation Study")
    print("=" * 72)
    print(f"Dataset       : {args.dataset}")
    print(f"Epochs        : {args.epochs}")
    print(f"Seeds         : {args.seeds}")
    print(f"K             : {args.K}")
    print(f"repr_dim      : {args.repr_dim}")
    print(f"Device        : {args.device}")
    print(f"Total runs    : {total_runs}")
    print()
    print("Fixed operating point:")
    print(
        f"  lambda_out = {DEFAULTS['lambda_out']}"
    )
    print(
        f"  lambda_rep = {DEFAULTS['lambda_rep']}"
    )
    print(
        f"  lambda_div = {DEFAULTS['lambda_div']}"
    )
    print(
        f"  d0         = {DEFAULTS['d0']}"
    )
    print(
        f"  eta        = {DEFAULTS['eta']}"
    )
    print("=" * 72)
    print()

    # ---------------------------------------------------------------
    # Fresh loaders for every run.
    #
    # DataLoader state (including shuffling/worker RNG state) changes
    # after iteration. Reusing one loader across ablation settings can
    # therefore make identical operating points follow different batch
    # orders. Construct fresh loaders for every run while keeping the
    # same seed, split, and loader configuration.
    # ---------------------------------------------------------------

    def get_fresh_loaders(seed: int):
        print(
            f"[data] Creating fresh loaders for seed={seed}"
        )

        seed_everything(seed)

        return make_loaders(
            dataset=args.dataset,
            seed=seed,
            batch_size=args.batch_size,
            workers=args.workers,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
        )

    # ---------------------------------------------------------------
    # Sweep each parameter independently.
    # ---------------------------------------------------------------

    for swept_param, grid in GRIDS.items():

        print()
        print("-" * 72)
        print(f"Sweeping {swept_param}")
        print(f"Values: {grid}")
        print("-" * 72)

        for value in grid:

            params = dict(DEFAULTS)
            params[swept_param] = float(value)

            for seed in range(args.seeds):

                key = run_key(
                    swept_param,
                    value,
                    seed,
                )

                if key in completed:

                    print(
                        f"[skip] {swept_param}={value} "
                        f"seed={seed}"
                    )

                    continue

                (
                    train_loader,
                    test_loader,
                    num_classes,
                ) = get_fresh_loaders(seed)

                print(
                    f"[ablation] "
                    f"{swept_param}={value} "
                    f"seed={seed}"
                )

                result = run_one(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_classes=num_classes,
                    params=params,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    repr_dim=args.repr_dim,
                    K=args.K,
                )

                result["swept_param"] = swept_param
                result["swept_value"] = float(value)

                runs.append(result)
                completed.add(key)

                ens_acc = result.get(
                    "ensemble_acc",
                    float("nan"),
                )

                d_rep = result.get(
                    "D_rep",
                    float("nan"),
                )

                e_d0 = result.get(
                    "E_d0",
                    float("nan"),
                )

                disagree = result.get(
                    "prediction_disagreement_rate",
                    float("nan"),
                )

                print(
                    f"    "
                    f"ens_acc={ens_acc:.4f}  "
                    f"D_rep={d_rep:.4f}  "
                    f"E_d0={e_d0:.4f}  "
                    f"disagree={disagree:.4f}"
                )

                # ---------------------------------------------------
                # Incremental save.
                # ---------------------------------------------------

                summary = aggregate_results(runs)

                output = {
                    "experiment": experiment_metadata,
                    "runs": runs,
                    "summary": summary,
                    "num_runs_completed": len(runs),
                    "num_runs_expected": total_runs,
                }

                save_json(
                    output_path,
                    output,
                )

    # ---------------------------------------------------------------
    # Final aggregation
    # ---------------------------------------------------------------

    summary = aggregate_results(runs)

    final_output = {
        "experiment": experiment_metadata,
        "runs": runs,
        "summary": summary,

        "num_runs_completed": len(runs),
        "num_runs_expected": total_runs,

        "complete": len(runs) >= total_runs,
    }

    save_json(
        output_path,
        final_output,
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL ABLATION SUMMARY")
    print("=" * 72)

    for swept_param, rows in summary.items():

        print()
        print(f"[{swept_param}]")

        for row in rows:

            value = row[swept_param]

            acc_mean = row.get(
                "ensemble_acc_mean",
                float("nan"),
            )

            acc_std = row.get(
                "ensemble_acc_std",
                float("nan"),
            )

            drep_mean = row.get(
                "D_rep_mean",
                float("nan"),
            )

            drep_std = row.get(
                "D_rep_std",
                float("nan"),
            )

            print(
                f"  {swept_param}={value:<6} "
                f"Acc={acc_mean:.4f}±{acc_std:.4f}  "
                f"D_rep={drep_mean:.4f}±{drep_std:.4f}"
            )

    print()
    print("-" * 72)

    if len(runs) >= total_runs:
        print(
            f"Status: COMPLETE "
            f"({len(runs)}/{total_runs} runs)"
        )
    else:
        print(
            f"Status: INCOMPLETE "
            f"({len(runs)}/{total_runs} runs)"
        )

    print(
        f"Saved results to: {output_path}"
    )

    if completed_before:
        print(
            f"Resumed from {completed_before} existing run(s)."
        )


if __name__ == "__main__":
    main()