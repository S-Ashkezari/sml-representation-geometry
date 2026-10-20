"""
Experiment 3 -- matched baseline comparison with dataset/backbone extensions and resume support.

Fixed-backbone methods:
  Independent, DML, Rep-only, Diversity-only, SML, Diff-DML

WML:
  Four-head adapted WML implementation following the supplied official
  training mechanism. It is not claimed to reproduce the official structured
  pruning code verbatim.

All runs use:
  CIFAR-10 or CIFAR-100, identical train/test transforms from data.py,
  identical optimization settings, epoch budget, and seeds. The dataset and
  cohort backbone are explicit configuration parameters.

A checkpoint is saved after every completed epoch for every method/seed.
Power loss can therefore resume from the last completed epoch.
"""
from __future__ import annotations

import argparse
import json
import inspect
import time
from pathlib import Path
from typing import Any

import torch

from sml.train import (
    SMLConfig,
    SMLTrainer,
    save_checkpoint,
    load_checkpoint_if_valid,
)
from sml.data import (
    seed_everything,
    get_cifar_loaders,
    get_synthetic_loaders,
)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

METHODS = [
    "independent",
    "DML",
    "rep_without_diversity",
    "diversity_without_rep",
    "SML",
    "WML",
    "DiffDML",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _supported_sml_fields() -> set[str]:
    return set(getattr(SMLConfig, "__dataclass_fields__", {}).keys())


def _cohort_widths(args: argparse.Namespace):
    """Return the requested cohort-width preset. None keeps the project default."""
    if args.backbone == "default":
        return None
    if args.backbone == "wider":
        # Width-scaled version of the same cohort architecture. The exact
        # block structure remains unchanged; only channel widths are increased.
        return [64, 128, 256, 512]
    raise ValueError(f"Unknown backbone preset: {args.backbone}")


def _sml_common(args: argparse.Namespace, *, K: int | None = None,
                num_classes: int | None = None, method: str = "SML") -> dict[str, Any]:
    """Build only fields accepted by the currently installed SMLConfig."""
    values = {
        "K": args.K if K is None else K,
        "num_classes": args.num_classes if num_classes is None else num_classes,
        "repr_dim": args.repr_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_out": args.lambda_out,
        "lambda_rep": args.lambda_rep,
        "lambda_div": args.lambda_div,
        "d0": args.d0,
        "eta": args.eta,
        "T": args.T,
        "device": args.device,
        "method": method,
        "widths": _cohort_widths(args),
    }
    # Older project revisions may expose momentum/epochs; pass them only
    # when supported, so the experiment script remains version-compatible.
    if "momentum" in _supported_sml_fields():
        values["momentum"] = args.momentum
    if "epochs" in _supported_sml_fields():
        values["epochs"] = args.epochs
    return {k: v for k, v in values.items() if k in _supported_sml_fields()}


def make_config(method: str, args: argparse.Namespace) -> SMLConfig:
    """Build the trainer configuration for one baseline method."""
    common = _sml_common(args)

    if method == "independent":
        return SMLConfig(**common, use_out=False, use_rep=False,
                         use_div=False, independent=True)

    if method == "DML":
        return SMLConfig(**common, use_out=True, use_rep=False, use_div=False)

    if method == "rep_without_diversity":
        return SMLConfig(**common, use_out=True, use_rep=True, use_div=False)

    if method == "diversity_without_rep":
        return SMLConfig(**common, use_out=True, use_rep=False, use_div=True)

    if method == "SML":
        return SMLConfig(**common, use_out=True, use_rep=True, use_div=True)

    if method == "WML":
        # WML has its own fixed four-head ResNet-18 implementation. It is
        # therefore not changed by --backbone; this is recorded explicitly.
        fields = _supported_sml_fields()
        wml = {
            "K": 4,
            "num_classes": args.num_classes,
            "repr_dim": args.repr_dim,
            "lr": args.wml_lr,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "method": "WML",
            "wml_alpha": args.wml_alpha,
            "wml_eta": args.wml_eta,
            "wml_update_w_step": args.wml_update_w_step,
            "T": args.wml_T,
        }
        if "momentum" in fields:
            wml["momentum"] = args.momentum
        if "epochs" in fields:
            wml["epochs"] = args.epochs
        return SMLConfig(**{k: v for k, v in wml.items() if k in fields})

    if method == "DiffDML":
        common["method"] = "DiffDML"
        common["use_out"] = True
        common["diffdml_margin"] = args.diffdml_margin
        common["diffdml_lambda_dpl"] = args.diffdml_lambda_dpl
        common["dts_base_strength"] = args.dts_base_strength
        common["dts_step_strength"] = args.dts_step_strength
        return SMLConfig(**{k: v for k, v in common.items() if k in _supported_sml_fields()})

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(
    args: argparse.Namespace,
    seed: int,
):
    """Create loaders using the project's current loader API."""
    if args.dataset in {"cifar10", "cifar100"}:
        signature = inspect.signature(get_cifar_loaders)
        accepted = set(signature.parameters)
        kwargs: dict[str, Any] = {}
        if "root" in accepted:
            kwargs["root"] = args.data_root
        if "dataset" in accepted:
            kwargs["dataset"] = args.dataset
        if "batch_size" in accepted:
            kwargs["batch_size"] = args.batch_size
        if "num_workers" in accepted:
            kwargs["num_workers"] = args.workers
        elif "workers" in accepted:
            kwargs["workers"] = args.workers
        if "seed" in accepted:
            kwargs["seed"] = seed
        return get_cifar_loaders(**kwargs)

    if args.dataset == "synthetic":
        return get_synthetic_loaders(
            num_classes=args.num_classes,
            n_per_class_train=args.synthetic_train_per_class,
            n_per_class_test=args.synthetic_test_per_class,
            batch_size=args.batch_size,
            seed=seed,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


# ---------------------------------------------------------------------------
# JSON handling
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    """Load an existing result file or create an empty result structure."""

    if not path.exists():
        return {
            "config": {},
            "results": {},
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Result file is not valid JSON: {path}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Could not read result file: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Invalid result-file structure: {path}"
        )

    if not isinstance(data.get("results", {}), dict):
        raise RuntimeError(
            f"'results' must be a dictionary in: {path}"
        )

    data.setdefault("config", {})
    data.setdefault("results", {})

    return data


def atomic_json_dump(
    obj: dict[str, Any],
    path: Path,
) -> None:
    """
    Atomically write the JSON result file.

    The temporary file is placed next to the destination so that replace()
    remains atomic on normal local filesystems.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with open(
            tmp,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                obj,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.flush()

        tmp.replace(path)

    except Exception:
        try:
            tmp.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def normalize_method_result(
    data: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """
    Ensure that data['results'][method] exists and has a valid structure.

    This is important for old/incomplete result files. In particular, it
    prevents the original:

        KeyError: 'DML'
    """

    results = data.setdefault(
        "results",
        {}
    )

    entry = results.get(
        method
    )

    if not isinstance(
        entry,
        dict,
    ):
        entry = {}
        results[method] = entry

    seeds = entry.get(
        "seeds",
        []
    )

    if not isinstance(
        seeds,
        list,
    ):
        seeds = []

    entry["seeds"] = [
        row
        for row in seeds
        if isinstance(
            row,
            dict,
        )
    ]

    return entry


def initialize_results(
    data: dict[str, Any],
) -> None:
    """
    Initialize every declared method before any result processing.
    """

    for method in METHODS:
        normalize_method_result(
            data,
            method,
        )


def sanitize_seed_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove malformed/duplicate seed records.

    If duplicate records exist for the same seed, the last record wins.
    """

    by_seed: dict[int, dict[str, Any]] = {}

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if "seed" not in row:
            continue

        try:
            seed = int(
                row["seed"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        by_seed[seed] = row

    return [
        by_seed[seed]
        for seed in sorted(by_seed)
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate numeric metrics over seed results.

    List-valued metrics are averaged element-wise when all arrays have the
    same length.
    """

    rows = sanitize_seed_rows(
        rows
    )

    if not rows:
        return {}

    out: dict[str, Any] = {}

    keys = set().union(
        *(row.keys() for row in rows)
    )

    for key in keys:
        values = [
            row[key]
            for row in rows
            if isinstance(
                row.get(key),
                (int, float),
            )
            and not isinstance(
                row.get(key),
                bool,
            )
        ]

        if values:
            out[key] = (
                sum(values)
                / len(values)
            )

    for key in (
        "per_model_acc",
        "wml_weights",
    ):
        arrays = [
            row[key]
            for row in rows
            if isinstance(
                row.get(key),
                list,
            )
        ]

        if arrays and all(
            len(array) == len(arrays[0])
            for array in arrays
        ):
            out[key] = [
                sum(
                    array[i]
                    for array in arrays
                )
                / len(arrays)
                for i in range(
                    len(arrays[0])
                )
            ]

    return out


# ---------------------------------------------------------------------------
# Configuration integrity
# ---------------------------------------------------------------------------

def build_config_snapshot(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Build the experiment configuration used to detect accidental mixing of
    different experiments.

    Output/checkpoint paths are excluded because changing paths does not
    change the experiment itself.
    """

    config = vars(
        args
    ).copy()

    config.pop(
        "output",
        None,
    )

    config.pop(
        "checkpoint_dir",
        None,
    )

    config.pop(
        "allow_config_mismatch",
        None,
    )

    return config


def check_existing_config(
    data: dict[str, Any],
    current_config: dict[str, Any],
    output: Path,
    allow_mismatch: bool,
) -> None:
    """
    Detect configuration changes before resuming an existing result file.

    This prevents accidentally mixing results produced with different
    hyperparameters.
    """

    previous = data.get(
        "config_snapshot"
    )

    if not previous:
        previous = data.get(
            "config"
        )

    if not previous:
        return

    if not isinstance(
        previous,
        dict,
    ):
        raise RuntimeError(
            f"Invalid configuration in result file: {output}"
        )

    differences = {}

    for key, current_value in current_config.items():
        if (
            key in previous
            and previous[key] != current_value
        ):
            differences[key] = (
                previous[key],
                current_value,
            )

    if not differences:
        return

    lines = [
        (
            "Existing result file has a different configuration: "
            f"{output}"
        ),
        "",
        "Changed parameters:",
    ]

    for key, (old, new) in differences.items():
        lines.append(
            f"  {key}: existing={old!r}, current={new!r}"
        )

    lines.extend(
        [
            "",
            (
                "Use a new --output and/or --checkpoint-dir for a "
                "new experiment."
            ),
            (
                "Alternatively, explicitly use "
                "--allow-config-mismatch if mixing configurations "
                "is intentional."
            ),
        ]
    )

    message = "\n".join(
        lines
    )

    if not allow_mismatch:
        raise RuntimeError(
            message
        )

    print(
        "[WARNING] Existing configuration differs from the current "
        "configuration."
    )

    for key, (old, new) in differences.items():
        print(
            f"[WARNING]   {key}: "
            f"existing={old!r}, current={new!r}"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    method: str,
    args: argparse.Namespace,
    seed: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """
    Train one method/seed pair with epoch-level checkpoint resume.
    """

    seed_everything(
        seed
    )

    train_loader, test_loader, num_classes = make_loaders(
        args,
        seed,
    )

    cfg = make_config(
        method,
        args,
    )

    # The dataset is authoritative.
    cfg.num_classes = num_classes

    trainer = SMLTrainer(
        cfg
    )

    method_checkpoint_dir = (
        checkpoint_dir
        / method.replace(
            "/",
            "_",
        )
    )

    ckpt = (
        method_checkpoint_dir
        / f"seed_{seed}.pt"
    )

    resumed = load_checkpoint_if_valid(
        trainer,
        cfg,
        seed,
        args.dataset,
        ckpt,
    )

    start_epoch = int(
        getattr(
            trainer,
            "completed_epochs",
            0,
        )
    )

    if start_epoch < 0:
        raise RuntimeError(
            f"Invalid negative completed_epochs={start_epoch} "
            f"for {method}, seed={seed}."
        )

    if start_epoch > args.epochs:
        raise RuntimeError(
            f"Checkpoint for {method}, seed={seed} reports "
            f"{start_epoch} completed epochs, but the requested run "
            f"has only {args.epochs} epochs."
        )

    if resumed:
        print(
            f"[RESUME] {method} seed={seed}: "
            f"continuing from epoch {start_epoch + 1}"
        )

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        total = 0.0

        for x, y in train_loader:
            logs, _, _ = trainer.step(
                x,
                y,
            )

            if (
                not isinstance(
                    logs,
                    dict,
                )
                or "total" not in logs
            ):
                raise RuntimeError(
                    f"{method} seed={seed}: trainer.step() did not "
                    "return a log dictionary containing 'total'."
                )

            total += float(
                logs["total"]
            )

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            trainer.scheduler.step()

        trainer.completed_epochs = (
            epoch + 1
        )

        save_checkpoint(
            trainer,
            cfg,
            seed,
            args.dataset,
            ckpt,
        )

        print(
            f"[{method}] "
            f"seed={seed} "
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={total / max(len(train_loader), 1):.4f} "
            f"lr={trainer.optim.param_groups[0]['lr']:.6g}"
        )

    metrics = trainer.evaluate(
        test_loader
    )

    if not isinstance(
        metrics,
        dict,
    ):
        raise RuntimeError(
            f"{method} seed={seed}: trainer.evaluate() did not "
            "return a dictionary."
        )

    metrics["seed"] = seed

    metrics["epochs_completed"] = int(
        trainer.completed_epochs
    )

    metrics["resumed"] = bool(
        resumed
    )
    metrics["training_seconds"] = float(time.perf_counter() - train_start)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        metrics["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        metrics["peak_gpu_memory_mb"] = None
    metrics["backbone"] = args.backbone
    metrics["dataset"] = args.dataset

    return metrics


# ---------------------------------------------------------------------------
# Result update
# ---------------------------------------------------------------------------

def update_method_result(
    data: dict[str, Any],
    method: str,
    metrics: dict[str, Any],
) -> None:
    """
    Insert or replace one seed result and recompute the method mean.
    """

    entry = normalize_method_result(
        data,
        method,
    )

    seed = int(
        metrics["seed"]
    )

    rows = [
        row
        for row in entry["seeds"]
        if int(
            row.get(
                "seed",
                -1,
            )
        ) != seed
    ]

    rows.append(
        metrics
    )

    rows = sanitize_seed_rows(
        rows
    )

    entry["seeds"] = rows

    entry["mean"] = aggregate(
        rows
    )


def update_interaction_gains(
    data: dict[str, Any],
) -> None:
    """
    Compute G_int relative to the independent baseline.

    For each method:

        G_int = mean_s [Acc_method(s) - Acc_independent(s)]

    Only seeds for which both accuracies are available are used.
    """

    results = data.setdefault(
        "results",
        {}
    )

    independent_entry = results.get(
        "independent"
    )

    if not isinstance(
        independent_entry,
        dict,
    ):
        return

    independent_rows = (
        independent_entry.get(
            "seeds",
            [],
        )
    )

    if not isinstance(
        independent_rows,
        list,
    ):
        return

    independent_by_seed: dict[int, float] = {}

    for row in independent_rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if (
            "seed" not in row
            or "ensemble_acc" not in row
        ):
            continue

        try:
            seed = int(
                row["seed"]
            )

            accuracy = float(
                row["ensemble_acc"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        independent_by_seed[seed] = accuracy

    if not independent_by_seed:
        return

    for method in METHODS:
        if method == "independent":
            continue

        entry = results.get(
            method
        )

        if not isinstance(
            entry,
            dict,
        ):
            continue

        rows = entry.get(
            "seeds",
            [],
        )

        if not isinstance(
            rows,
            list,
        ):
            continue

        gains = []

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            if (
                "seed" not in row
                or "ensemble_acc" not in row
            ):
                continue

            try:
                seed = int(
                    row["seed"]
                )

                accuracy = float(
                    row["ensemble_acc"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            if seed in independent_by_seed:
                gains.append(
                    accuracy
                    - independent_by_seed[seed]
                )

        if gains:
            entry.setdefault(
                "mean",
                {}
            )

            entry["mean"]["G_int"] = (
                sum(gains)
                / len(gains)
            )

        else:
            # Remove stale G_int values from old result files.
            if isinstance(
                entry.get("mean"),
                dict,
            ):
                entry["mean"].pop(
                    "G_int",
                    None,
                )


def get_completed_seeds(
    data: dict[str, Any],
    method: str,
    epochs: int,
) -> set[int]:
    """
    Return seeds whose final result is already stored.
    """

    entry = normalize_method_result(
        data,
        method,
    )

    completed = set()

    for row in entry["seeds"]:
        if "seed" not in row:
            continue

        try:
            seed = int(
                row["seed"]
            )

            epochs_completed = int(
                row.get(
                    "epochs_completed",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if epochs_completed >= epochs:
            completed.add(
                seed
            )

    return completed


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        default="./data",
    )

    p.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar100",
            "synthetic",
        ],
        default="cifar10",
    )

    p.add_argument(
        "--num-classes",
        type=int,
        default=10,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    p.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--K",
        type=int,
        default=3,
    )

    p.add_argument(
        "--repr_dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--backbone",
        choices=["default", "wider"],
        default="default",
        help=(
            "Cohort backbone preset. 'default' uses the project default; "
            "'wider' uses the same block structure with wider channels. "
            "WML retains its fixed four-head ResNet-18 implementation."
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    p.add_argument(
        "--output",
        default="results/exp3_baseline_comparison_extended.json",
    )

    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/exp3_extended",
    )

    p.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "Allow an existing result/checkpoint set to be resumed "
            "despite configuration differences."
        ),
    )

    # Shared strong CIFAR-10 recipe.
    p.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    # Shared method coefficients.
    p.add_argument(
        "--lambda-out",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--lambda-rep",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--lambda-div",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--d0",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--T",
        type=float,
        default=4.0,
    )

    # WML.
    p.add_argument(
        "--wml-lr",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--wml-T",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--wml-alpha",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--wml-eta",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--wml-update-w-step",
        type=int,
        default=10,
    )

    # Diff-DML.
    p.add_argument(
        "--diffdml-margin",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--diffdml-lambda-dpl",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--dts-base-strength",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--dts-step-strength",
        type=float,
        default=0.15,
    )

    # Synthetic smoke-test size.
    p.add_argument(
        "--synthetic-train-per-class",
        type=int,
        default=200,
    )

    p.add_argument(
        "--synthetic-test-per-class",
        type=int,
        default=40,
    )

    return p.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate arguments before training starts."""

    if args.K < 2:
        raise ValueError(
            "--K must be >= 2"
        )

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1"
        )

    if args.seeds < 1:
        raise ValueError(
            "--seeds must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.workers < 0:
        raise ValueError(
            "--workers must be >= 0"
        )

    if args.repr_dim < 1:
        raise ValueError(
            "--repr_dim must be >= 1"
        )

    if args.num_classes < 2:
        raise ValueError(
            "--num-classes must be >= 2"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    output = Path(
        args.output
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    data = load_json(
        output
    )

    current_config = build_config_snapshot(
        args
    )

    # Validate the existing result file BEFORE modifying it.
    check_existing_config(
        data,
        current_config,
        output,
        args.allow_config_mismatch,
    )

    # Store both the full command-line configuration and the normalized
    # experiment snapshot.
    data["config"] = vars(
        args
    ).copy()

    data["config_snapshot"] = (
        current_config
    )

    # ------------------------------------------------------------------
    # IMPORTANT FIX FOR THE ORIGINAL KeyError: 'DML'
    #
    # Initialize every method before accessing data["results"][method].
    # This also repairs old/incomplete JSON files.
    # ------------------------------------------------------------------

    initialize_results(
        data
    )

    # Clean malformed/duplicate rows and recompute means.
    for method in METHODS:
        entry = normalize_method_result(
            data,
            method,
        )

        entry["seeds"] = sanitize_seed_rows(
            entry["seeds"]
        )

        entry["mean"] = aggregate(
            entry["seeds"]
        )

    # Save repaired result structure before training.
    atomic_json_dump(
        data,
        output,
    )

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------

    for method in METHODS:
        completed = get_completed_seeds(
            data,
            method,
            args.epochs,
        )

        for seed in range(
            args.seeds
        ):
            if seed in completed:
                print(
                    f"[SKIP] {method} seed={seed}: "
                    "already complete"
                )
                continue

            print(
                f"[RUN] {method} seed={seed}"
            )

            metrics = train_one(
                method,
                args,
                seed,
                checkpoint_dir,
            )

            epochs_completed = int(
                metrics.get(
                    "epochs_completed",
                    0,
                )
            )

            if epochs_completed < args.epochs:
                raise RuntimeError(
                    f"{method} seed={seed} returned before completing "
                    f"the requested {args.epochs} epochs."
                )

            # Insert/replace this seed result.
            update_method_result(
                data,
                method,
                metrics,
            )

            # Recompute G_int using all currently available independent
            # baseline results.
            update_interaction_gains(
                data
            )

            # Save after EVERY seed.
            atomic_json_dump(
                data,
                output,
            )

            print(
                f"[SAVED] {method} seed={seed} -> {output}"
            )

    # ------------------------------------------------------------------
    # Final consistency pass
    # ------------------------------------------------------------------

    for method in METHODS:
        entry = normalize_method_result(
            data,
            method,
        )

        entry["seeds"] = sanitize_seed_rows(
            entry["seeds"]
        )

        entry["mean"] = aggregate(
            entry["seeds"]
        )

    update_interaction_gains(
        data
    )

    atomic_json_dump(
        data,
        output,
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()