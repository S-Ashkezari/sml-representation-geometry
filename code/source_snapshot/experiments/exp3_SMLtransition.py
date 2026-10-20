"""
experiments/exp3_SMLtransition.py
=================================

Exp3-A: SML eta-transition experiment.

Goal
----
Validate the predicted finite-target representation-diversity transition
by sweeping eta while keeping the remaining SML parameters fixed.

Default:
    lambda_rep = 0.1
    lambda_div = 0.1
    d0         = 1.0

Therefore:
    w   = 2 * lambda_rep / lambda_div = 2
    rho = eta * d0^2 / w              = eta / 2

Theoretical equilibrium:
    d_star^2 = max(0, d0^2 - w / eta)

Eta sweep:
    [0.5, 1.0, 2.0, 3.0, 4.0, 8.0]

Default experiment:
    6 eta values x 5 seeds = 30 runs

For each seed, all eta configurations start from exactly the same
initial cohort of K models.

Optimizer state is NOT shared between eta configurations.

Plotting
--------
The script can also generate the final two-panel publication figure:

    figures/exp3a_transition.pdf

using the existing Synthetic and CIFAR-10 result JSON files.

Plot only:
    python -m experiments.exp3_SMLtransition --plot-only
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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


# =====================================================================
# EXPERIMENT CONSTANTS
# =====================================================================

ETAS = [
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    8.0,
]


# =====================================================================
# JSON
# =====================================================================

def json_safe(obj: Any) -> Any:

    if obj is None:
        return None

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, int):
        return obj

    if isinstance(obj, float):

        if math.isnan(obj) or math.isinf(obj):
            return None

        return obj

    if isinstance(obj, str):
        return obj

    if isinstance(obj, torch.Tensor):

        if obj.numel() == 1:
            return json_safe(
                obj.detach().cpu().item()
            )

        return obj.detach().cpu().tolist()

    if isinstance(obj, dict):

        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):

        return [
            json_safe(v)
            for v in obj
        ]

    if hasattr(obj, "item"):

        try:
            return json_safe(
                obj.item()
            )
        except Exception:
            pass

    return str(obj)


def save_json(
    data: Dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(data),
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp.replace(path)


def load_json_if_exists(
    path: Path,
) -> Dict[str, Any] | None:

    if not path.exists():
        return None

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception as exc:

        print(
            f"[WARN] Could not load "
            f"{path}: {exc}"
        )

        return None


# =====================================================================
# DETERMINISM
# =====================================================================

def enable_deterministic_mode() -> None:

    try:

        torch.use_deterministic_algorithms(
            True
        )

        print(
            "[SETUP] "
            "torch.use_deterministic_algorithms(True)"
        )

    except Exception as exc:

        print(
            "[WARN] Deterministic algorithms "
            f"could not be enabled: {exc}"
        )

    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


# =====================================================================
# DATA LOADER ADAPTER
# =====================================================================

def _call_loader_adaptively(
    loader_fn,
    candidate_kwargs: Dict[str, Any],
):
    """
    Call a data-loader function using only arguments supported by its
    actual Python signature.
    """

    signature = inspect.signature(
        loader_fn
    )

    parameters = signature.parameters

    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )

    if accepts_kwargs:

        kwargs = dict(
            candidate_kwargs
        )

    else:

        kwargs = {
            name: value
            for name, value in candidate_kwargs.items()
            if name in parameters
        }

    print(
        f"[DATA] {loader_fn.__name__} "
        f"using arguments: "
        f"{sorted(kwargs.keys())}"
    )

    return loader_fn(
        **kwargs
    )


def make_loaders(
    args: argparse.Namespace,
    seed: int,
):
    """
    Construct loaders using the actual sml.data API.
    """

    common_kwargs = {
        "batch_size": args.batch_size,
        "batch_size_train": args.batch_size,
        "batch_size_test": args.batch_size,

        "num_workers": args.workers,
        "workers": args.workers,

        "seed": seed,

        "num_classes": args.num_classes,

        "train_samples": args.train_samples,
        "test_samples": args.test_samples,

        "n_train": args.train_samples,
        "n_test": args.test_samples,

        "train_size": args.train_samples,
        "test_size": args.test_samples,
    }

    common_kwargs = {
        k: v
        for k, v in common_kwargs.items()
        if v is not None
    }

    if args.dataset == "synthetic":

        result = _call_loader_adaptively(
            get_synthetic_loaders,
            common_kwargs,
        )

    elif args.dataset == "cifar":

        result = _call_loader_adaptively(
            get_cifar_loaders,
            common_kwargs,
        )

    else:

        raise ValueError(
            f"Unsupported dataset: "
            f"{args.dataset}"
        )

    if not isinstance(
        result,
        (tuple, list),
    ):

        raise TypeError(
            f"{args.dataset} loader returned "
            f"{type(result)} instead of a tuple/list."
        )

    if len(result) == 3:

        train_loader, test_loader, num_classes = result

    elif len(result) == 2:

        train_loader, test_loader = result

        num_classes = args.num_classes

        print(
            "[DATA] Loader returned two values; "
            f"using num_classes={num_classes}"
        )

    else:

        raise ValueError(
            f"{args.dataset} loader returned "
            f"{len(result)} values. "
            "Expected 2 or 3."
        )

    return (
        train_loader,
        test_loader,
        int(num_classes),
    )


# =====================================================================
# SML CONFIG
# =====================================================================

def build_config(
    args: argparse.Namespace,
    num_classes: int,
    eta: float,
) -> SMLConfig:

    return SMLConfig(
        K=args.K,
        num_classes=num_classes,
        repr_dim=args.repr_dim,
        widths=None,

        lambda_out=args.lambda_out,
        lambda_rep=args.lambda_rep,
        lambda_div=args.lambda_div,

        d0=args.d0,
        eta=eta,
        T=args.T,

        lr=args.lr,
        weight_decay=args.weight_decay,

        use_out=True,
        use_rep=True,
        use_div=True,

        independent=False,

        device=args.device,

        method="SML",
    )


# =====================================================================
# THEORY
# =====================================================================

def theoretical_d_star_squared(
    d0: float,
    w: float,
    eta: float,
) -> float:
    """
    Explicit theoretical equilibrium used by Exp3-A.

    d_star^2 = max(0, d0^2 - w / eta)

    This is intentionally computed here rather than using
    cfg.d_star_squared because the implementation-level helper
    in sml.losses may return None in the subcritical regime
    (rho <= 1), whereas Exp3-A explicitly defines the theoretical
    value as zero in that regime.
    """

    if eta <= 0:

        raise ValueError(
            f"eta must be positive, got {eta}"
        )

    return max(
        0.0,
        d0 * d0 - w / eta,
    )


# =====================================================================
# MODEL DIGEST
# =====================================================================

def model_digest(
    trainer: SMLTrainer,
) -> str:
    """
    SHA-256 digest of the complete K-model cohort.
    """

    h = hashlib.sha256()

    state = trainer.models.state_dict()

    for key in sorted(
        state.keys()
    ):

        tensor = (
            state[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode("utf-8")
        )

        h.update(
            str(tensor.dtype).encode("utf-8")
        )

        h.update(
            str(
                tuple(tensor.shape)
            ).encode("utf-8")
        )

        h.update(
            tensor.numpy().tobytes()
        )

    return h.hexdigest()


# =====================================================================
# CHECKPOINT PATHS
# =====================================================================

def eta_tag(
    eta: float,
) -> str:
    """
    Convert eta to a filesystem-safe tag.

    Examples:
        0.5 -> 0p5
        1.0 -> 1
        2.0 -> 2
    """

    return (
        f"{float(eta):g}"
        .replace(".", "p")
    )


def run_checkpoint_path(
    checkpoint_dir: Path,
    seed: int,
    eta: float,
) -> Path:
    """
    Return the checkpoint FILE for one specific (seed, eta) run.

    IMPORTANT:
    save_checkpoint/load_checkpoint_if_valid in sml.train.py expect
    the final checkpoint path, not the directory itself.
    """

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        checkpoint_dir
        / (
            f"seed{seed}_eta"
            f"{eta_tag(eta)}.pt"
        )
    )


def common_initial_state_path(
    checkpoint_dir: Path,
    seed: int,
) -> Path:

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        checkpoint_dir
        / f"common_initial_state_seed{seed}.pt"
    )


# =====================================================================
# COMMON INITIALIZATION
# =====================================================================

def save_initial_state(
    trainer: SMLTrainer,
    path: Path,
) -> str:
    """
    Save the initial state of the complete model cohort.

    Only model parameters are stored.
    Optimizer state is intentionally excluded.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = trainer.cfg

    digest = model_digest(
        trainer
    )

    payload = {
        "format_version": 1,

        "models":
            trainer.models.state_dict(),

        "K":
            int(cfg.K),

        "num_classes":
            int(cfg.num_classes),

        "repr_dim":
            int(cfg.repr_dim),

        "initial_digest":
            digest,
    }

    torch.save(
        payload,
        path,
    )

    print(
        "[INIT] Saved common initial state:"
    )

    print(
        f"       {path}"
    )

    print(
        f"       digest={digest}"
    )

    return digest


def load_initial_state(
    trainer: SMLTrainer,
    path: Path,
) -> str:
    """
    Restore the common model initialization.

    Optimizer state is deliberately left untouched.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Common initial state does not exist:\n"
            f"{path}"
        )

    state = torch.load(
        path,
        map_location=trainer.cfg.device,
    )

    if int(
        state.get("K", -1)
    ) != int(
        trainer.cfg.K
    ):

        raise ValueError(
            "Initial-state K does not "
            "match current K."
        )

    if int(
        state.get("num_classes", -1)
    ) != int(
        trainer.cfg.num_classes
    ):

        raise ValueError(
            "Initial-state num_classes does not "
            "match current num_classes."
        )

    if int(
        state.get("repr_dim", -1)
    ) != int(
        trainer.cfg.repr_dim
    ):

        raise ValueError(
            "Initial-state repr_dim does not "
            "match current repr_dim."
        )

    if "models" not in state:

        raise KeyError(
            "Initial-state file does not contain "
            "'models'."
        )

    trainer.models.load_state_dict(
        state["models"]
    )

    digest = model_digest(
        trainer
    )

    saved_digest = state.get(
        "initial_digest"
    )

    if (
        saved_digest is not None
        and digest != saved_digest
    ):

        raise RuntimeError(
            "Initial-state digest mismatch.\n"
            f"Saved : {saved_digest}\n"
            f"Loaded: {digest}"
        )

    print(
        "[INIT] Loaded common initial state:"
    )

    print(
        f"       {path}"
    )

    print(
        f"       digest={digest}"
    )

    return digest


# =====================================================================
# ONE EXPERIMENTAL RUN
# =====================================================================

def train_one(
    args: argparse.Namespace,
    seed: int,
    eta: float,
    common_initial_state: Path,
    checkpoint_dir: Path,
) -> Dict[str, Any]:

    print()
    print(
        "-" * 72
    )

    print(
        f"RUN seed={seed} | eta={eta:g}"
    )

    print(
        "-" * 72
    )

    # -------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------

    seed_everything(
        seed
    )

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    (
        train_loader,
        test_loader,
        num_classes,
    ) = make_loaders(
        args,
        seed,
    )

    # -------------------------------------------------------------
    # Config
    # -------------------------------------------------------------

    cfg = build_config(
        args=args,
        num_classes=num_classes,
        eta=eta,
    )

    # -------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------

    trainer = SMLTrainer(
        cfg
    )

    # -------------------------------------------------------------
    # CRITICAL:
    # One independent checkpoint FILE per (seed, eta).
    # -------------------------------------------------------------

    checkpoint_path = run_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        eta=eta,
    )

    print(
        f"[CKPT] {checkpoint_path}"
    )

    # -------------------------------------------------------------
    # Resume
    #
    # IMPORTANT:
    # Pass checkpoint_path, NOT checkpoint_dir.
    # -------------------------------------------------------------

    resumed = load_checkpoint_if_valid(
        trainer,
        cfg,
        seed,
        args.dataset,
        checkpoint_path,
    )

    start_epoch = int(
        getattr(
            trainer,
            "completed_epochs",
            0,
        )
    )

    # -------------------------------------------------------------
    # New run -> common initialization
    # -------------------------------------------------------------

    if not resumed:

        initial_digest = load_initial_state(
            trainer,
            common_initial_state,
        )

        trainer.completed_epochs = 0

        start_epoch = 0

    else:

        # The checkpoint contains the trained model.
        # For bookkeeping, recover the expected initial digest
        # from the common initialization file.
        initial_digest = None

        try:

            initial_state = torch.load(
                common_initial_state,
                map_location="cpu",
            )

            initial_digest = (
                initial_state.get(
                    "initial_digest"
                )
            )

        except Exception as exc:

            print(
                "[WARN] Could not recover "
                f"initial digest: {exc}"
            )

        print(
            f"[RESUME] "
            f"seed={seed}, "
            f"eta={eta:g}, "
            f"epoch={start_epoch}"
        )

    # -------------------------------------------------------------
    # Theory
    # -------------------------------------------------------------

    rho = float(
        cfg.rho
    )

    # IMPORTANT:
    # Do NOT use cfg.d_star_squared here.
    # The implementation helper may return None when rho <= 1.
    #
    # Exp3-A explicitly defines:
    #
    #     d*^2 = max(0, d0^2 - w / eta)
    #
    d_star_squared = theoretical_d_star_squared(
        d0=float(args.d0),
        w=float(cfg.w),
        eta=float(eta),
    )

    print(
        f"[CONFIG] "
        f"w={cfg.w:.6f}, "
        f"rho={rho:.6f}, "
        f"d*²={d_star_squared:.6f}"
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:

            logs, _, _ = trainer.step(
                x,
                y,
            )

            total_loss += float(
                logs["total"]
            )

            n_batches += 1

        # ---------------------------------------------------------
        # Optional scheduler compatibility.
        # ---------------------------------------------------------

        scheduler = getattr(
            trainer,
            "scheduler",
            None,
        )

        if scheduler is not None:

            scheduler.step()

        trainer.completed_epochs = (
            epoch + 1
        )

        # ---------------------------------------------------------
        # CRITICAL:
        # Save to the RUN-SPECIFIC FILE.
        # ---------------------------------------------------------

        save_checkpoint(
            trainer,
            cfg,
            seed,
            args.dataset,
            checkpoint_path,
        )

        avg_loss = (
            total_loss / n_batches
            if n_batches > 0
            else float("nan")
        )

        print(
            f"    epoch "
            f"{epoch + 1:4d}/"
            f"{args.epochs} "
            f"loss={avg_loss:.6f}"
        )

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    metrics = trainer.evaluate(
        test_loader
    )

    final_digest = model_digest(
        trainer
    )

    result = {
        "seed":
            int(seed),

        "eta":
            float(eta),

        "epochs_completed":
            int(
                getattr(
                    trainer,
                    "completed_epochs",
                    args.epochs,
                )
            ),

        "resumed":
            bool(resumed),

        "checkpoint":
            str(checkpoint_path),

        "initial_digest":
            initial_digest,

        "final_digest":
            final_digest,

        "ensemble_acc":
            metrics["ensemble_acc"],

        "best_single_acc":
            metrics["best_single_acc"],

        "per_model_acc":
            metrics["per_model_acc"],

        "D_rep":
            metrics["D_rep"],

        "E_d0":
            metrics["E_d0"],

        "representation_dispersion":
            metrics["representation_dispersion"],

        "prediction_disagreement_rate":
            metrics[
                "prediction_disagreement_rate"
            ],

        "prediction_symmetric_kl":
            metrics[
                "prediction_symmetric_kl"
            ],

        "rho":
            metrics["rho"],

        "w":
            metrics["w"],

        # IMPORTANT:
        # Store the explicit theoretical value computed above.
        "d_star_squared_theory":
            float(d_star_squared),
    }

    print(
        f"[DONE] "
        f"seed={seed} "
        f"eta={eta:g} "
        f"D_rep="
        f"{float(result['D_rep']):.6f} "
        f"Acc="
        f"{float(result['ensemble_acc']):.6f} "
        f"rho="
        f"{float(result['rho']):.6f}"
    )

    return result


# =====================================================================
# THEORY TABLE
# =====================================================================

def make_theory_table(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:

    if args.lambda_div == 0:

        raise ValueError(
            "lambda_div must be non-zero."
        )

    w = (
        2.0
        * args.lambda_rep
        / args.lambda_div
    )

    rows = []

    for eta in ETAS:

        rho = (
            eta
            * args.d0
            * args.d0
            / w
        )

        d_star_squared = (
            theoretical_d_star_squared(
                d0=float(args.d0),
                w=float(w),
                eta=float(eta),
            )
        )

        rows.append(
            {
                "eta":
                    float(eta),

                "w":
                    float(w),

                "rho":
                    float(rho),

                "d_star_squared_theory":
                    float(
                        d_star_squared
                    ),
            }
        )

    return rows


# =====================================================================
# AGGREGATION
# =====================================================================

def aggregate_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:

    grouped: Dict[
        float,
        List[Dict[str, Any]]
    ] = {}

    for row in results:

        eta = float(
            row["eta"]
        )

        grouped.setdefault(
            eta,
            []
        ).append(row)

    metrics = [
        "ensemble_acc",
        "best_single_acc",
        "D_rep",
        "E_d0",
        "representation_dispersion",
        "prediction_disagreement_rate",
        "prediction_symmetric_kl",
    ]

    output = {}

    for eta in sorted(
        grouped.keys()
    ):

        rows = grouped[eta]

        summary = {
            "eta":
                eta,

            "n_seeds":
                len(rows),

            "rho":
                rows[0]["rho"],

            "w":
                rows[0]["w"],

            "d_star_squared_theory":
                rows[0][
                    "d_star_squared_theory"
                ],
        }

        for metric in metrics:

            values = [
                float(row[metric])
                for row in rows
                if row.get(metric) is not None
            ]

            if not values:

                summary[
                    f"{metric}_mean"
                ] = None

                summary[
                    f"{metric}_std"
                ] = None

                continue

            mean = (
                sum(values)
                / len(values)
            )

            if len(values) > 1:

                variance = (
                    sum(
                        (v - mean) ** 2
                        for v in values
                    )
                    / (len(values) - 1)
                )

                std = math.sqrt(
                    variance
                )

            else:

                std = 0.0

            summary[
                f"{metric}_mean"
            ] = mean

            summary[
                f"{metric}_std"
            ] = std

        output[
            str(eta)
        ] = summary

    return output


# =====================================================================
# INITIALIZATION VERIFICATION
# =====================================================================

def verify_common_initialization(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify identical initial model states within each seed.
    """

    by_seed: Dict[
        int,
        List[str]
    ] = {}

    for row in results:

        digest = row.get(
            "initial_digest"
        )

        if digest is None:
            continue

        seed = int(
            row["seed"]
        )

        by_seed.setdefault(
            seed,
            []
        ).append(digest)

    checks = {}

    all_ok = True

    for seed in sorted(
        by_seed.keys()
    ):

        digests = by_seed[seed]

        unique = sorted(
            set(digests)
        )

        identical = (
            len(unique) <= 1
        )

        checks[
            str(seed)
        ] = {
            "n_runs_with_digest":
                len(digests),

            "n_unique_initial_digests":
                len(unique),

            "identical":
                identical,
        }

        if not identical:
            all_ok = False

    return {
        "all_seeds_pass":
            all_ok,

        "by_seed":
            checks,
    }


# =====================================================================
# OUTPUT BUILDER
# =====================================================================

def build_output(
    args: argparse.Namespace,
    seeds: List[int],
    theory: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    summary = aggregate_results(
        results
    )

    init_check = (
        verify_common_initialization(
            results
        )
    )

    return {
        "experiment":
            "Exp3-A",

        "description":
            (
                "SML finite-target diversity "
                "transition via eta sweep"
            ),

        "dataset":
            args.dataset,

        "device":
            args.device,

        "K":
            args.K,

        "repr_dim":
            args.repr_dim,

        "epochs":
            args.epochs,

        "seeds":
            seeds,

        "etas":
            ETAS,

        "num_classes":
            args.num_classes,

        "batch_size":
            args.batch_size,

        "workers":
            args.workers,

        "train_samples":
            args.train_samples,

        "test_samples":
            args.test_samples,

        "lambda_out":
            args.lambda_out,

        "lambda_rep":
            args.lambda_rep,

        "lambda_div":
            args.lambda_div,

        "d0":
            args.d0,

        "T":
            args.T,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "deterministic":
            args.deterministic,

        "theory":
            theory,

        "runs":
            results,

        "summary_by_eta":
            summary,

        "common_initialization_check":
            init_check,
    }


# =====================================================================
# PLOTTING
# =====================================================================

def _load_transition_summary(
    path: Path,
    dataset_label: str,
):
    """
    Load and validate the Exp3-A summary from a result JSON.

    The final JSON structure stores summary_by_eta as a dictionary
    keyed by eta, e.g.:

        {
            "0.5": {...},
            "1.0": {...},
            ...
        }
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{dataset_label} result file does not exist:\n"
            f"{path}"
        )

    payload = load_json_if_exists(
        path
    )

    if payload is None:

        raise RuntimeError(
            f"Could not load {dataset_label} "
            f"result file:\n{path}"
        )

    summary = payload.get(
        "summary_by_eta"
    )

    if not isinstance(
        summary,
        dict,
    ):

        raise RuntimeError(
            f"{dataset_label}: "
            "'summary_by_eta' is missing or "
            "has an unexpected format."
        )

    rows = []

    for eta_key, row in summary.items():

        if not isinstance(
            row,
            dict,
        ):

            continue

        required = [
            "eta",
            "rho",
            "D_rep_mean",
            "D_rep_std",
            "d_star_squared_theory",
        ]

        missing = [
            key
            for key in required
            if key not in row
        ]

        if missing:

            raise RuntimeError(
                f"{dataset_label}: eta={eta_key} "
                f"is missing fields: {missing}"
            )

        if (
            row["D_rep_mean"] is None
            or row["D_rep_std"] is None
        ):

            continue

        rows.append(
            {
                "eta":
                    float(row["eta"]),

                "rho":
                    float(row["rho"]),

                "D_rep_mean":
                    float(row["D_rep_mean"]),

                "D_rep_std":
                    float(row["D_rep_std"]),

                "d_star_squared_theory":
                    float(
                        row[
                            "d_star_squared_theory"
                        ]
                    ),

                "n_seeds":
                    int(
                        row.get(
                            "n_seeds",
                            0,
                        )
                    ),
            }
        )

    if not rows:

        raise RuntimeError(
            f"{dataset_label}: no valid "
            "transition summary rows found."
        )

    rows.sort(
        key=lambda row: row["rho"]
    )

    n_seeds = sorted(
        set(
            row["n_seeds"]
            for row in rows
        )
    )

    if n_seeds != [5]:

        print(
            f"[PLOT] WARNING: {dataset_label} "
            f"contains n_seeds={n_seeds}; "
            "expected five seeds."
        )

    return payload, rows


def plot_exp3a_transition(
    synthetic_json: Path,
    cifar_json: Path,
    output_path: Path,
) -> None:
    """
    Generate the final two-panel Exp3-A transition figure.

    Panel (a):
        Synthetic

    Panel (b):
        CIFAR-10

    Empirical:
        mean +/- one standard deviation across seeds.

    Theory:
        d_*^2 = max(0, d0^2 - w / eta)

    For the default experiment:
        d0 = 1
        w  = 2

    hence:
        d_*^2 = max(0, 1 - 1/rho)

    The vertical dashed line marks the predicted transition rho=1.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:

        raise RuntimeError(
            "Matplotlib is required for plotting. "
            "Install it in the experiment environment."
        ) from exc

    synthetic_payload, synthetic_rows = (
        _load_transition_summary(
            synthetic_json,
            "Synthetic",
        )
    )

    cifar_payload, cifar_rows = (
        _load_transition_summary(
            cifar_json,
            "CIFAR-10",
        )
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.6),
        sharey=True,
    )

    datasets = [
        (
            axes[0],
            synthetic_payload,
            synthetic_rows,
            "(a) Synthetic",
        ),
        (
            axes[1],
            cifar_payload,
            cifar_rows,
            "(b) CIFAR-10",
        ),
    ]

    for ax, payload, rows, title in datasets:

        rho = [
            row["rho"]
            for row in rows
        ]

        mean = [
            row["D_rep_mean"]
            for row in rows
        ]

        std = [
            row["D_rep_std"]
            for row in rows
        ]

        # ---------------------------------------------------------
        # Empirical D_rep
        # ---------------------------------------------------------

        ax.errorbar(
            rho,
            mean,
            yerr=std,
            fmt="o",
            markersize=5,
            capsize=3,
            linewidth=1.0,
            label=r"Empirical $D_{\mathrm{rep}}$",
        )

        # ---------------------------------------------------------
        # Theoretical curve
        #
        # Use d0 from the actual result file rather than hard-coding
        # it. The curve in rho-space is:
        #
        # d_*^2 = max(0, d0^2 (1 - 1/rho))
        #
        # because:
        #
        # rho = eta d0^2 / w
        # w/eta = d0^2/rho
        # ---------------------------------------------------------

        d0 = float(
            payload.get(
                "d0",
                1.0,
            )
        )

        rho_min = min(rho)
        rho_max = max(rho)

        # Use a dense linear grid because the actual experiment is
        # displayed on a linear rho axis.
        n_curve = 400

        step = (
            rho_max - rho_min
        ) / max(
            n_curve - 1,
            1,
        )

        if step <= 0:

            rho_curve = [
                rho_min
            ]

        else:

            rho_curve = [
                rho_min + i * step
                for i in range(n_curve)
            ]

        theory_curve = [
            max(
                0.0,
                d0 * d0
                * (
                    1.0
                    - 1.0 / r
                ),
            )
            for r in rho_curve
        ]

        ax.plot(
            rho_curve,
            theory_curve,
            linestyle="-",
            linewidth=1.5,
            label=r"Theory $d_*^2$",
        )

        # ---------------------------------------------------------
        # Predicted transition
        # ---------------------------------------------------------

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label=r"Transition $\rho=1$",
        )

        # ---------------------------------------------------------
        # Axes
        # ---------------------------------------------------------

        ax.set_xlabel(
            r"$\rho=\eta d_0^2/w$"
        )

        ax.set_title(
            title
        )

        ax.set_xticks(
            [
                0.25,
                0.5,
                1.0,
                1.5,
                2.0,
                4.0,
            ]
        )

        ax.grid(
            True,
            alpha=0.20,
        )

        ax.set_xlim(
            max(
                0.0,
                rho_min - 0.10,
            ),
            rho_max + 0.10,
        )

    axes[0].set_ylabel(
        r"Representation diversity $D_{\mathrm{rep}}$"
    )

    # -------------------------------------------------------------
    # One common legend
    # -------------------------------------------------------------

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.90,
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"[PLOT] Saved Exp3-A transition figure:"
    )
    print(
        f"       {output_path}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Exp3-A: SML eta-transition "
            "experiment"
        )
    )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=[
            "synthetic",
            "cifar",
        ],
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
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
        default=2,
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

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help=(
            "Number of seeds; "
            "uses seeds 0,...,seeds-1."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
    )

    # -------------------------------------------------------------
    # SML
    # -------------------------------------------------------------

    parser.add_argument(
        "--lambda_out",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda_rep",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--d0",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--T",
        type=float,
        default=4.0,
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "exp3_transition.json"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=(
            "checkpoints/"
            "exp3_transition"
        ),
    )

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------

    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help=(
            "Generate the final Exp3-A figure from "
            "existing result JSON files without training."
        ),
    )

    parser.add_argument(
        "--synthetic-results",
        type=str,
        default=(
            "results/"
            "exp3_transition.json"
        ),
        help=(
            "Synthetic Exp3-A result JSON used for plotting."
        ),
    )

    parser.add_argument(
        "--cifar-results",
        type=str,
        default=(
            "results/"
            "exp3_transition_cifar10.json"
        ),
        help=(
            "CIFAR-10 Exp3-A result JSON used for plotting."
        ),
    )

    parser.add_argument(
        "--figure-output",
        type=str,
        default=(
            "figures/"
            "exp3a_transition.pdf"
        ),
        help=(
            "Output path for the final Exp3-A figure."
        ),
    )

    args = parser.parse_args()

    # =================================================================
    # PLOT-ONLY MODE
    # =================================================================

    if args.plot_only:

        print()
        print(
            "=" * 72
        )

        print(
            "EXP3-A: PLOT ONLY"
        )

        print(
            "=" * 72
        )

        synthetic_json = Path(
            args.synthetic_results
        )

        cifar_json = Path(
            args.cifar_results
        )

        figure_output = Path(
            args.figure_output
        )

        print(
            f"Synthetic results = {synthetic_json}"
        )

        print(
            f"CIFAR-10 results  = {cifar_json}"
        )

        print(
            f"Figure output     = {figure_output}"
        )

        print(
            "=" * 72
        )

        plot_exp3a_transition(
            synthetic_json=synthetic_json,
            cifar_json=cifar_json,
            output_path=figure_output,
        )

        print()
        print(
            "[DONE] Plot-only mode completed."
        )

        return

    # =================================================================
    # PATHS
    # =================================================================

    output_path = Path(
        args.output
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # DETERMINISM
    # =================================================================

    if args.deterministic:
        enable_deterministic_mode()

    # =================================================================
    # SEEDS
    # =================================================================

    seeds = list(
        range(args.seeds)
    )

    # =================================================================
    # THEORY
    # =================================================================

    theory = make_theory_table(
        args
    )

    # =================================================================
    # HEADER
    # =================================================================

    print()
    print(
        "=" * 72
    )

    print(
        "EXP3-A: SML ETA TRANSITION"
    )

    print(
        "=" * 72
    )

    print(
        f"dataset       = {args.dataset}"
    )

    print(
        f"device        = {args.device}"
    )

    print(
        f"K             = {args.K}"
    )

    print(
        f"repr_dim      = {args.repr_dim}"
    )

    print(
        f"epochs        = {args.epochs}"
    )

    print(
        f"seeds         = {seeds}"
    )

    print(
        f"etas          = {ETAS}"
    )

    print(
        f"lambda_out    = {args.lambda_out}"
    )

    print(
        f"lambda_rep    = {args.lambda_rep}"
    )

    print(
        f"lambda_div    = {args.lambda_div}"
    )

    print(
        f"d0            = {args.d0}"
    )

    print(
        f"batch_size    = {args.batch_size}"
    )

    print(
        f"checkpoint dir = {checkpoint_dir}"
    )

    print(
        f"total runs    = "
        f"{len(seeds) * len(ETAS)}"
    )

    print(
        "=" * 72
    )

    # =================================================================
    # THEORY TABLE
    # =================================================================

    print()
    print(
        "THEORETICAL TRANSITION"
    )

    print(
        "-" * 72
    )

    for row in theory:

        print(
            f"eta={row['eta']:>4.1f}   "
            f"rho={row['rho']:>6.3f}   "
            f"d*²="
            f"{row['d_star_squared_theory']:>8.5f}"
        )

    # =================================================================
    # EXISTING RESULTS
    # =================================================================

    existing = load_json_if_exists(
        output_path
    )

    if existing is not None:

        results = existing.get(
            "runs",
            []
        )

        print()
        print(
            f"[RESUME] Loaded "
            f"{len(results)} existing result rows "
            f"from {output_path}"
        )

    else:

        results = []

        print()
        print(
            "[RESUME] No existing result file."
        )

    # =================================================================
    # COMPLETED KEYS
    #
    # A result is considered completed only if:
    #
    #     epochs_completed >= requested epochs
    #
    # This prevents an old 2-epoch smoke-test result from causing
    # a later 50-epoch experiment to be skipped.
    # =================================================================

    def result_key(
        row: Dict[str, Any]
    ):

        return (
            int(row["seed"]),
            float(row["eta"]),
        )

    completed_keys = set()

    for row in results:

        if (
            "seed" not in row
            or "eta" not in row
        ):
            continue

        completed_epochs = int(
            row.get(
                "epochs_completed",
                0,
            )
        )

        if completed_epochs >= args.epochs:

            checkpoint = row.get(
                "checkpoint"
            )

            if checkpoint is None:

                checkpoint = str(
                    run_checkpoint_path(
                        checkpoint_dir,
                        int(row["seed"]),
                        float(row["eta"]),
                    )
                )

            if Path(
                checkpoint
            ).exists():

                completed_keys.add(
                    result_key(row)
                )

    # =================================================================
    # SEED LOOP
    # =================================================================

    for seed in seeds:

        print()
        print(
            "=" * 72
        )

        print(
            f"START SEED {seed}"
        )

        print(
            "=" * 72
        )

        common_path = (
            common_initial_state_path(
                checkpoint_dir,
                seed,
            )
        )

        # ---------------------------------------------------------
        # Create common initialization once.
        # ---------------------------------------------------------

        if not common_path.exists():

            print(
                f"[INIT] Creating common initial "
                f"state for seed={seed}"
            )

            seed_everything(
                seed
            )

            (
                train_loader,
                test_loader,
                num_classes,
            ) = make_loaders(
                args,
                seed,
            )

            del train_loader
            del test_loader

            cfg_init = build_config(
                args=args,
                num_classes=num_classes,
                eta=ETAS[0],
            )

            trainer_init = SMLTrainer(
                cfg_init
            )

            save_initial_state(
                trainer_init,
                common_path,
            )

            del trainer_init

        else:

            print(
                "[INIT] Existing common initial "
                f"state found:\n"
                f"       {common_path}"
            )

        # ---------------------------------------------------------
        # Eta loop
        # ---------------------------------------------------------

        for eta in ETAS:

            key = (
                int(seed),
                float(eta),
            )

            checkpoint_path = (
                run_checkpoint_path(
                    checkpoint_dir,
                    seed,
                    eta,
                )
            )

            if key in completed_keys:

                print(
                    f"[SKIP] seed={seed} "
                    f"eta={eta:g} "
                    f"already completed."
                )

                continue

            result = train_one(
                args=args,
                seed=seed,
                eta=eta,
                common_initial_state=common_path,
                checkpoint_dir=checkpoint_dir,
            )

            results.append(
                result
            )

            completed_keys.add(
                key
            )

            # -----------------------------------------------------
            # Incremental save
            # -----------------------------------------------------

            output = build_output(
                args=args,
                seeds=seeds,
                theory=theory,
                results=results,
            )

            save_json(
                output,
                output_path,
            )

            print(
                f"[SAVE] {output_path}"
            )

    # =================================================================
    # FINAL
    # =================================================================

    final_output = build_output(
        args=args,
        seeds=seeds,
        theory=theory,
        results=results,
    )

    save_json(
        final_output,
        output_path,
    )

    summary = final_output[
        "summary_by_eta"
    ]

    init_check = final_output[
        "common_initialization_check"
    ]

    # =================================================================
    # FINAL TABLE
    # =================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "EXP3-A FINAL SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'eta':>7} "
        f"{'rho':>8} "
        f"{'theory':>10} "
        f"{'D_rep':>22} "
        f"{'Accuracy':>22}"
    )

    print(
        "-" * 100
    )

    for eta in ETAS:

        row = summary.get(
            str(eta)
        )

        if row is None:
            continue

        d_mean = row.get(
            "D_rep_mean"
        )

        d_std = row.get(
            "D_rep_std"
        )

        a_mean = row.get(
            "ensemble_acc_mean"
        )

        a_std = row.get(
            "ensemble_acc_std"
        )

        if d_mean is None:

            d_text = "N/A"

        else:

            d_text = (
                f"{d_mean:.5f} "
                f"+/- {d_std:.5f}"
            )

        if a_mean is None:

            a_text = "N/A"

        else:

            a_text = (
                f"{a_mean:.5f} "
                f"+/- {a_std:.5f}"
            )

        print(
            f"{eta:7.2f} "
            f"{row['rho']:8.3f} "
            f"{row['d_star_squared_theory']:10.5f} "
            f"{d_text:>22} "
            f"{a_text:>22}"
        )

    print(
        "-" * 100
    )

    print(
        "Common initialization: "
        + (
            "PASS"
            if init_check["all_seeds_pass"]
            else "FAIL"
        )
    )

    print(
        f"Completed runs: "
        f"{len(results)} / "
        f"{len(seeds) * len(ETAS)}"
    )

    print(
        f"Results: {output_path}"
    )

    print(
        f"Checkpoints: {checkpoint_dir}"
    )

    print(
        "=" * 100
    )

    # =================================================================
    # AUTOMATIC FINAL FIGURE
    #
    # Only generate the publication figure when BOTH datasets'
    # result files are available.
    #
    # This prevents an incomplete one-panel figure from being
    # produced after running only one dataset.
    # =================================================================

    synthetic_results_path = Path(
        args.synthetic_results
    )

    cifar_results_path = Path(
        args.cifar_results
    )

    figure_output_path = Path(
        args.figure_output
    )

    if (
        synthetic_results_path.exists()
        and cifar_results_path.exists()
    ):

        print()
        print(
            "[PLOT] Both Synthetic and CIFAR-10 "
            "result files are available."
        )

        try:

            plot_exp3a_transition(
                synthetic_json=synthetic_results_path,
                cifar_json=cifar_results_path,
                output_path=figure_output_path,
            )

        except Exception as exc:

            print(
                "[PLOT] WARNING: automatic "
                "figure generation failed:"
            )

            print(
                f"        {type(exc).__name__}: {exc}"
            )

    else:

        print()
        print(
            "[PLOT] Final two-panel figure was "
            "not generated because both result files "
            "are not yet available."
        )

        print(
            f"        Synthetic: "
            f"{synthetic_results_path.exists()} "
            f"({synthetic_results_path})"
        )

        print(
            f"        CIFAR-10:  "
            f"{cifar_results_path.exists()} "
            f"({cifar_results_path})"
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()