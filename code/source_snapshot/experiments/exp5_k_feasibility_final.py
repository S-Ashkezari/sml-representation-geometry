"""
exp5_k_feasibility_final.py
===========================

Primary Exp5 for the SML paper.

Tests the finite-K spherical feasibility constraint for a common pairwise
representation-distance target. For K peers with unit-normalized
representations, the equal-pairwise squared-distance cap is

    d_cap^2 = 2K/(K-1).

For the representation/diversity potential used by SML, the unconstrained
stationary prediction is

    d_*^2 = max(0, d0^2 - w/eta),
    w = 2*lambda_rep/lambda_div,

and the finite-K prediction is

    d_pred^2 = min(d_*^2, 2K/(K-1)).

The experiment sweeps K in {2,3,4,5} and target values below, near, and
above the finite-K boundary.

This version is compatible with the current sml/data.py API:
    get_cifar_loaders(root, dataset, batch_size, num_workers)
    get_synthetic_loaders(..., seed)

Recommended paper run:
    python -m experiments.exp5_k_feasibility_final \
        --dataset cifar10 --epochs 50 --seeds 5 \
        --batch-size 128 --workers 0 --device cuda \
        --deterministic \
        --output results/exp5_k_feasibility_cifar10.json

Smoke test:
    python -m experiments.exp5_k_feasibility_final \
        --synthetic --epochs 2 --seeds 1 --device cpu \
        --output results/exp5_k_feasibility_smoke.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sml.train import SMLConfig, SMLTrainer
from sml.data import get_cifar_loaders, get_synthetic_loaders


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything_local(seed: int) -> None:
    """Self-contained seeding; does not require seed_everything in data.py."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------

def make_config(**kwargs: Any) -> SMLConfig:
    """
    Pass only fields supported by the installed SMLConfig.
    This prevents Exp5 from conflicting with small train.py API differences.
    """
    try:
        sig = inspect.signature(SMLConfig)
        allowed = set(sig.parameters)
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        filtered = kwargs
    return SMLConfig(**filtered)


def maybe_step_scheduler(trainer: SMLTrainer) -> None:
    scheduler = getattr(trainer, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step()


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def make_loaders(args: argparse.Namespace, seed: int):
    # Reset RNG immediately before DataLoader construction. The current
    # data.py does not accept seed in get_cifar_loaders; shuffle therefore
    # follows the globally seeded PyTorch RNG.
    seed_everything_local(seed)

    if args.synthetic:
        return get_synthetic_loaders(
            num_classes=10,
            n_per_class_train=args.synthetic_train_per_class,
            n_per_class_test=args.synthetic_test_per_class,
            batch_size=min(args.batch_size, 64),
            seed=seed,
        )

    return get_cifar_loaders(
        root=args.data_root,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


# ---------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------

def theory_values(
    K: int,
    d0: float,
    lambda_rep: float,
    lambda_div: float,
    eta: float,
) -> Dict[str, float]:
    w = 2.0 * lambda_rep / lambda_div
    cap_sq = 2.0 * K / (K - 1)
    unconstrained_sq = max(d0 * d0 - w / eta, 0.0)
    predicted_sq = min(unconstrained_sq, cap_sq)
    rho = eta * d0 * d0 / w if w > 0 else float("inf")
    return {
        "w": w,
        "rho": rho,
        "cap_squared": cap_sq,
        "d_star_squared_unconstrained": unconstrained_sq,
        "D_pred_squared": predicted_sq,
    }


def d0_grid_for_K(
    K: int,
    lambda_rep: float,
    lambda_div: float,
    eta: float,
) -> List[float]:
    """
    Choose targets relative to the boundary in d0-space.

    The finite-K cap constrains d_*^2, not d0^2 directly:
        d0_boundary^2 = cap^2 + w/eta.
    """
    w = 2.0 * lambda_rep / lambda_div
    cap_sq = 2.0 * K / (K - 1)
    d0_boundary = math.sqrt(cap_sq + w / eta)

    candidates = [
        0.80,
        1.20,
        0.95 * d0_boundary,   # below boundary
        d0_boundary,          # at boundary
        1.10 * d0_boundary,   # above boundary
    ]
    return sorted({round(float(x), 4) for x in candidates})


# ---------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------

def run_one(
    train_loader,
    test_loader,
    num_classes: int,
    K: int,
    d0: float,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, Any]:

    seed_everything_local(seed)

    cfg_kwargs: Dict[str, Any] = dict(
        K=K,
        num_classes=num_classes,
        repr_dim=args.repr_dim,
        lambda_out=args.lambda_out,
        lambda_rep=args.lambda_rep,
        lambda_div=args.lambda_div,
        d0=d0,
        eta=args.eta,
        T=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_out=True,
        use_rep=True,
        use_div=True,
        method="SML",
        device=args.device,
    )

    # Optional heterogeneous cohort. Use one width per peer, including K=5.
    if args.heterogeneous:
        width_bank = [32, 48, 64, 80, 96, 112, 128]
        cfg_kwargs["widths"] = width_bank[:K]

    cfg = make_config(**cfg_kwargs)
    trainer = SMLTrainer(cfg, in_channels=3)

    for _ in range(args.epochs):
        for x, y in train_loader:
            trainer.step(x, y)
        maybe_step_scheduler(trainer)

    metrics = trainer.evaluate(test_loader)
    theory = theory_values(
        K, d0, args.lambda_rep, args.lambda_div, args.eta
    )

    out: Dict[str, Any] = {
        "K": K,
        "d0": d0,
        "seed": seed,
        "dataset": "synthetic" if args.synthetic else args.dataset,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "repr_dim": args.repr_dim,
        "lambda_out": args.lambda_out,
        "lambda_rep": args.lambda_rep,
        "lambda_div": args.lambda_div,
        "eta": args.eta,
        **theory,
    }
    out.update(metrics)
    return out


# ---------------------------------------------------------------------
# JSON / aggregation
# ---------------------------------------------------------------------

def json_safe(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.ndim else x.item()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def mean_std(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["K"], r["d0"]), []).append(r)

    keys = [
        "ensemble_acc",
        "best_single_acc",
        "D_rep",
        "E_d0",
        "representation_dispersion",
        "prediction_disagreement_rate",
        "prediction_symmetric_kl",
    ]

    rows = []
    for (K, d0), rr in sorted(groups.items()):
        row: Dict[str, Any] = {
            "K": K,
            "d0": d0,
            "n_seeds": len(rr),
            "seeds": [r["seed"] for r in rr],
            "cap_squared": rr[0]["cap_squared"],
            "d_star_squared_unconstrained": rr[0]["d_star_squared_unconstrained"],
            "D_pred_squared": rr[0]["D_pred_squared"],
            "rho": rr[0]["rho"],
        }
        for key in keys:
            vals = [float(r[key]) for r in rr if r.get(key) is not None]
            if vals:
                s = mean_std(vals)
                row[f"{key}_mean"] = s["mean"]
                row[f"{key}_std"] = s["std"]
        rows.append(row)
    return rows


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp5: finite-K feasibility of controlled SML representation disagreement."
    )
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--repr-dim", "--repr_dim", dest="repr_dim", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--heterogeneous", action="store_true")

    # Exp5 coefficients: small w makes the finite-K saturation visible.
    ap.add_argument("--lambda-out", type=float, default=1.0)
    ap.add_argument("--lambda-rep", type=float, default=0.05)
    ap.add_argument("--lambda-div", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)

    ap.add_argument("--synthetic-train-per-class", type=int, default=64)
    ap.add_argument("--synthetic-test-per-class", type=int, default=32)
    ap.add_argument(
        "--output",
        "--out",
        dest="output",
        default="results/exp5_k_feasibility_cifar10.json",
    )
    args = ap.parse_args()

    if args.seeds < 1 or args.epochs < 1:
        raise ValueError("--seeds and --epochs must be >= 1.")
    if args.lambda_div <= 0 or args.eta <= 0:
        raise ValueError("--lambda-div and --eta must be positive.")

    if args.deterministic:
        enable_deterministic_mode()

    output_path = Path(args.output)
    runs: List[Dict[str, Any]] = []

    for K in (2, 3, 4, 5):
        d0s = d0_grid_for_K(
            K, args.lambda_rep, args.lambda_div, args.eta
        )
        for d0 in d0s:
            for seed in range(args.seeds):
                print(f"[Exp5] K={K} d0={d0:.4f} seed={seed}")
                train_loader, test_loader, num_classes = make_loaders(args, seed)
                result = run_one(
                    train_loader, test_loader, num_classes,
                    K, d0, args, seed
                )
                runs.append(result)

                print(
                    f"       D_rep={float(result['D_rep']):.4f} | "
                    f"pred={result['D_pred_squared']:.4f} | "
                    f"cap={result['cap_squared']:.4f} | "
                    f"E_d0={float(result['E_d0']):.4f} | "
                    f"acc={100*float(result['ensemble_acc']):.2f}%"
                )

                payload = {
                    "experiment": "Exp5 — finite-K representation feasibility",
                    "config": {
                        **vars(args),
                        "K_values": [2, 3, 4, 5],
                        "data_api": (
                            "get_cifar_loaders(root,dataset,batch_size,num_workers)"
                            if not args.synthetic else "get_synthetic_loaders"
                        ),
                    },
                    "runs": runs,
                    "aggregated": aggregate(runs),
                }
                save_json(output_path, payload)

    print(f"Saved {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
