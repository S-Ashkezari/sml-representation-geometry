"""
exp6_strong_backbone_resnet18.py
================================

Targeted strong-backbone robustness study for SML.

Backbone:
    CIFAR-adapted ResNet-18 (3x3 stride-1 stem, no initial max-pool)

Total planned runs: 18
    A) Matched comparison: Independent, DML, SML x 3 seeds = 9
       Uses the main Exp3 operating point:
           lambda_out=0.5, lambda_rep=0.1, lambda_div=0.1,
           d0=1.0, eta=1.0

    B) Geometry check: SML at three d0 values x 3 seeds = 9
       Uses Exp5 geometry-stress coefficients:
           lambda_out=1.0, lambda_rep=0.05, lambda_div=1.0,
           eta=1.0, d0 in {1.2, 1.7607, 1.9367}

POWER-LOSS / RESUME BEHAVIOR
----------------------------
1. A separate checkpoint is stored for every job.
2. The trainer checkpoint contains model, optimizer, scheduler, completed
   epoch, and Python/NumPy/PyTorch/CUDA RNG states.
3. A checkpoint is written atomically after every epoch.
4. Results JSON is written atomically after every completed run.
5. On restart:
      - completed JSON jobs are skipped;
      - an unfinished job loads its checkpoint and resumes at the next epoch.
6. Do not use --fresh unless you intentionally want to discard prior progress.

Recommended run:
    python -m experiments.exp6_strong_backbone_resnet18 \
        --dataset cifar10 \
        --epochs 100 \
        --seeds 3 \
        --device cuda \
        --deterministic \
        --checkpoint-dir checkpoints/exp6_resnet18 \
        --output results/exp6_resnet18.json

If GPU memory is insufficient, reduce --batch-size to 96 or 64.
Do not change it after checkpoints have been created for that experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sml.train_resnet18 import (
    SMLConfig,
    SMLTrainer,
    save_checkpoint,
    load_checkpoint_if_valid,
)
from sml.data import get_cifar_loaders


# ---------------------------------------------------------------------
# Reproducibility / IO
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
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


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def atomic_json_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------

def job_specs() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    # A) Matched strong-backbone comparison.
    for method in ("Independent", "DML", "SML"):
        jobs.append({
            "study": "matched",
            "method": method,
            "d0": 1.0,
            "lambda_out": 0.5,
            "lambda_rep": 0.1,
            "lambda_div": 0.1,
            "eta": 1.0,
        })

    # B) Minimal finite-K geometry replication at K=3.
    for d0 in (1.2, 1.7607, 1.9367):
        jobs.append({
            "study": "geometry",
            "method": "SML",
            "d0": d0,
            "lambda_out": 1.0,
            "lambda_rep": 0.05,
            "lambda_div": 1.0,
            "eta": 1.0,
        })

    return jobs


def job_id(spec: Dict[str, Any], seed: int) -> str:
    return (
        f"{spec['study']}__{spec['method'].lower()}"
        f"__d0_{spec['d0']:.4f}__seed_{seed}"
    )


def make_loaders(args: argparse.Namespace, seed: int):
    # Seed before DataLoader construction so shuffling is reproducible.
    seed_everything(seed)
    return get_cifar_loaders(
        root=args.data_root,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


def build_config(
    spec: Dict[str, Any],
    args: argparse.Namespace,
    num_classes: int,
) -> SMLConfig:

    method = spec["method"]

    common = dict(
        K=3,
        num_classes=num_classes,
        repr_dim=args.repr_dim,
        backbone="cifar_resnet18",
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        lambda_out=spec["lambda_out"],
        lambda_rep=spec["lambda_rep"],
        lambda_div=spec["lambda_div"],
        d0=spec["d0"],
        eta=spec["eta"],
        T=args.temperature,
        device=args.device,
        method="SML",
    )

    if method == "Independent":
        return SMLConfig(
            **common,
            use_out=False,
            use_rep=False,
            use_div=False,
            independent=True,
        )

    if method == "DML":
        return SMLConfig(
            **common,
            use_out=True,
            use_rep=False,
            use_div=False,
            independent=False,
        )

    if method == "SML":
        return SMLConfig(
            **common,
            use_out=True,
            use_rep=True,
            use_div=True,
            independent=False,
        )

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

METRICS = [
    "ensemble_acc",
    "best_single_acc",
    "D_rep",
    "E_d0",
    "representation_dispersion",
    "prediction_disagreement_rate",
    "prediction_symmetric_kl",
]


def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in runs:
        key = (r["study"], r["method"], float(r["d0"]))
        groups.setdefault(key, []).append(r)

    out = []
    for key, rr in sorted(groups.items()):
        study, method, d0 = key
        row: Dict[str, Any] = {
            "study": study,
            "method": method,
            "d0": d0,
            "n_seeds": len(rr),
            "seeds": sorted(int(r["seed"]) for r in rr),
            "backbone": "cifar_resnet18",
        }
        for metric in METRICS:
            vals = [
                float(r[metric])
                for r in rr
                if r.get(metric) is not None
            ]
            if vals:
                mu, sd = mean_std(vals)
                row[f"{metric}_mean"] = mu
                row[f"{metric}_std"] = sd
        out.append(row)
    return out


# ---------------------------------------------------------------------
# One resumable job
# ---------------------------------------------------------------------

def run_job(
    spec: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:

    jid = job_id(spec, seed)
    ckpt_path = Path(args.checkpoint_dir) / f"{jid}.pt"

    train_loader, test_loader, num_classes = make_loaders(args, seed)
    cfg = build_config(spec, args, num_classes)

    # Re-seed immediately before model construction. If a checkpoint exists,
    # load_checkpoint_if_valid() restores the exact saved RNG state afterward.
    seed_everything(seed)
    trainer = SMLTrainer(cfg, in_channels=3)

    loaded = load_checkpoint_if_valid(
        trainer=trainer,
        cfg=cfg,
        seed=seed,
        data_name=f"{args.dataset}:{spec['study']}:{spec['method']}:d0={spec['d0']}",
        checkpoint_path=ckpt_path,
    )

    if loaded:
        print(
            f"[resume] {jid}: epoch "
            f"{trainer.completed_epochs}/{args.epochs}"
        )

    start_epoch = int(trainer.completed_epochs)

    if start_epoch > args.epochs:
        raise RuntimeError(
            f"Checkpoint has {start_epoch} completed epochs, "
            f"but --epochs={args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs):
        for x, y in train_loader:
            trainer.step(x, y)

        # One scheduler step per completed epoch.
        trainer.scheduler.step()
        trainer.completed_epochs = epoch + 1

        save_checkpoint(
            trainer=trainer,
            cfg=cfg,
            seed=seed,
            data_name=f"{args.dataset}:{spec['study']}:{spec['method']}:d0={spec['d0']}",
            checkpoint_path=ckpt_path,
        )

        print(
            f"  [{jid}] epoch {epoch + 1:03d}/{args.epochs} "
            f"lr={trainer.optim.param_groups[0]['lr']:.6f}"
        )

    metrics = trainer.evaluate(test_loader)

    result: Dict[str, Any] = {
        "job_id": jid,
        "study": spec["study"],
        "method": spec["method"],
        "seed": seed,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "backbone": "cifar_resnet18",
        "epochs": args.epochs,
        "K": 3,
        "repr_dim": args.repr_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "lambda_out": spec["lambda_out"],
        "lambda_rep": spec["lambda_rep"],
        "lambda_div": spec["lambda_div"],
        "eta": spec["eta"],
        "d0": spec["d0"],
        **metrics,
    }

    # For geometry jobs, record the finite-K prediction explicitly.
    if spec["study"] == "geometry":
        w = 2.0 * spec["lambda_rep"] / spec["lambda_div"]
        unc = max(spec["d0"] ** 2 - w / spec["eta"], 0.0)
        cap = 2.0 * 3 / (3 - 1)
        result.update({
            "w_theory": w,
            "d_star_squared_unconstrained": unc,
            "finite_K_cap_squared": cap,
            "D_pred_squared": min(unc, cap),
        })

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Strong-backbone SML robustness study with epoch-level resume."
    )
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--repr-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--checkpoint-dir", default="checkpoints/exp6_resnet18")
    p.add_argument("--output", default="results/exp6_resnet18.json")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore prior JSON/checkpoints. Use only intentionally.",
    )
    args = p.parse_args()

    if args.seeds != 3:
        print(
            f"[note] --seeds={args.seeds}; the paper protocol is designed for 3."
        )

    if args.deterministic:
        enable_deterministic_mode()

    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        runs: List[Dict[str, Any]] = []
    elif output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            previous = json.load(f)
        runs = list(previous.get("runs", []))
        print(f"[resume] loaded {len(runs)} completed run(s) from {output_path}")
    else:
        runs = []

    completed = {r["job_id"] for r in runs if "job_id" in r}

    specs = job_specs()
    total = len(specs) * args.seeds

    print("=" * 76)
    print("Exp6 — Strong Backbone Robustness (CIFAR ResNet-18)")
    print(f"Planned runs : {total}")
    print(f"Epochs/run   : {args.epochs}")
    print(f"Seeds        : {args.seeds}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Checkpoint   : {checkpoint_dir}")
    print(f"Output       : {output_path}")
    print("=" * 76)

    for spec in specs:
        for seed in range(args.seeds):
            jid = job_id(spec, seed)

            if jid in completed:
                print(f"[skip completed] {jid}")
                continue

            print(f"\n[start] {jid}")
            result = run_job(spec, seed, args)
            runs.append(result)
            completed.add(jid)

            payload = {
                "experiment": "Exp6 — strong-backbone robustness",
                "config": {
                    "dataset": args.dataset,
                    "backbone": "cifar_resnet18",
                    "epochs": args.epochs,
                    "seeds": args.seeds,
                    "K": 3,
                    "repr_dim": args.repr_dim,
                    "batch_size": args.batch_size,
                    "workers": args.workers,
                    "lr": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "temperature": args.temperature,
                    "deterministic": args.deterministic,
                    "total_planned_runs": total,
                },
                "runs": runs,
                "aggregated": aggregate(runs),
            }
            atomic_json_save(output_path, payload)

            print(
                f"[done] {jid} | "
                f"ensemble={100*float(result['ensemble_acc']):.2f}% | "
                f"best-single={100*float(result['best_single_acc']):.2f}% | "
                f"D_rep={float(result['D_rep']):.4f}"
            )

    final_payload = {
        "experiment": "Exp6 — strong-backbone robustness",
        "config": {
            "dataset": args.dataset,
            "backbone": "cifar_resnet18",
            "epochs": args.epochs,
            "seeds": args.seeds,
            "K": 3,
            "repr_dim": args.repr_dim,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "deterministic": args.deterministic,
            "total_planned_runs": total,
        },
        "runs": runs,
        "aggregated": aggregate(runs),
    }
    atomic_json_save(output_path, final_payload)

    print("\n" + "=" * 76)
    print(f"Completed results: {len(runs)}/{total}")
    print(f"Saved to: {output_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
