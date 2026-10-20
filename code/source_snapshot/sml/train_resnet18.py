"""
train.py
========
Training/checkpoint engine for Experiment 3.

Design goals:
- All fixed-architecture cohort baselines use the same project backbone,
  optimizer, LR schedule, batch size, augmentation and training budget.
- WML keeps its four-head architecture because that is intrinsic to the
  supplied official WML mechanism; it is explicitly an adapted WML baseline.
- Checkpoints are written atomically after every epoch.
- A run can resume after a power loss without restarting completed epochs.
- Python, NumPy, PyTorch CPU, and CUDA RNG states are stored and restored.
- Older checkpoints whose RNG states were serialized as Python lists are
  accepted and converted back to the required ByteTensor representation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_resnet18 import build_cohort
from .losses import (
    total_objective,
    coop_strength,
    rho as rho_fn,
    d_star_squared,
)
from .metrics import (
    mean_pairwise_squared_distance,
    target_deviation,
    batch_averaged_dispersion,
    prediction_disagreement,
    ensemble_accuracy,
)
from .baselines_extra import (
    wml_task_and_peer_losses,
    wml_hypergradient_update,
    WMLFourHeadResNet18,
    diffdml_dpl_objective,
    differentiated_training_views,
    ncl_inspired_objective,
    kd_kl,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SMLConfig:
    K: int = 3
    num_classes: int = 10
    repr_dim: int = 64
    widths: Optional[list] = None
    backbone: str = "small_cnn"

    # ------------------------------------------------------------------
    # Common training recipe for fixed-backbone methods.
    # ------------------------------------------------------------------
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    epochs: int = 200
    warmup_epochs: int = 5

    # ------------------------------------------------------------------
    # SML / fixed-backbone losses.
    # ------------------------------------------------------------------
    lambda_out: float = 0.5
    lambda_rep: float = 0.1
    lambda_div: float = 0.1
    d0: float = 1.0
    eta: float = 1.0
    T: float = 4.0

    use_out: bool = True
    use_rep: bool = True
    use_div: bool = True
    independent: bool = False
    device: str = "cpu"

    method: str = "SML"

    # ------------------------------------------------------------------
    # WML settings adapted from the supplied official run50 mechanism.
    # ------------------------------------------------------------------
    wml_alpha: float = 0.1
    wml_eta: float = 0.1
    wml_update_w_step: int = 10
    wml_T: float = 3.0
    wml_prune_head_probs: tuple = (0.0, 0.2, 0.5, 0.8)

    # ------------------------------------------------------------------
    # Diff-DML mechanism-level adapter.
    # ------------------------------------------------------------------
    diffdml_margin: float = 0.5
    diffdml_lambda_dpl: float = 1.0
    dts_base_strength: float = 0.0
    dts_step_strength: float = 0.15

    # ------------------------------------------------------------------
    # NCL-inspired baseline.
    # ------------------------------------------------------------------
    ncl_strength: float = 0.05

    @property
    def w(self):
        return coop_strength(self.lambda_rep, self.lambda_div)

    @property
    def rho(self):
        return rho_fn(self.d0, self.w, eta=self.eta)

    @property
    def d_star_squared(self):
        return d_star_squared(self.d0, self.w, eta=self.eta)


# ============================================================================
# Reproducibility / RNG helpers
# ============================================================================

def _rng_state():
    """
    Capture all RNG states relevant to the training process.

    torch.get_rng_state() returns a CPU ByteTensor.
    torch.cuda.get_rng_state_all() returns one ByteTensor per CUDA device.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def _as_byte_tensor(value):
    """
    Convert an arbitrary serialized RNG state into a CPU uint8 Tensor.

    This is intentionally tolerant of older checkpoints where a Tensor may
    have been converted into a Python list before being saved.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)

    try:
        return torch.as_tensor(
            value,
            dtype=torch.uint8,
            device="cpu",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not convert a serialized PyTorch RNG state to "
            "torch.uint8 ByteTensor."
        ) from exc


def _restore_numpy_state(state):
    """
    Restore NumPy RNG state.

    np.random.get_state() normally returns:
        (
            bit_generator_name,
            ndarray[uint32],
            position,
            has_gauss,
            cached_gaussian
        )

    Older checkpoints may contain the ndarray as a Python list.
    """
    if state is None:
        return

    if not isinstance(state, (list, tuple)) or len(state) != 5:
        raise RuntimeError(
            "Invalid NumPy RNG state in checkpoint."
        )

    state = list(state)

    if not isinstance(state[1], np.ndarray):
        state[1] = np.asarray(state[1], dtype=np.uint32)
    else:
        state[1] = state[1].astype(np.uint32, copy=False)

    np.random.set_state(tuple(state))


def _set_rng_state(state):
    """
    Restore Python, NumPy, PyTorch CPU, and CUDA RNG states.

    The function is deliberately backward-compatible with checkpoints where
    PyTorch RNG tensors were serialized as Python lists.

    Missing RNG information is tolerated for backward compatibility.
    """
    if not state:
        return

    if not isinstance(state, dict):
        raise RuntimeError(
            "Invalid RNG state object in checkpoint."
        )

    # ------------------------------------------------------------------
    # Python RNG
    # ------------------------------------------------------------------
    python_state = state.get("python")

    if python_state is not None:
        try:
            random.setstate(python_state)
        except Exception as exc:
            raise RuntimeError(
                "Could not restore Python RNG state from checkpoint."
            ) from exc

    # ------------------------------------------------------------------
    # NumPy RNG
    # ------------------------------------------------------------------
    numpy_state = state.get("numpy")

    if numpy_state is not None:
        try:
            _restore_numpy_state(numpy_state)
        except Exception as exc:
            raise RuntimeError(
                "Could not restore NumPy RNG state from checkpoint."
            ) from exc

    # ------------------------------------------------------------------
    # PyTorch CPU RNG
    # ------------------------------------------------------------------
    torch_state = state.get("torch")

    if torch_state is not None:
        torch_state = _as_byte_tensor(torch_state)

        if torch_state.ndim != 1:
            torch_state = torch_state.reshape(-1)

        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:
            raise RuntimeError(
                "Could not restore PyTorch CPU RNG state from checkpoint."
            ) from exc

    # ------------------------------------------------------------------
    # PyTorch CUDA RNG
    # ------------------------------------------------------------------
    cuda_states = state.get("cuda")

    if torch.cuda.is_available() and cuda_states is not None:

        if isinstance(cuda_states, torch.Tensor):
            cuda_states = [cuda_states]

        elif not isinstance(cuda_states, (list, tuple)):
            cuda_states = [cuda_states]

        normalized_cuda_states = []

        for cuda_state in cuda_states:
            cuda_state = _as_byte_tensor(cuda_state)

            if cuda_state.ndim != 1:
                cuda_state = cuda_state.reshape(-1)

            normalized_cuda_states.append(cuda_state)

        n_gpu = torch.cuda.device_count()

        if len(normalized_cuda_states) > n_gpu:
            normalized_cuda_states = normalized_cuda_states[:n_gpu]

        if normalized_cuda_states:
            try:
                torch.cuda.set_rng_state_all(
                    normalized_cuda_states
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not restore PyTorch CUDA RNG state from checkpoint."
                ) from exc


# ============================================================================
# Checkpoint utilities
# ============================================================================

def _atomic_torch_save(obj, path: Path):
    """
    Atomically save a PyTorch checkpoint.

    The temporary file is written first and then replaced into place.
    Therefore an interrupted write should not leave a partially written
    checkpoint at the final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def config_signature(
    cfg: SMLConfig,
    seed: int,
    data_name: str,
) -> str:
    """
    Create a compact signature identifying the experimental configuration.

    Device and epoch count are deliberately excluded.

    Excluding epochs permits a checkpoint produced during a shorter run to
    remain structurally compatible with a later run using a larger epoch
    budget.
    """
    d = asdict(cfg)

    d.pop("device", None)
    d.pop("epochs", None)

    payload = json.dumps(
        {
            "cfg": d,
            "seed": seed,
            "data": data_name,
        },
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================================
# Trainer
# ============================================================================

class SMLTrainer:

    def __init__(
        self,
        cfg: SMLConfig,
        in_channels=3,
    ):
        self.cfg = cfg
        self.is_wml = cfg.method.upper() == "WML"
        self.completed_epochs = 0

        # ==============================================================
        # WML
        # ==============================================================

        if self.is_wml:

            if cfg.K != 4:
                raise ValueError(
                    "WML uses K=4 in this experiment."
                )

            self.models = WMLFourHeadResNet18(
                num_classes=cfg.num_classes
            ).to(cfg.device)

            self.optim = torch.optim.SGD(
                self.models.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim,
                T_max=max(cfg.epochs, 1),
            )

            self.wml_weights = (
                torch.ones(
                    4,
                    device=cfg.device,
                )
                / 4.0
            )

            self._wml_epoch = 0

            return

        # ==============================================================
        # Fixed-backbone cohort
        # ==============================================================

        self.models = build_cohort(
            cfg.K,
            num_classes=cfg.num_classes,
            repr_dim=cfg.repr_dim,
            widths=cfg.widths,
            in_channels=in_channels,
            backbone=cfg.backbone,
        ).to(cfg.device)

        self.optim = torch.optim.SGD(
            self.models.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=max(cfg.epochs, 1),
        )

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def state_dict(self):
        state = {
            "models": self.models.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "completed_epochs": int(self.completed_epochs),
            "rng": _rng_state(),
        }

        if self.is_wml:
            state.update(
                {
                    "wml_weights": (
                        self.wml_weights
                        .detach()
                        .cpu()
                    ),
                    "wml_epoch": int(self._wml_epoch),
                }
            )

        return state

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise RuntimeError(
                "Invalid trainer state in checkpoint."
            )

        # --------------------------------------------------------------
        # Models
        # --------------------------------------------------------------

        if "models" not in state:
            raise RuntimeError(
                "Checkpoint does not contain model state."
            )

        self.models.load_state_dict(
            state["models"]
        )

        # --------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------

        if "optim" not in state:
            raise RuntimeError(
                "Checkpoint does not contain optimizer state."
            )

        self.optim.load_state_dict(
            state["optim"]
        )

        # --------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------

        if "scheduler" not in state:
            raise RuntimeError(
                "Checkpoint does not contain scheduler state."
            )

        self.scheduler.load_state_dict(
            state["scheduler"]
        )

        # --------------------------------------------------------------
        # Epoch counter
        # --------------------------------------------------------------

        self.completed_epochs = int(
            state.get("completed_epochs", 0)
        )

        if self.completed_epochs < 0:
            raise RuntimeError(
                "Invalid negative completed_epochs in checkpoint."
            )

        # --------------------------------------------------------------
        # WML-specific state
        # --------------------------------------------------------------

        if self.is_wml:

            if "wml_weights" in state:
                self.wml_weights = (
                    state["wml_weights"]
                    .detach()
                    .to(self.cfg.device)
                )

            else:
                # Backward-compatible fallback.
                self.wml_weights = (
                    torch.ones(
                        4,
                        device=self.cfg.device,
                    )
                    / 4.0
                )

            self._wml_epoch = int(
                state.get(
                    "wml_epoch",
                    self.completed_epochs,
                )
            )

        # --------------------------------------------------------------
        # RNG state
        # --------------------------------------------------------------

        _set_rng_state(
            state.get("rng")
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_all(self, x):
        hs = []
        zs = []

        for model in self.models:
            h, z = model(x)
            hs.append(h)
            zs.append(z)

        return hs, zs

    # ------------------------------------------------------------------
    # One optimization step
    # ------------------------------------------------------------------

    def step(self, x, y):

        cfg = self.cfg

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        # ==============================================================
        # WML
        # ==============================================================

        if self.is_wml:

            self.optim.zero_grad(
                set_to_none=True
            )

            z_list = self.models(x)

            total, ce, kd = wml_task_and_peer_losses(
                z_list,
                y,
                self.wml_weights,
                T=cfg.wml_T,
                alpha=cfg.wml_alpha,
            )

            total.backward()

            self.optim.step()

            self._wml_epoch += 1

            # Official mechanism updates weights every N optimization steps.
            if (
                self._wml_epoch
                % cfg.wml_update_w_step
                == 0
            ):
                self.wml_weights = (
                    wml_hypergradient_update(
                        self.models,
                        x,
                        y,
                        self.wml_weights,
                        lr=self.optim.param_groups[0]["lr"],
                        eta=cfg.wml_eta,
                        T=cfg.wml_T,
                        alpha=cfg.wml_alpha,
                    )
                )

            return (
                {
                    "total": total.item(),
                    "task": [
                        v.item()
                        for v in ce
                    ],
                    "out": [
                        sum(
                            kd[k][j]
                            for j in range(4)
                            if j != k
                        ).item()
                        for k in range(4)
                    ],
                    "rep": [0.0] * 4,
                    "div": 0.0,
                    "wml_weights": (
                        self.wml_weights
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                },
                [None] * 4,
                [
                    z.detach()
                    for z in z_list
                ],
            )

        # ==============================================================
        # Independent baseline
        # ==============================================================

        if cfg.independent:

            self.optim.zero_grad(
                set_to_none=True
            )

            hs, zs = self._forward_all(x)

            losses = [
                F.cross_entropy(z, y)
                for z in zs
            ]

            total = sum(losses)

            total.backward()

            self.optim.step()

            return (
                {
                    "total": total.item(),
                    "task": [
                        v.item()
                        for v in losses
                    ],
                    "out": [0.0] * cfg.K,
                    "rep": [0.0] * cfg.K,
                    "div": 0.0,
                },
                [
                    h.detach()
                    for h in hs
                ],
                [
                    z.detach()
                    for z in zs
                ],
            )

        # ==============================================================
        # Common optimizer reset
        # ==============================================================

        self.optim.zero_grad(
            set_to_none=True
        )

        # ==============================================================
        # Diff-DML
        # ==============================================================

        if cfg.method.upper() == "DIFFDML":

            xs = [
                differentiated_training_views(
                    x,
                    k,
                    base_strength=cfg.dts_base_strength,
                    step_strength=cfg.dts_step_strength,
                    training=True,
                )
                for k in range(cfg.K)
            ]

            hs = []
            zs = []

            for model, xk in zip(
                self.models,
                xs,
            ):
                h, z = model(xk)
                hs.append(h)
                zs.append(z)

            task = [
                F.cross_entropy(z, y)
                for z in zs
            ]

            out = [
                sum(
                    kd_kl(
                        zs[k],
                        zs[j],
                        T=cfg.T,
                    )
                    for j in range(cfg.K)
                    if j != k
                )
                for k in range(cfg.K)
            ]

            dpl = diffdml_dpl_objective(
                zs,
                margin=cfg.diffdml_margin,
            )

            total = (
                sum(task)
                + cfg.lambda_out * sum(out)
                + cfg.diffdml_lambda_dpl * dpl
            )

            total.backward()

            self.optim.step()

            return (
                {
                    "total": total.item(),
                    "task": [
                        v.item()
                        for v in task
                    ],
                    "out": [
                        v.item()
                        for v in out
                    ],
                    "rep": [0.0] * cfg.K,
                    "div": dpl.item(),
                },
                [
                    h.detach()
                    for h in hs
                ],
                [
                    z.detach()
                    for z in zs
                ],
            )

        # ==============================================================
        # NCL-inspired
        # ==============================================================

        if cfg.method.upper() in {
            "NCL",
            "NCL_INSPIRED",
            "NCL-INSPIRED",
        }:

            hs, zs = self._forward_all(x)

            total = ncl_inspired_objective(
                hs,
                zs,
                y,
                lambda_out=cfg.lambda_out,
                strength=cfg.ncl_strength,
                T=cfg.T,
            )

            total.backward()

            self.optim.step()

            return (
                {
                    "total": total.item(),
                    "task": [
                        F.cross_entropy(z, y).item()
                        for z in zs
                    ],
                    "out": [0.0] * cfg.K,
                    "rep": [0.0] * cfg.K,
                    "div": 0.0,
                },
                [
                    h.detach()
                    for h in hs
                ],
                [
                    z.detach()
                    for z in zs
                ],
            )

        # ==============================================================
        # Standard SML / DML / Rep-only / Diversity-only
        # ==============================================================

        hs, zs = self._forward_all(x)

        out = total_objective(
            hs,
            zs,
            y,
            lambda_out=cfg.lambda_out,
            lambda_rep=cfg.lambda_rep,
            lambda_div=cfg.lambda_div,
            d0=cfg.d0,
            eta=cfg.eta,
            T=cfg.T,
            use_out=cfg.use_out,
            use_rep=cfg.use_rep,
            use_div=cfg.use_div,
        )

        out.total.backward()

        self.optim.step()

        return (
            {
                "total": out.total.item(),
                "task": [
                    v.item()
                    for v in out.task
                ],
                "out": [
                    v.item()
                    if torch.is_tensor(v)
                    else v
                    for v in out.out
                ],
                "rep": [
                    v.item()
                    if torch.is_tensor(v)
                    else v
                    for v in out.rep
                ],
                "div": out.div.item(),
            },
            [
                h.detach()
                for h in hs
            ],
            [
                z.detach()
                for z in zs
            ],
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader):

        cfg = self.cfg

        self.models.eval()

        K = 4 if self.is_wml else cfg.K

        n = 0.0
        ens_correct = 0.0

        single_correct = [
            0.0
        ] * K

        d_rep_sum = 0.0
        e_d0_sum = 0.0
        disp_sum = 0.0
        disagree_sum = 0.0
        skl_sum = 0.0

        nb = 0

        for x, y in loader:

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            # ----------------------------------------------------------
            # WML evaluation
            # ----------------------------------------------------------

            if self.is_wml:

                z_list = list(
                    self.models(x)
                )

                w = self.wml_weights.to(
                    x.device
                )

                z_ens = sum(
                    w[k] * z_list[k]
                    for k in range(4)
                )

                ens_correct += (
                    z_ens.argmax(-1) == y
                ).float().sum().item()

                h_list = None

            # ----------------------------------------------------------
            # Fixed-backbone evaluation
            # ----------------------------------------------------------

            else:

                h_list, z_list = (
                    self._forward_all(x)
                )

                ens_correct += (
                    ensemble_accuracy(
                        z_list,
                        y,
                    )
                    * y.size(0)
                )

            n += y.size(0)

            for k, z in enumerate(z_list):

                single_correct[k] += (
                    (z.argmax(-1) == y)
                    .float()
                    .sum()
                    .item()
                )

            # ----------------------------------------------------------
            # Representation metrics
            # ----------------------------------------------------------

            if not self.is_wml:

                d_rep_sum += (
                    mean_pairwise_squared_distance(
                        h_list
                    )
                )

                disp_sum += (
                    batch_averaged_dispersion(
                        h_list
                    )
                )

                if cfg.method.upper() in {
                    "SML",
                    "DIVERSITY_ONLY",
                }:

                    e_d0_sum += (
                        target_deviation(
                            h_list,
                            cfg.d0,
                        )
                    )

            # ----------------------------------------------------------
            # Prediction disagreement
            # ----------------------------------------------------------

            dis, skl = prediction_disagreement(
                z_list
            )

            disagree_sum += dis
            skl_sum += skl
            nb += 1

        self.models.train()

        finite_target = (
            cfg.method.upper()
            in {
                "SML",
                "DIVERSITY_ONLY",
            }
        )

        result = {
            "ensemble_acc": (
                ens_correct
                / max(n, 1)
            ),
            "best_single_acc": max(
                v / max(n, 1)
                for v in single_correct
            ),
            "per_model_acc": [
                v / max(n, 1)
                for v in single_correct
            ],
            "prediction_disagreement_rate": (
                disagree_sum
                / max(nb, 1)
            ),
            "prediction_symmetric_kl": (
                skl_sum
                / max(nb, 1)
            ),
            "rho": (
                cfg.rho
                if not self.is_wml
                else None
            ),
            "w": (
                cfg.w
                if not self.is_wml
                else None
            ),
            "eta": (
                cfg.eta
                if not self.is_wml
                else None
            ),
            "d_star_squared_theory": (
                cfg.d_star_squared
                if finite_target
                else None
            ),
        }

        if self.is_wml:

            result.update(
                {
                    "D_rep": None,
                    "E_d0": None,
                    "representation_dispersion": None,
                    "wml_weights": (
                        self.wml_weights
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                }
            )

        else:

            result.update(
                {
                    "D_rep": (
                        d_rep_sum
                        / max(nb, 1)
                    ),
                    "E_d0": (
                        e_d0_sum
                        / max(nb, 1)
                        if finite_target
                        else None
                    ),
                    "representation_dispersion": (
                        disp_sum
                        / max(nb, 1)
                    ),
                }
            )

        return result


# ============================================================================
# Checkpoint save / load
# ============================================================================

def save_checkpoint(
    trainer: SMLTrainer,
    cfg: SMLConfig,
    seed: int,
    data_name: str,
    checkpoint_path: Path,
):
    """
    Save a complete resumable checkpoint.

    The checkpoint contains:
      - model parameters
      - optimizer state
      - LR scheduler state
      - completed epoch
      - Python RNG
      - NumPy RNG
      - PyTorch CPU RNG
      - CUDA RNG states
      - WML-specific state when applicable
      - configuration signature
    """

    payload = {
        "version": 3,
        "signature": config_signature(
            cfg,
            seed,
            data_name,
        ),
        "seed": seed,
        "data": data_name,
        "config": asdict(cfg),
        "trainer": trainer.state_dict(),
    }

    _atomic_torch_save(
        payload,
        checkpoint_path,
    )


def _validate_checkpoint_payload(
    payload,
    checkpoint_path: Path,
):
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid checkpoint structure: "
            f"{checkpoint_path}"
        )

    if "trainer" not in payload:
        raise RuntimeError(
            f"Checkpoint does not contain trainer state: "
            f"{checkpoint_path}"
        )

    version = int(
        payload.get("version", 1)
    )

    if version < 1:
        raise RuntimeError(
            f"Invalid checkpoint version {version}: "
            f"{checkpoint_path}"
        )

    return version


def load_checkpoint_if_valid(
    trainer: SMLTrainer,
    cfg: SMLConfig,
    seed: int,
    data_name: str,
    checkpoint_path: Path,
):
    """
    Load a checkpoint if it exists and its experiment signature matches.

    Returns:
        True  -- checkpoint was loaded.
        False -- no checkpoint exists.

    Older checkpoints are accepted as long as their experiment signature
    matches. RNG states are normalized by _set_rng_state().
    """

    if not checkpoint_path.exists():
        return False

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=cfg.device,
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint: "
            f"{checkpoint_path}"
        ) from exc

    _validate_checkpoint_payload(
        payload,
        checkpoint_path,
    )

    expected = config_signature(
        cfg,
        seed,
        data_name,
    )

    actual = payload.get(
        "signature"
    )

    if actual != expected:

        raise RuntimeError(
            f"Checkpoint signature mismatch for "
            f"{checkpoint_path}.\n"
            f"Expected: {expected}\n"
            f"Found:    {actual}\n"
            f"The previous run used different experiment "
            f"settings.\n"
            f"Use a new --checkpoint-dir or remove the old "
            f"checkpoint."
        )

    try:
        trainer.load_state_dict(
            payload["trainer"]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint exists and has a matching experiment "
            f"signature, but its trainer state could not be restored:\n"
            f"{checkpoint_path}"
        ) from exc

    return True