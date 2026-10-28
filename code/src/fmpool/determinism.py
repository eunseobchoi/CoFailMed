"""Seed / RNG-state / checkpoint-hash utilities.

All training and evaluation scripts must call :func:`set_seed` before any
stochastic operation. :func:`get_rng_state` / :func:`set_rng_state` support
checkpointable randomness. :func:`sha256_file` hashes checkpoints for the
``checkpoint_sha256`` field in per-case-Dice JSONs.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import os
import random
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 1 << 20  # 1 MiB
_CUBLAS_CONFIG = ":4096:8"


def set_seed(seed: int) -> None:
    """Set every RNG and *request* deterministic kernels (warn-only).

    Sets ``random``, ``numpy``, ``torch`` CPU / CUDA seeds, disables cuDNN
    benchmark and TF32, sets the cuBLAS workspace configuration needed by
    ``torch.use_deterministic_algorithms`` on Ampere+ GPUs, and calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)``. With
    ``warn_only`` an op lacking a deterministic implementation (which ops
    depends on the PyTorch version/device) only warns, so bit-identical reruns
    are not guaranteed, e.g. across GPU models.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Required for deterministic matmul on Ampere+.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_CONFIG
    # Request, not enforce: warn_only=True warns (does not raise) on nondeterministic ops.
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("set_seed: seed=%d CUDA=%s", seed, torch.cuda.is_available())


def get_rng_state() -> dict[str, Any]:
    """Snapshot of Python/NumPy/Torch RNG state for checkpointing."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state produced by :func:`get_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def sha256_file(path: str | os.PathLike[str], chunk_size: int = _DEFAULT_CHUNK) -> str:
    """SHA-256 of a file on disk (for checkpoint provenance).

    Reads ``path`` in streaming chunks so that multi-GB checkpoints hash
    without blowing memory. Returns the hex digest.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sha256_file: not a regular file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class SeededShuffle(torch.utils.data.Sampler):
    """Exactly one randperm per epoch, matching the cached-head recipe."""

    def __init__(self, data, seed):
        self.n = len(data)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def __iter__(self):
        return iter(torch.randperm(self.n, generator=self.generator).tolist())

    def __len__(self):
        return self.n


class NumericalHealthError(FloatingPointError):
    """A failed numerical boundary; never a model-performance observation."""


def require_finite(value: Any, label: str) -> None:
    """Reject nonfinite tensors, including nested checkpoint state."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise NumericalHealthError(
                f"{label}: nonfinite tensor (shape={tuple(value.shape)}, dtype={value.dtype})"
            )
    elif isinstance(value, dict):
        for name, item in value.items():
            require_finite(item, f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")


def checked_step(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    """Keep the optimizer recipe unchanged; fail before/after its update."""
    require_finite(loss, "loss")
    loss.backward()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            require_finite(parameter.grad, "gradient before step")
    optimizer.step()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            require_finite(parameter, "updated parameter")
    require_finite(list(optimizer.state.values()), "updated optimizer state")


@contextmanager
def numerical_failure_record(path: Path, **context):
    """Preserve a failed-cell diagnostic and re-raise; never emit scores/retry."""
    try:
        yield
    except NumericalHealthError as exc:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique sidecars preserve every failure and existing successful artifacts.
        fd, name = tempfile.mkstemp(prefix=path.name + ".failed.", suffix=".json",
                                    dir=path.parent)
        with os.fdopen(fd, "w") as handle:
            json.dump({"status": "failed", "failure_type": "numerical_health",
                       "context": context, "error": str(exc),
                       "traceback": traceback.format_exc()}, handle, indent=2)
        logger.error("numerical failure recorded in %s", name)
        raise
