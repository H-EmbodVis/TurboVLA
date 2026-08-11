from __future__ import annotations

import contextlib
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeterministicProfile:
    name: str
    deterministic: bool


PRODUCTION_BF16 = DeterministicProfile("production_bf16", False)
DETERMINISTIC_BF16 = DeterministicProfile("deterministic_bf16", True)


@contextlib.contextmanager
def deterministic_context(seed: int, *, enabled: bool = True):
    if not enabled:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.use_deterministic_algorithms(True, warn_only=True)
            yield caught
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
