from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def emit(tracer: Any | None, name: str, value: torch.Tensor, layout: str = "", operation: str = "unknown",
         level: str = "op", module_path: str | None = None) -> None:
    if tracer is not None and tracer.active:
        tracer.tensor(name, value, layout=layout, operation=operation, required_level=level, module_path=module_path)


def trace_linear(tracer: Any | None, prefix: str, value: torch.Tensor, layer: nn.Linear, *,
                 layout: str = "B,N,D", level: str = "exhaustive", names: tuple[str, str] = ("matmul", "bias_add")) -> torch.Tensor:
    # Keep the decomposed matmul as an exhaustive diagnostic, but return and
    # emit the native Linear result.  Under CUDA autocast, F.linear dispatches
    # a different GEMM/epilogue than `matmul + bias`; using the decomposition
    # as the operational value makes the trace describe math the model did
    # not execute.
    matmul = torch.matmul(value, layer.weight.transpose(-1, -2))
    emit(tracer, f"{prefix}.{names[0]}", matmul, layout, "matmul", level)
    output = F.linear(value, layer.weight, layer.bias)
    emit(tracer, f"{prefix}.{names[1]}", output, layout, "add", level)
    return output


def trace_layer_norm(tracer: Any | None, prefix: str, value: torch.Tensor, layer: nn.LayerNorm, *,
                     layout: str = "B,N,D", level: str = "exhaustive", include_normalized: bool = True,
                     include_parameters: bool = False) -> torch.Tensor:
    mean = value.mean(dim=-1, keepdim=True)
    variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(variance + layer.eps)
    normalized = (value - mean) * inv_std
    # The statistics above are useful diagnostics.  The operational output
    # must come from the same vectorized Welford + affine kernel used by the
    # model forward, rather than from this scalar decomposition.
    output = F.layer_norm(value, layer.normalized_shape, layer.weight, layer.bias, layer.eps)
    emit(tracer, f"{prefix}.input", value, layout, "identity", level)
    emit(tracer, f"{prefix}.mean", mean, layout.replace("D", "1"), "mean", level)
    emit(tracer, f"{prefix}.variance", variance, layout.replace("D", "1"), "variance", level)
    emit(tracer, f"{prefix}.inv_std", inv_std, layout.replace("D", "1"), "rsqrt", level)
    if include_normalized:
        emit(tracer, f"{prefix}.normalized", normalized, layout, "multiply", level)
    if include_parameters and layer.elementwise_affine:
        emit(tracer, f"{prefix}.weight", layer.weight, "D", "parameter", level)
        emit(tracer, f"{prefix}.bias", layer.bias, "D", "parameter", level)
    emit(tracer, f"{prefix}.output", output, layout, "layer_norm", level)
    return output


def trace_softmax(tracer: Any | None, prefix: str, logits: torch.Tensor, *, layout: str,
                  level: str = "exhaustive", output_dtype: torch.dtype | None = None) -> torch.Tensor:
    max_value = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_value
    exp = shifted.float().exp()
    total = exp.sum(dim=-1, keepdim=True)
    # Emit decomposed terms for inspection, but use the native softmax kernel
    # as the value consumed by subsequent traced operations.
    probs = F.softmax(logits, dim=-1, dtype=output_dtype)
    emit(tracer, f"{prefix}.max", max_value, layout, "max", level)
    emit(tracer, f"{prefix}.shifted", shifted, layout, "subtract", level)
    emit(tracer, f"{prefix}.exp", exp, layout, "exp", level)
    emit(tracer, f"{prefix}.sum", total, layout, "sum", level)
    emit(tracer, f"{prefix}.probs", probs, layout, "softmax", "op")
    return probs


def trace_activation(tracer: Any | None, prefix: str, value: torch.Tensor, activation, *,
                     layout: str = "B,N,D", name: str = "activation", level: str = "exhaustive") -> torch.Tensor:
    output = activation(value)
    emit(tracer, f"{prefix}.{name}.input", value, layout, name, level)
    emit(tracer, f"{prefix}.{name}.output", output, layout, name, level)
    return output


def tensor_axis(value: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int64, device=device)
