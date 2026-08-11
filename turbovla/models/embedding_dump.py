from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import warnings
from typing import Any, Mapping

import torch
from torch import nn

from .configuration import EmbeddingDumpConfig


class EmbeddingDumper:
    """Stream exact PyTorch module I/O tensors to a trace consumable from C++."""

    FORMAT_VERSION = 2

    def __init__(self, config: EmbeddingDumpConfig) -> None:
        self.config = config
        self.forward_index = 0
        self.dump_index = 0
        self.rank = int(os.environ.get("RANK", "0"))
        self._payload: dict[str, Any] | None = None
        self._trace_dir: Path | None = None
        self.last_trace_dir: Path | None = None
        self._tensor_index = 0
        self._module_call_counts: defaultdict[str, int] = defaultdict(int)
        self._pending_module_calls: defaultdict[str, list[int]] = defaultdict(list)
        self._hook_handles: list[Any] = []

    @property
    def active(self) -> bool:
        return self._payload is not None

    def _new_trace_dir(self, forward_index: int) -> Path:
        output_dir = Path(self.config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_dir = output_dir / f"forward_{forward_index:06d}_rank_{self.rank}"
        collision_index = 1
        while trace_dir.exists():
            trace_dir = output_dir / (
                f"forward_{forward_index:06d}_rank_{self.rank}_copy_{collision_index}"
            )
            collision_index += 1
        trace_dir.mkdir(parents=True)
        (trace_dir / "tensors").mkdir()
        return trace_dir

    def begin(self, model: nn.Module | None = None) -> bool:
        selected = (
            self.config.enabled
            and (not self.config.rank_zero_only or self.rank == 0)
            and self.dump_index < self.config.max_dumps
            and self.forward_index % self.config.every_n_forwards == 0
        )
        current_forward = self.forward_index
        self.forward_index += 1
        if not selected:
            return False
        warnings.warn(
            "EmbeddingDumper is legacy-only; use TraceContext + TensorWriter for parity output",
            DeprecationWarning,
            stacklevel=2,
        )
        self._trace_dir = self._new_trace_dir(current_forward)
        self._tensor_index = 0
        self._module_call_counts.clear()
        self._pending_module_calls.clear()
        self._payload = {
            "metadata": {
                "format_version": self.FORMAT_VERSION,
                "forward_index": current_forward,
                "dump_index": self.dump_index,
                "rank": self.rank,
                "byte_order": sys.byteorder,
                "complete": False,
            },
            "module_calls": [],
            "tensors": {},
        }
        if model is not None:
            self._payload["model"] = self._describe_model(model)
        if model is not None and self.config.capture_module_io:
            self._attach_module_hooks(model)
        return True

    @staticmethod
    def _describe_model(model: nn.Module) -> dict[str, Any]:
        config = getattr(model, "config", None)
        config_payload = config.to_dict() if hasattr(config, "to_dict") else None
        modules = []
        for module_name, module in model.named_modules():
            parameters = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "requires_grad": bool(value.requires_grad),
                }
                for name, value in module.named_parameters(recurse=False)
            }
            buffers = {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in module.named_buffers(recurse=False)
            }
            modules.append(
                {
                    "module_name": module_name,
                    "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
                    "extra_repr": module.extra_repr(),
                    "parameters": parameters,
                    "buffers": buffers,
                }
            )
        return {
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": config_payload,
            "modules": modules,
        }

    def _attach_module_hooks(self, model: nn.Module) -> None:
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            pre_hook = module.register_forward_pre_hook(
                self._make_module_pre_hook(module_name),
                with_kwargs=True,
            )
            post_hook = module.register_forward_hook(
                self._make_module_post_hook(module_name),
                with_kwargs=True,
            )
            self._hook_handles.extend((pre_hook, post_hook))

    def _make_module_pre_hook(self, module_name: str):
        def capture(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self._payload is None:
                return
            local_call_index = self._module_call_counts[module_name]
            self._module_call_counts[module_name] += 1
            global_call_index = len(self._payload["module_calls"])
            prefix = f"module_calls.{global_call_index:06d}.{module_name}"
            self._payload["module_calls"].append(
                {
                    "global_call_index": global_call_index,
                    "module_call_index": local_call_index,
                    "module_name": module_name,
                    "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
                    "inputs": self._capture_value(args, f"{prefix}.inputs"),
                    "kwargs": self._capture_value(kwargs, f"{prefix}.kwargs"),
                    "output": None,
                }
            )
            self._pending_module_calls[module_name].append(global_call_index)

        return capture

    def _make_module_post_hook(self, module_name: str):
        def capture(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
            if self._payload is None:
                return
            global_call_index = self._pending_module_calls[module_name].pop()
            call = self._payload["module_calls"][global_call_index]
            prefix = f"module_calls.{global_call_index:06d}.{module_name}"
            call["output"] = self._capture_value(output, f"{prefix}.output")

        return capture

    def _capture_value(self, value: Any, name: str) -> dict[str, Any]:
        if isinstance(value, torch.Tensor):
            tensor_name = self.record(name, value, print_summary=False)
            return {"kind": "tensor", "name": tensor_name}
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "items": {
                    str(key): self._capture_value(item, f"{name}.{key}")
                    for key, item in value.items()
                },
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [self._capture_value(item, f"{name}.{index}") for index, item in enumerate(value)],
            }
        if isinstance(value, list):
            return {
                "kind": "list",
                "items": [self._capture_value(item, f"{name}.{index}") for index, item in enumerate(value)],
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": "value", "value": value}
        return {"kind": "repr", "value": repr(value)}

    @staticmethod
    def _raw_bytes(value: torch.Tensor) -> tuple[bytes, str]:
        contiguous = value.contiguous()
        if contiguous.dtype == torch.bfloat16:
            return contiguous.view(torch.uint16).numpy().tobytes(order="C"), "uint16"
        try:
            array = contiguous.numpy()
            return array.tobytes(order="C"), str(array.dtype)
        except (TypeError, RuntimeError):
            return contiguous.view(torch.uint8).numpy().tobytes(order="C"), "uint8"

    def record(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        print_summary: bool | None = None,
    ) -> str:
        if self._payload is None or self._trace_dir is None:
            return name
        # copy=True snapshots this exact boundary even if a later op mutates the CPU tensor in-place.
        value = tensor.detach().to(device="cpu", copy=True)
        stats_value = value.float()
        finite = torch.isfinite(stats_value)
        finite_values = stats_value[finite]
        stats: dict[str, Any] = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "source_device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "numel": value.numel(),
            "finite": int(finite.sum().item()),
        }
        if finite_values.numel():
            stats.update(
                min=float(finite_values.min().item()),
                max=float(finite_values.max().item()),
                mean=float(finite_values.mean().item()),
                std=float(finite_values.std(unbiased=False).item()),
            )

        tensor_id = self._tensor_index
        self._tensor_index += 1
        stem = f"tensor_{tensor_id:08d}"
        entry: dict[str, Any] = {"id": tensor_id, "stats": stats}
        if self.config.save_pt:
            pt_path = Path("tensors") / f"{stem}.pt"
            torch.save(value, self._trace_dir / pt_path)
            entry["pt_file"] = pt_path.as_posix()
        if self.config.save_raw:
            raw_path = Path("tensors") / f"{stem}.bin"
            raw_bytes, storage_dtype = self._raw_bytes(value)
            (self._trace_dir / raw_path).write_bytes(raw_bytes)
            entry.update(
                raw_file=raw_path.as_posix(),
                storage_dtype=storage_dtype,
                byte_length=len(raw_bytes),
            )
        self._payload["tensors"][name] = entry

        should_print = self.config.print_summary if print_summary is None else print_summary
        if should_print:
            preview = value.flatten()[: self.config.preview_values].tolist()
            print(
                f"[TurboVLA dump] {name}: shape={stats['shape']} dtype={stats['dtype']} "
                f"min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')} "
                f"preview={preview}",
                flush=True,
            )
        return name

    def _remove_hooks(self) -> None:
        for hook in self._hook_handles:
            hook.remove()
        self._hook_handles.clear()

    def finish(self, error: str | None = None) -> Path | None:
        self._remove_hooks()
        if self._payload is None or self._trace_dir is None:
            return None
        if error is not None:
            self._payload["metadata"]["error"] = error
        self._payload["metadata"]["complete"] = error is None
        manifest_path = self._trace_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(self._payload, self._trace_dir / "trace_index.pt")
        trace_dir = self._trace_dir
        self.last_trace_dir = trace_dir
        self._payload = None
        self._trace_dir = None
        self.dump_index += 1
        print(f"[TurboVLA dump] saved exhaustive trace {trace_dir}", flush=True)
        return trace_dir
