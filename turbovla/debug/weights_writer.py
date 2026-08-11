from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .trace_utils import canonical_f32_bytes, compute_tensor_stats, native_bytes, sanitize_filename, sha256_of_bytes


def _layout(module: nn.Module | None, name: str, tensor: torch.Tensor) -> str:
    if isinstance(module, nn.Linear) and name.endswith("weight"):
        return "OUT,IN"
    if isinstance(module, nn.Conv2d) and name.endswith("weight"):
        return "OUT,IN,KH,KW"
    if isinstance(module, nn.Embedding) and name.endswith("weight"):
        return "VOCAB,D"
    if isinstance(module, nn.LayerNorm):
        return "D"
    return ",".join(f"D{index}" for index in range(tensor.ndim))


def _module_for_name(model: nn.Module, tensor_name: str) -> tuple[str, nn.Module | None]:
    module_path = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else ""
    return module_path, model.get_submodule(module_path) if module_path else model


def write_weights_and_buffers(model: nn.Module, output_dir: Path | str) -> dict:
    output = Path(output_dir)
    tensors_dir = output / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    parameter_count = buffer_count = total_bytes = 0
    manifest_paths = {"parameter": output / "weights_manifest.jsonl", "buffer": output / "buffers_manifest.jsonl"}
    streams = {kind: path.open("w", encoding="utf-8") for kind, path in manifest_paths.items()}
    try:
        entries: list[tuple[str, str, torch.Tensor]] = [("parameter", name, value) for name, value in model.named_parameters()]
        for name, value in model.named_buffers():
            module_path, module = _module_for_name(model, name)
            local_name = name.rsplit(".", 1)[-1]
            if module is not None and local_name in module._non_persistent_buffers_set:
                continue
            entries.append(("buffer", name, value))
        for index, (kind, name, tensor) in enumerate(entries):
            cpu = tensor.detach().cpu().contiguous()
            native, native_dtype, native_suffix = native_bytes(cpu)
            f32 = canonical_f32_bytes(cpu)
            safe = sanitize_filename(name)
            native_path = tensors_dir / f"{index:06d}__{safe}.{native_suffix}.bin"
            f32_path = tensors_dir / f"{index:06d}__{safe}.f32le.bin"
            native_path.write_bytes(native)
            f32_path.write_bytes(f32)
            module_path, module = _module_for_name(model, name)
            stats = compute_tensor_stats(cpu)
            record = {
                "semantic_name": name, "checkpoint_key": name, "module_path": module_path,
                "kind": kind, "shape": list(cpu.shape), "layout": _layout(module, name, cpu),
                "source_dtype": str(tensor.dtype), "native_storage_dtype": native_dtype,
                "native_file": native_path.relative_to(output).as_posix(),
                "canonical_f32_file": f32_path.relative_to(output).as_posix(),
                "sha256_native": sha256_of_bytes(native), "sha256_f32": sha256_of_bytes(f32),
                "numel": cpu.numel(), **{key: stats[key] for key in ("min", "max", "mean", "std")},
            }
            streams[kind].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            total_bytes += len(native) + len(f32)
            if kind == "parameter":
                parameter_count += 1
            else:
                buffer_count += 1
    finally:
        for stream in streams.values():
            stream.close()
    return {"status": "PASS", "parameter_count": parameter_count, "buffer_count": buffer_count,
            "tensor_count": parameter_count + buffer_count, "total_bytes": total_bytes}
