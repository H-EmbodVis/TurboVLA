from __future__ import annotations

import time
from pathlib import Path

import torch

from .trace_config import TraceConfig
from .tensor_record import TensorRecord
from .trace_utils import (
    compute_tensor_stats,
    sha256_of_bytes,
    semantic_name_to_subdir,
    canonical_f32_bytes,
    raw_bf16_bytes,
)


class TensorWriter:
    """Writes tensors to disk in canonical format with manifest and summary CSV."""

    def __init__(self, trace_dir: Path, config: TraceConfig) -> None:
        self.trace_dir = trace_dir
        self.config = config
        self._trace_counter = 0
        self._total_bytes = 0

        if trace_dir.exists() and not config.overwrite:
            raise FileExistsError(
                f"Trace directory {trace_dir} already exists. "
                f"Use --trace-overwrite to allow overwriting."
            )
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self._manifest_file = open(
            self.trace_dir / "manifest.jsonl", "w", encoding="utf-8"
        )
        self._csv_file = open(
            self.trace_dir / "summary.csv", "w", encoding="utf-8"
        )
        self._csv_file.write(
            "trace_id,semantic_name,shape,layout,source_dtype,storage_dtype,"
            "min,max,mean,std,abs_max,l2_norm,nan_count,inf_count,file\n"
        )
        self._records: list[TensorRecord] = []

    def write(
        self,
        *,
        semantic_name: str,
        tensor: torch.Tensor,
        layout: str,
        operation: str = "unknown",
        module_path: str | None = None,
        io: str = "intermediate",
        stage: str | None = None,
        metadata: dict | None = None,
    ) -> TensorRecord | None:
        """Write a tensor to disk and record its metadata.

        Returns the TensorRecord, or None if the tensor was filtered out.
        """
        if not self.config.matches(semantic_name):
            return None

        numel = tensor.numel()
        if self.config.max_tensor_numel > 0 and numel > self.config.max_tensor_numel:
            return None

        if self.config.max_bytes > 0 and self._total_bytes > self.config.max_bytes:
            return None

        trace_id = self._trace_counter
        self._trace_counter += 1

        # Record original dtype before any conversion.
        source_dtype = str(tensor.dtype)

        # Snapshot: detach, CPU, contiguous.
        # copy=True ensures we capture the current state even if the tensor
        # is mutated in-place later.
        t_cpu = tensor.detach().to(device="cpu", copy=True).contiguous()

        # Stats.
        if self.config.save_stats:
            stats = compute_tensor_stats(
                t_cpu, num_first_last=self.config.save_first_last_values
            )
        else:
            stats = compute_tensor_stats(torch.tensor([]))

        # Determine stage from semantic name if not provided.
        if not stage:
            stage = semantic_name.split(".")[0]

        # Build output subdirectory.
        subdir = semantic_name_to_subdir(semantic_name)
        save_dir = self.trace_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        filename_base = f"{trace_id:04d}__{semantic_name.replace('.', '__')}"

        # --- Canonical float32 binary ---
        f32_bytes = canonical_f32_bytes(t_cpu)
        rel_f32_path = ""
        if self.config.save_f32:
            f32_path = save_dir / f"{filename_base}.f32le.bin"
            f32_path.write_bytes(f32_bytes)
            rel_f32_path = f32_path.relative_to(self.trace_dir).as_posix()
            self._total_bytes += len(f32_bytes)

        sha256_hash = sha256_of_bytes(f32_bytes)

        # --- PyTorch .pt ---
        pt_path_str = None
        if self.config.save_pt:
            pt_path = save_dir / f"{filename_base}.pt"
            torch.save(t_cpu, pt_path)
            pt_path_str = pt_path.relative_to(self.trace_dir).as_posix()

        # --- Raw BF16 ---
        raw_bf16_str = None
        if self.config.save_raw_bf16 and tensor.dtype == torch.bfloat16:
            bf16_data = raw_bf16_bytes(t_cpu.to(torch.bfloat16))
            if bf16_data:
                bf16_path = save_dir / f"{filename_base}.bf16le.bin"
                bf16_path.write_bytes(bf16_data)
                raw_bf16_str = bf16_path.relative_to(self.trace_dir).as_posix()
                self._total_bytes += len(bf16_data)

        # --- Build record ---
        record = TensorRecord(
            trace_id=trace_id,
            semantic_name=semantic_name,
            module_path=module_path,
            operation=operation,
            call_index=0,
            io=io,
            stage=stage,
            shape=list(t_cpu.shape),
            layout=layout,
            strides=list(t_cpu.stride()),
            source_dtype=source_dtype,
            storage_dtype="float32",
            endianness="little",
            contiguous=t_cpu.is_contiguous(),
            numel=numel,
            file=rel_f32_path,
            pt_file=pt_path_str,
            raw_bf16_file=raw_bf16_str,
            min=stats["min"],
            max=stats["max"],
            mean=stats["mean"],
            std=stats["std"],
            abs_max=stats["abs_max"],
            l1_norm=stats["l1_norm"],
            l2_norm=stats["l2_norm"],
            zero_count=stats["zero_count"],
            zero_ratio=stats["zero_ratio"],
            nan_count=stats["nan_count"],
            inf_count=stats["inf_count"],
            first_values=stats["first_values"],
            last_values=stats["last_values"],
            sha256_f32=sha256_hash,
            timestamp_ns=time.time_ns(),
        )

        self._records.append(record)
        self._manifest_file.write(record.to_jsonl_line())
        self._manifest_file.flush()

        self._csv_file.write(record.to_csv_row())
        self._csv_file.flush()

        print(
            f"[trace] {trace_id:04d} {semantic_name} "
            f"shape={record.shape} dtype={source_dtype} "
            f"min={record.min:.4f} max={record.max:.4f} mean={record.mean:.4f}",
            flush=True,
        )

        if self.config.fail_on_nan and record.nan_count > 0:
            raise ValueError(
                f"NaN detected in tensor '{semantic_name}': {record.nan_count} NaN values"
            )

        return record

    def close(self) -> None:
        """Flush and close manifest and CSV files."""
        if not self._manifest_file.closed:
            self._manifest_file.close()
        if not self._csv_file.closed:
            self._csv_file.close()

    @property
    def record_count(self) -> int:
        return self._trace_counter

    @property
    def records(self) -> list[TensorRecord]:
        return list(self._records)
