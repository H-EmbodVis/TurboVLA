from __future__ import annotations

import csv
import os
import shutil
import time
from pathlib import Path

import torch

from .tensor_record import TensorRecord
from .trace_config import TraceConfig
from .trace_utils import (
    canonical_bytes,
    compute_tensor_stats,
    native_bytes,
    sanitize_filename,
    semantic_name_to_subdir,
    sha256_of_bytes,
)


class TraceByteLimitExceeded(RuntimeError):
    pass


class TensorWriter:
    """Authoritative raw-binary writer.

    The writer stages a whole trace in a sibling temporary directory and only
    publishes it after ``close(success=True)``. Hitting a byte limit is fatal;
    an exhaustive trace is never silently truncated.
    """

    CSV_FIELDS = [
        "trace_id", "semantic_name", "call_index", "module_path", "operation", "io", "stage",
        "required_level", "shape", "layout", "source_dtype", "storage_dtype", "native_storage_dtype",
        "numel", "min", "max", "mean", "std", "abs_max", "l2_norm", "nan_count", "inf_count",
        "file", "native_file", "sha256_f32", "sha256_native",
    ]

    def __init__(self, trace_dir: Path, config: TraceConfig) -> None:
        self.trace_dir = Path(trace_dir)
        self.config = config
        self._trace_counter = 0
        self._total_bytes = 0
        self._records: list[TensorRecord] = []
        self._closed = False
        self._published = False

        if self.trace_dir.exists() and not config.overwrite:
            raise FileExistsError(f"trace directory already exists: {self.trace_dir}")
        self._work_dir = (
            self.trace_dir.with_name(f"{self.trace_dir.name}.tmp.{os.getpid()}")
            if config.atomic else self.trace_dir
        )
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir)
        self._work_dir.mkdir(parents=True)
        self._manifest_file = (self._work_dir / "manifest.jsonl").open("w", encoding="utf-8")
        self._csv_stream = (self._work_dir / "summary.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_stream, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
        self._csv_writer.writeheader()

    def _reserve(self, size: int, semantic_name: str) -> None:
        requested = self._total_bytes + size
        if self.config.max_bytes and requested > self.config.max_bytes:
            raise TraceByteLimitExceeded(
                f"trace byte limit exceeded by {semantic_name!r}: requested={requested}, "
                f"limit={self.config.max_bytes}; trace is incomplete"
            )

    def write(
        self,
        *,
        semantic_name: str,
        tensor: torch.Tensor,
        call_index: int = 0,
        required_level: str = "boundary",
        layout: str,
        operation: str = "unknown",
        module_path: str | None = None,
        io: str = "intermediate",
        stage: str | None = None,
        metadata: dict | None = None,
    ) -> TensorRecord | None:
        if not self.config.matches(semantic_name) or not self.config.includes_level(required_level):
            return None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"trace value {semantic_name!r} is not a tensor")
        if self.config.max_tensor_numel and tensor.numel() > self.config.max_tensor_numel:
            raise TraceByteLimitExceeded(
                f"tensor {semantic_name!r} has {tensor.numel()} elements, exceeding configured limit "
                f"{self.config.max_tensor_numel}; trace is incomplete"
            )

        trace_id = self._trace_counter
        cpu = tensor.detach().to(device="cpu", copy=True).contiguous()
        canonical, storage_dtype, canonical_suffix = canonical_bytes(cpu)
        native, native_storage_dtype, native_suffix = native_bytes(cpu)
        save_native_separately = self.config.save_native and (
            native_storage_dtype != storage_dtype or native != canonical
        )
        bytes_to_write = (len(canonical) if self.config.save_f32 else 0) + (
            len(native) if save_native_separately else 0
        )
        self._reserve(bytes_to_write, semantic_name)

        safe = sanitize_filename(semantic_name)
        base = f"{trace_id:06d}__{safe}__call_{call_index:02d}"
        canonical_dir = self._work_dir / semantic_name_to_subdir(semantic_name)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = canonical_dir / f"{base}.{canonical_suffix}.bin"
        file_rel = ""
        if self.config.save_f32:
            canonical_path.write_bytes(canonical)
            file_rel = canonical_path.relative_to(self._work_dir).as_posix()

        native_rel = None
        if save_native_separately:
            native_dir = self._work_dir / semantic_name_to_subdir(semantic_name, native=True)
            native_dir.mkdir(parents=True, exist_ok=True)
            native_path = native_dir / f"{base}.{native_suffix}.bin"
            native_path.write_bytes(native)
            native_rel = native_path.relative_to(self._work_dir).as_posix()

        stats = compute_tensor_stats(cpu, self.config.save_first_last_values) if self.config.save_stats else compute_tensor_stats(torch.tensor([]))
        stage = stage or semantic_name.split(".")[0]
        record = TensorRecord(
            trace_id=trace_id, semantic_name=semantic_name, call_index=call_index,
            module_path=module_path, operation=operation, io=io, stage=stage,
            required_level=required_level, shape=list(cpu.shape), layout=layout,
            strides=list(cpu.stride()), source_dtype=str(tensor.dtype), storage_dtype=storage_dtype,
            native_storage_dtype=native_storage_dtype, contiguous=cpu.is_contiguous(), numel=cpu.numel(),
            file=file_rel, native_file=native_rel, sha256_f32=sha256_of_bytes(canonical) if cpu.is_floating_point() else None,
            sha256_native=sha256_of_bytes(native), timestamp_ns=time.time_ns(), **stats,
        )
        self._records.append(record)
        self._trace_counter += 1
        self._total_bytes += bytes_to_write
        self._manifest_file.write(record.to_jsonl_line())
        row = record.to_dict()
        row["shape"] = "x".join(map(str, record.shape)) if record.shape else "scalar"
        self._csv_writer.writerow(row)
        self._manifest_file.flush()
        self._csv_stream.flush()
        expected_infinity = semantic_name.endswith(("masked_logits", "softmax.shifted"))
        if self.config.fail_on_nan and (record.nan_count or (record.inf_count and not expected_infinity)):
            raise ValueError(f"NaN detected in tensor {semantic_name!r}: nan={record.nan_count}, inf={record.inf_count}")
        return record

    def write_auxiliary_text(self, relative_path: str, value: str) -> None:
        path = self._work_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def close(self, *, success: bool = True) -> None:
        if self._closed:
            return
        self._manifest_file.close()
        self._csv_stream.close()
        self._closed = True
        if success and self.config.atomic:
            if self.trace_dir.exists():
                old = self.trace_dir.with_name(f"{self.trace_dir.name}.old.{os.getpid()}")
                if old.exists():
                    shutil.rmtree(old)
                self.trace_dir.rename(old)
                self._work_dir.rename(self.trace_dir)
                shutil.rmtree(old)
            else:
                self._work_dir.rename(self.trace_dir)
            self._published = True
        elif success:
            self._published = True
        elif self._work_dir.exists():
            shutil.rmtree(self._work_dir)

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def record_count(self) -> int:
        return self._trace_counter

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def records(self) -> list[TensorRecord]:
        return list(self._records)
