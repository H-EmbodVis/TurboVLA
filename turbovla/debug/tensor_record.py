from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

@dataclass
class TensorRecord:
    trace_id: int
    semantic_name: str
    module_path: str | None = None
    operation: str = "unknown"
    call_index: int = 0
    io: str = "intermediate"
    stage: str = ""
    shape: list[int] = field(default_factory=list)
    layout: str = ""
    strides: list[int] = field(default_factory=list)
    source_dtype: str = ""
    storage_dtype: str = "float32"
    endianness: str = "little"
    contiguous: bool = True
    numel: int = 0
    file: str = ""
    pt_file: str | None = None
    raw_bf16_file: str | None = None
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    abs_max: float = 0.0
    l1_norm: float = 0.0
    l2_norm: float = 0.0
    zero_count: int = 0
    zero_ratio: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    first_values: list[float] = field(default_factory=list)
    last_values: list[float] = field(default_factory=list)
    sha256_f32: str = ""
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict()) + "\n"

    def to_csv_row(self) -> str:
        shape_str = "x".join(map(str, self.shape)) if self.shape else "scalar"
        fields = [
            str(self.trace_id),
            self.semantic_name,
            shape_str,
            self.layout,
            self.source_dtype,
            self.storage_dtype,
            f"{self.min:.6g}",
            f"{self.max:.6g}",
            f"{self.mean:.6g}",
            f"{self.std:.6g}",
            f"{self.abs_max:.6g}",
            f"{self.l2_norm:.6g}",
            str(self.nan_count),
            str(self.inf_count),
            self.file
        ]
        return ",".join(fields) + "\n"
