from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class TensorRecord:
    trace_id: int
    semantic_name: str
    call_index: int
    module_path: str | None = None
    operation: str = "unknown"
    io: str = "intermediate"
    stage: str = ""
    required_level: str = "boundary"
    shape: list[int] = field(default_factory=list)
    layout: str = ""
    strides: list[int] = field(default_factory=list)
    source_dtype: str = ""
    storage_dtype: str = ""
    native_storage_dtype: str = ""
    endianness: str = "little"
    contiguous: bool = True
    numel: int = 0
    file: str = ""
    native_file: str | None = None
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
    first_values: list[float | int | bool | None] = field(default_factory=list)
    last_values: list[float | int | bool | None] = field(default_factory=list)
    sha256_f32: str | None = None
    sha256_native: str = ""
    timestamp_ns: int = 0

    # Backwards-compatible aliases for old trace readers.
    @property
    def raw_bf16_file(self) -> str | None:
        return self.native_file if self.native_storage_dtype == "bfloat16" else None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["raw_bf16_file"] = self.raw_bf16_file
        payload["pt_file"] = None
        return payload

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
