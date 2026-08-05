from __future__ import annotations

import os
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    root_dir: Path = field(default_factory=lambda: Path("outputs/parity_traces"))
    level: str = "boundary"
    max_forwards: int = 1
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    save_pt: bool = True
    save_f32: bool = True
    save_raw_bf16: bool = False
    save_stats: bool = True
    save_first_last_values: int = 4
    fail_on_nan: bool = False
    overwrite: bool = False
    max_bytes: int = 0
    max_tensor_numel: int = 0

    @classmethod
    def from_env(cls) -> TraceConfig:
        enabled = os.environ.get("TURBOVLA_TRACE_ENABLED", "0") == "1"
        root_dir = Path(os.environ.get("TURBOVLA_TRACE_ROOT", "outputs/parity_traces"))
        level = os.environ.get("TURBOVLA_TRACE_LEVEL", "boundary")
        max_forwards = int(os.environ.get("TURBOVLA_TRACE_MAX_FORWARDS", "1"))
        
        include_str = os.environ.get("TURBOVLA_TRACE_INCLUDE", "")
        include = tuple(include_str.split(",")) if include_str else ()
        
        exclude_str = os.environ.get("TURBOVLA_TRACE_EXCLUDE", "")
        exclude = tuple(exclude_str.split(",")) if exclude_str else ()
        
        save_pt = os.environ.get("TURBOVLA_TRACE_SAVE_PT", "1") == "1"
        save_f32 = os.environ.get("TURBOVLA_TRACE_SAVE_F32", "1") == "1"
        save_raw_bf16 = os.environ.get("TURBOVLA_TRACE_SAVE_RAW_BF16", "0") == "1"
        fail_on_nan = os.environ.get("TURBOVLA_TRACE_FAIL_ON_NAN", "0") == "1"
        
        return cls(
            enabled=enabled,
            root_dir=root_dir,
            level=level,
            max_forwards=max_forwards,
            include=include,
            exclude=exclude,
            save_pt=save_pt,
            save_f32=save_f32,
            save_raw_bf16=save_raw_bf16,
            fail_on_nan=fail_on_nan,
        )

    def matches(self, semantic_name: str) -> bool:
        if self.exclude:
            for pattern in self.exclude:
                if fnmatch.fnmatch(semantic_name, pattern):
                    return False
        if self.include:
            matched = False
            for pattern in self.include:
                if fnmatch.fnmatch(semantic_name, pattern):
                    matched = True
                    break
            if not matched:
                return False
        return True
