from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path


TRACE_LEVEL_ORDER = {"boundary": 0, "layer": 1, "op": 2, "exhaustive": 3}


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    root_dir: Path = field(default_factory=lambda: Path("outputs/parity_traces"))
    level: str = "boundary"
    max_forwards: int = 1
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    # Kept only for source compatibility. Authoritative traces never save .pt.
    save_pt: bool = False
    save_f32: bool = True
    save_raw_bf16: bool = True
    save_native: bool = True
    save_stats: bool = True
    save_first_last_values: int = 4
    fail_on_nan: bool = False
    overwrite: bool = False
    max_bytes: int = 0
    max_tensor_numel: int = 0
    atomic: bool = True

    def __post_init__(self) -> None:
        if self.level not in TRACE_LEVEL_ORDER:
            raise ValueError(f"invalid trace level {self.level!r}; expected {tuple(TRACE_LEVEL_ORDER)}")
        if self.max_bytes < 0 or self.max_tensor_numel < 0:
            raise ValueError("trace byte and tensor limits cannot be negative")

    @classmethod
    def from_env(cls) -> "TraceConfig":
        def flag(name: str, default: str) -> bool:
            return os.environ.get(name, default).lower() in {"1", "true", "yes"}

        return cls(
            enabled=flag("TURBOVLA_TRACE_ENABLED", "0"),
            root_dir=Path(os.environ.get("TURBOVLA_TRACE_ROOT", "outputs/parity_traces")),
            level=os.environ.get("TURBOVLA_TRACE_LEVEL", "boundary"),
            max_forwards=int(os.environ.get("TURBOVLA_TRACE_MAX_FORWARDS", "1")),
            include=tuple(filter(None, os.environ.get("TURBOVLA_TRACE_INCLUDE", "").split(","))),
            exclude=tuple(filter(None, os.environ.get("TURBOVLA_TRACE_EXCLUDE", "").split(","))),
            save_pt=False,
            save_f32=flag("TURBOVLA_TRACE_SAVE_F32", "1"),
            save_native=flag("TURBOVLA_TRACE_SAVE_NATIVE", "1"),
            save_raw_bf16=flag("TURBOVLA_TRACE_SAVE_NATIVE", "1"),
            fail_on_nan=flag("TURBOVLA_TRACE_FAIL_ON_NAN", "0"),
            overwrite=flag("TURBOVLA_TRACE_OVERWRITE", "0"),
            max_bytes=int(os.environ.get("TURBOVLA_TRACE_MAX_BYTES", "0")),
        )

    def matches(self, semantic_name: str) -> bool:
        if any(fnmatch.fnmatch(semantic_name, pattern) for pattern in self.exclude):
            return False
        return not self.include or any(fnmatch.fnmatch(semantic_name, pattern) for pattern in self.include)

    def includes_level(self, required_level: str) -> bool:
        if required_level not in TRACE_LEVEL_ORDER:
            raise ValueError(f"invalid required trace level {required_level!r}")
        return TRACE_LEVEL_ORDER[self.level] >= TRACE_LEVEL_ORDER[required_level]
