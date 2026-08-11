from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Callable

import torch

from .tensor_writer import TensorWriter
from .trace_config import TraceConfig
from .trace_utils import build_environment_json, build_trace_tree


class TraceContext:
    """Own one externally-controlled forward trace lifecycle."""

    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self._rank = int(os.environ.get("RANK", "0"))
        self._scope_stack: list[str] = []
        self._writer: TensorWriter | None = None
        self._call_indices: dict[str, int] = {}
        self._forwards_done = 0
        self._trace_dir: Path | None = None

    def begin(self, forward_index: int, fixture_id: str = "") -> bool:
        if self.active:
            raise RuntimeError("a trace session is already active")
        if not self.config.enabled or (self.config.max_forwards > 0 and self._forwards_done >= self.config.max_forwards):
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
        target = self.config.root_dir / fixture_id if fixture_id else self.config.root_dir
        self._trace_dir = target / f"forward_{forward_index:06d}_rank_{self._rank}"
        self._scope_stack.clear()
        self._call_indices.clear()
        self._writer = TensorWriter(self._trace_dir, self.config)
        return True

    def tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        layout: str = "",
        operation: str = "unknown",
        module_path: str | None = None,
        io: str = "intermediate",
        required_level: str = "boundary",
    ) -> None:
        if not self.active:
            return
        semantic_name = ".".join((*self._scope_stack, name)) if self._scope_stack else name
        # Crucially perform level/filter checks before touching accelerator memory.
        if not self.config.includes_level(required_level) or not self.config.matches(semantic_name):
            return
        call_index = self._call_indices.get(semantic_name, 0)
        self._call_indices[semantic_name] = call_index + 1
        assert self._writer is not None
        self._writer.write(
            semantic_name=semantic_name, tensor=tensor, call_index=call_index,
            required_level=required_level, layout=layout, operation=operation,
            module_path=module_path, io=io,
        )

    @contextlib.contextmanager
    def scope(self, name: str):
        self._scope_stack.append(name)
        try:
            yield
        finally:
            self._scope_stack.pop()

    def finish(self, *, error: BaseException | None = None) -> Path | None:
        if not self.active:
            return None
        assert self._writer is not None
        writer = self._writer
        trace_dir = self._trace_dir
        try:
            if error is None:
                writer.write_auxiliary_text("trace_tree.txt", build_trace_tree(writer.records))
                writer.write_auxiliary_text(
                    "environment.json", json.dumps(build_environment_json(), indent=2, ensure_ascii=False)
                )
            writer.close(success=error is None)
        finally:
            self._writer = None
            self._scope_stack.clear()
            self._call_indices.clear()
            if error is None:
                self._forwards_done += 1
        return trace_dir if error is None else None

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def records(self):
        return [] if self._writer is None else self._writer.records

    @property
    def total_bytes(self) -> int:
        return 0 if self._writer is None else self._writer.total_bytes

    @property
    def dump_fn(self) -> Callable[[str, torch.Tensor], None] | None:
        if not self.active:
            return None
        return lambda name, tensor: self.tensor(name, tensor, required_level="boundary")

    def make_boundary_dump(self) -> Callable[[str, torch.Tensor], None] | None:
        return self.dump_fn
