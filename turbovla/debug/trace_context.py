from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

from .trace_config import TraceConfig
from .tensor_writer import TensorWriter
from .trace_utils import build_trace_tree, build_environment_json


class TraceContext:
    """Manages the lifecycle of a single tensor trace session.

    Usage::

        tracer = TraceContext(config)
        if tracer.begin(forward_index=0, fixture_id="test"):
            with tracer.scope("interaction.layer_02"):
                tracer.tensor("visual.input", visual, layout="B,N,D")
            tracer.finish()
    """

    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self._forward_index = 0
        self._rank = int(os.environ.get("RANK", "0"))
        self._scope_stack: list[str] = []
        self._writer: TensorWriter | None = None
        self._call_indices: dict[str, int] = {}
        self._forwards_done = 0
        self._fixture_id = ""
        self._trace_dir: Path | None = None

    def begin(self, forward_index: int, fixture_id: str = "") -> bool:
        """Begin a new trace session for a given forward index.

        Returns True if tracing is active for this forward.
        """
        if not self.config.enabled:
            return False

        if self.config.max_forwards > 0 and self._forwards_done >= self.config.max_forwards:
            return False

        self._forward_index = forward_index
        self._fixture_id = fixture_id
        self._scope_stack.clear()
        self._call_indices.clear()

        # Determine rank from distributed context if available.
        try:
            if torch.distributed.is_initialized():
                self._rank = torch.distributed.get_rank()
        except Exception:
            pass

        # Build trace directory path.
        trace_dir = self.config.root_dir
        if fixture_id:
            trace_dir = trace_dir / fixture_id
        trace_dir = trace_dir / f"forward_{forward_index:06d}_rank_{self._rank}"

        self._trace_dir = trace_dir
        self._writer = TensorWriter(trace_dir, self.config)
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
    ) -> None:
        """Record a tensor with the given semantic name relative to the current scope."""
        if self._writer is None:
            return

        # Build full semantic name from scope stack.
        if self._scope_stack:
            semantic_name = ".".join(self._scope_stack + [name])
        else:
            semantic_name = name

        self._writer.write(
            semantic_name=semantic_name,
            tensor=tensor,
            layout=layout,
            operation=operation,
            module_path=module_path,
            io=io,
        )

    @contextlib.contextmanager
    def scope(self, name: str):
        """Context manager that pushes a scope segment onto the semantic name stack."""
        self._scope_stack.append(name)
        try:
            yield
        finally:
            self._scope_stack.pop()

    def finish(self) -> Path | None:
        """Finalize the current trace: write tree, environment, close files.

        Returns the trace directory path, or None if no trace was active.
        """
        if self._writer is None:
            return None

        trace_dir = self._writer.trace_dir

        # Write the trace tree summary.
        tree_content = build_trace_tree(self._writer.records)
        (trace_dir / "trace_tree.txt").write_text(tree_content, encoding="utf-8")

        # Write environment info.
        env_info = build_environment_json()
        (trace_dir / "environment.json").write_text(
            json.dumps(env_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._writer.close()
        record_count = self._writer.record_count
        self._writer = None
        self._forwards_done += 1
        self._scope_stack.clear()
        self._call_indices.clear()

        print(
            f"[trace] Saved {record_count} tensors to {trace_dir}",
            flush=True,
        )
        return trace_dir

    @property
    def active(self) -> bool:
        """True if a trace session is currently active."""
        return self._writer is not None

    @property
    def dump_fn(self) -> Callable[[str, torch.Tensor], None] | None:
        """Return a dump callable compatible with the existing model code's
        ``dump`` parameter, or None if tracing is not active.
        """
        if not self.active:
            return None

        def dump(name: str, tensor: torch.Tensor) -> None:
            self.tensor(name, tensor)

        return dump

    def make_boundary_dump(self) -> Callable[[str, torch.Tensor], None] | None:
        """Return a dump callable for boundary-level tracing, or None."""
        if not self.active:
            return None

        def dump(name: str, tensor: torch.Tensor) -> None:
            self.tensor(name, tensor)

        return dump
