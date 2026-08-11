import json
import torch

from turbovla.debug import TraceConfig, TraceContext


def test_call_index_is_per_semantic_name(tmp_path):
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, overwrite=True))
    tracer.begin(0)
    tracer.tensor("same", torch.ones(1))
    tracer.tensor("same", torch.ones(1))
    trace = tracer.finish()
    records = [json.loads(line) for line in (trace / "manifest.jsonl").read_text().splitlines()]
    assert [record["call_index"] for record in records] == [0, 1]
