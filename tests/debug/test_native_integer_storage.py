import json
import numpy as np
import torch

from turbovla.debug import TraceConfig, TraceContext


def test_integer_and_mask_storage_are_native(tmp_path):
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, overwrite=True))
    tracer.begin(0)
    tracer.tensor("ids", torch.tensor([3, 4], dtype=torch.int64))
    tracer.tensor("mask", torch.tensor([True, False]))
    trace = tracer.finish()
    records = [json.loads(line) for line in (trace / "manifest.jsonl").read_text().splitlines()]
    assert records[0]["storage_dtype"] == "int64"
    assert records[1]["storage_dtype"] == "uint8"
    np.testing.assert_array_equal(np.fromfile(trace / records[0]["file"], dtype="<i8"), [3, 4])
