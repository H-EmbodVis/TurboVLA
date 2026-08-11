import numpy as np

from turbovla.debug import FixtureWriter, validate_fixture


def test_fixture_hash_validation_detects_corruption(tmp_path):
    writer = FixtureWriter(tmp_path, "fixture", overwrite=True)
    writer.write_image("raw/view", np.zeros((2, 2, 3), dtype=np.uint8))
    path = writer.finalize()
    assert validate_fixture(path)["status"] == "PASS"
    (path / "raw/view.bin").write_bytes(b"corrupt")
    result = validate_fixture(path)
    assert result["status"] == "FAIL"
    assert any("byte length mismatch" in item or "sha256 mismatch" in item for item in result["errors"])


def test_boolean_reproduction_uses_exact_comparison(tmp_path):
    writer = FixtureWriter(tmp_path, "fixture", overwrite=True)
    import torch
    writer.write_tensor("mask", torch.tensor([True, False]), "u8")
    path = writer.finalize()
    result = validate_fixture(path, reproduce_preprocessing=lambda reader: {"mask.npy": np.array([True, False])})
    assert result["status"] == "PASS"
