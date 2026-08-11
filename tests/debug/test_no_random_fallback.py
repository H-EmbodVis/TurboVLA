from pathlib import Path


def test_replay_script_has_no_random_fallback():
    source = Path("scripts/run_fixture_inference.py").read_text(encoding="utf-8")
    assert "torch.randn" not in source
    assert "strict=False" not in source
    assert "checkpoint\", type=Path, required=True" in source
