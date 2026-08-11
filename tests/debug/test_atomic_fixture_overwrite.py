import numpy as np

from turbovla.debug import FixtureWriter


def test_atomic_fixture_overwrite_removes_stale_files(tmp_path):
    first = FixtureWriter(tmp_path, "fixture", overwrite=True)
    first.write_image("raw/old", np.zeros((1, 1, 3), dtype=np.uint8))
    first.finalize()
    second = FixtureWriter(tmp_path, "fixture", overwrite=True)
    second.write_instruction("new")
    path = second.finalize()
    assert not (path / "raw/old.npy").exists()
    assert not list(tmp_path.glob("fixture.tmp.*"))
