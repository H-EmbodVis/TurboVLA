import numpy as np

from turbovla.evaluation.policy import _build_manual_rgb_normalizer


def test_pixel_values_are_derived_from_same_raw_images():
    raw = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    processor = _build_manual_rgb_normalizer([0, 0, 0], [1, 1, 1], 1 / 255, 4, 2, "test")
    first = processor(raw)["pixel_values"].numpy()
    second = processor(raw.copy())["pixel_values"].numpy()
    np.testing.assert_array_equal(first, second)
