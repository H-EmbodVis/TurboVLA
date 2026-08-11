import numpy as np

from turbovla.evaluation.policy import normalized_action_to_env_action


def test_policy_postprocessing_is_exact_on_replay():
    normalized = np.linspace(-1, 1, 7, dtype=np.float32)
    first = normalized_action_to_env_action(normalized)
    second = normalized_action_to_env_action(normalized.copy())
    np.testing.assert_array_equal(first, second)
