from __future__ import annotations

from pathlib import Path
from turbovla.debug import TraceConfig


def test_trace_config_filter():
    config = TraceConfig(
        include=("interaction.*", "action.*"),
        exclude=("*.dropout*", "*.norm*"),
    )

    assert config.matches("interaction.layer_00.visual") is True
    assert config.matches("interaction.layer_00.dropout") is False
    assert config.matches("action.decoder.layer_01") is True
    assert config.matches("vision.encoder.patch") is False
