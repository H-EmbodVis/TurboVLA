from turbovla.debug.exhaustive_coverage_spec import CoverageShape, expected_semantic_names


def test_spec_expands_actual_layer_counts():
    names = expected_semantic_names(CoverageShape(bert_layers=2, dino_layers=3, views=2,
                                                  interaction_layers=2, action_decoder_layers=1))
    assert "text.bert.layer_01.attn.softmax.probs" in names
    assert "text.bert.layer_02.output" not in names
    assert "vision.view_1.block_02.attn.softmax.probs" in names
    assert "interaction.layer_01.text_enhancer.output" in names
    assert "action.decoder.layer_00.output" in names
