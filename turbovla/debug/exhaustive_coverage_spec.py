from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoverageShape:
    bert_layers: int = 12
    dino_layers: int = 12
    views: int = 2
    interaction_layers: int = 6
    action_decoder_layers: int = 3


def shape_from_config(config: Any) -> CoverageShape:
    bert = getattr(getattr(config, "text", None), "num_hidden_layers", None)
    # Backbone layer counts are not part of TurboVLAConfig, so callers should
    # pass the loaded model to ``expected_semantic_names`` when available.
    return CoverageShape(
        bert_layers=int(bert or 12),
        dino_layers=12,
        views=int(config.vision.num_views),
        interaction_layers=int(config.interaction.num_layers),
        action_decoder_layers=int(config.action.num_layers),
    )


def _add_prefixed(target: set[str], prefix: str, suffixes: list[str]) -> None:
    target.update(f"{prefix}.{suffix}" for suffix in suffixes)


def expected_semantic_names(config_or_shape: Any = CoverageShape(), model: Any | None = None) -> set[str]:
    shape = config_or_shape if isinstance(config_or_shape, CoverageShape) else shape_from_config(config_or_shape)
    if model is not None:
        shape = CoverageShape(
            bert_layers=len(model.text_encoder.bert.encoder.layer),
            dino_layers=len(model.vision_encoder.backbone.layer), views=model.num_views,
            interaction_layers=len(model.vision_language_interaction.fusion_layers),
            action_decoder_layers=len(model.action_head.decoder.decoder.layers),
        )
    names: set[str] = {
        "input.raw.view_0_rgb_u8", "input.raw.view_1_rgb_u8", "input.rotated.view_0_rgb_u8",
        "input.rotated.view_1_rgb_u8", "input.rescale.factor", "input.rescaled.view_0",
        "input.rescaled.view_1", "input.normalize.mean", "input.normalize.std",
        "input.normalized.view_0", "input.normalized.view_1", "input.pixel_values_f32",
        "input.pixel_values_model_dtype", "state.raw", "state.normalize.mean", "state.normalize.std",
        "state.centered", "state.normalized_f32", "state.normalized_model_dtype",
        "text.instruction_utf8_bytes", "text.tokenizer.input_ids_unpadded",
        "text.tokenizer.token_type_ids_unpadded", "text.tokenizer.attention_mask_unpadded",
        "text.tokenizer.self_attention_mask_unpadded", "text.tokenizer.position_ids_unpadded",
        "text.layout.input_ids_padded", "text.layout.attention_mask_padded",
        "text.layout.key_padding_mask_padded", "text.layout.self_attention_mask_padded",
        "text.layout.position_ids_padded", "text.bert.embeddings.input_ids", "text.bert.embeddings.word",
        "text.bert.embeddings.token_type", "text.bert.embeddings.position",
        "text.bert.embeddings.sum_before_norm", "text.bert.embeddings.norm.input", "text.bert.embeddings.norm.mean",
        "text.bert.embeddings.norm.variance", "text.bert.embeddings.norm.inv_std",
        "text.bert.embeddings.norm.normalized", "text.bert.embeddings.norm.weight",
        "text.bert.embeddings.norm.bias", "text.bert.embeddings.norm.output",
        "text.bert.embeddings.output", "text.bert.last_hidden_state_unpadded", "text.bert.hidden_padded",
        "text.projection.input", "text.projection.matmul", "text.projection.bias_add",
        "text.projection.before_zero_fill", "text.projection.zero_fill_mask", "text.projection.output",
        "vision.patch_tokens_stacked", "vision_projection.input", "vision_projection.skip.matmul",
        "vision_projection.skip.output", "vision_projection.mlp.linear_1.matmul",
        "vision_projection.mlp.linear_1.bias_add", "vision_projection.mlp.gelu.input",
        "vision_projection.mlp.gelu.output", "vision_projection.mlp.linear_2.matmul",
        "vision_projection.mlp.linear_2.bias_add", "vision_projection.residual.left",
        "vision_projection.residual.right", "vision_projection.residual.sum",
        "vision_projection.view_embedding", "vision_projection.position.before_add",
        "vision_projection.position.after_add", "vision_projection.before_flatten",
        "vision_projection.flattened", "condition.visual_tokens", "condition.text_tokens",
        "condition.concat_axis", "condition.concatenated", "state_projection.input_raw",
        "state_projection.input_normalized", "state_projection.linear_1.matmul",
        "state_projection.linear_1.bias_add", "state_projection.gelu.input", "state_projection.gelu.output",
        "state_projection.linear_2.matmul", "state_projection.linear_2.bias_add",
        "state_projection.before_reshape", "state_projection.after_reshape",
        "state_projection.position_embedding", "state_projection.position.before_add",
        "state_projection.position.after_add", "state_projection.output", "action.memory.condition",
        "action.memory.state_tokens", "action.memory.concat_axis", "action.memory.concatenated",
        "action.queries.weight", "action.queries.expanded", "action.mlp.input",
        "action.before_tanh", "action.normalized", "action.denormalize.action_min",
        "action.denormalize.action_max", "action.denormalize.input", "action.denormalize.plus_one",
        "action.denormalize.range", "action.denormalize.scaled", "action.denormalize.arm_output",
        "action.gripper.source", "action.gripper.deadband", "action.gripper.positive_mask",
        "action.gripper.negative_mask", "action.gripper.output", "action.denormalized",
        "action.first_step", "action.executed_steps",
    }
    for norm in ("vision_projection.input_norm", "vision_projection.output_norm", "state_projection.norm",
                 "state_projection.output_norm"):
        _add_prefixed(names, norm, ["input", "mean", "variance", "inv_std", "normalized", "output"])

    bert_suffixes = [
        "input", "attn.q_linear.matmul", "attn.q_linear.bias_add", "attn.k_linear.matmul",
        "attn.k_linear.bias_add", "attn.v_linear.matmul", "attn.v_linear.bias_add",
        "attn.q.reshape", "attn.q.transpose", "attn.k.reshape", "attn.k.transpose",
        "attn.v.reshape", "attn.v.transpose", "attn.k_transposed", "attn.qk_matmul", "attn.scale",
        "attn.scaled_logits", "attn.mask", "attn.masked_logits", "attn.softmax.max",
        "attn.softmax.shifted", "attn.softmax.exp", "attn.softmax.sum", "attn.softmax.probs",
        "attn.context_heads", "attn.context_transpose", "attn.context_merged", "attn.output.matmul",
        "attn.output.bias_add", "attn.output", "residual_1.left", "residual_1.right", "residual_1.sum",
        "norm_1.input", "norm_1.mean", "norm_1.variance", "norm_1.inv_std", "norm_1.normalized",
        "norm_1.output", "ffn.linear_1.matmul", "ffn.linear_1.bias_add", "ffn.gelu.input",
        "ffn.gelu.output", "ffn.linear_2.matmul", "ffn.linear_2.bias_add", "residual_2.left",
        "residual_2.right", "residual_2.sum", "norm_2.input", "norm_2.mean", "norm_2.variance",
        "norm_2.inv_std", "norm_2.normalized", "norm_2.output", "output",
    ]
    for index in range(shape.bert_layers):
        _add_prefixed(names, f"text.bert.layer_{index:02d}", bert_suffixes)

    vision_embedding = [
        "input", "patch_embed.input", "patch_embed.conv.weight", "patch_embed.conv.bias",
        "patch_embed.conv.output", "patch_embed.flatten", "patch_embed.transpose", "cls_token",
        "register_tokens", "position_embedding", "tokens_before_position", "tokens_after_position",
        "tokens_with_prefix", "tokens_with_prefix_final", "prefix_tokens", "patch_tokens",
    ]
    vision_block = [
        "input", "norm_1.input", "norm_1.mean", "norm_1.variance", "norm_1.inv_std",
        "norm_1.normalized", "norm_1.output", "attn.qkv_linear.matmul", "attn.qkv_linear.bias_add",
        "attn.q", "attn.k", "attn.v", "attn.q_heads", "attn.k_heads", "attn.v_heads", "attn.logits",
        "attn.scale", "attn.scaled_logits", "attn.softmax.max", "attn.softmax.shifted",
        "attn.softmax.exp", "attn.softmax.sum", "attn.softmax.probs", "attn.context_heads",
        "attn.context_merged", "attn.output_projection", "layer_scale_1.input", "layer_scale_1.output",
        "residual_1.left", "residual_1.right", "residual_1.sum", "norm_2.input", "norm_2.mean",
        "norm_2.variance", "norm_2.inv_std", "norm_2.normalized", "norm_2.output",
        "mlp.linear_1.matmul", "mlp.linear_1.bias_add", "mlp.activation.input", "mlp.activation.output",
        "mlp.linear_2.matmul", "mlp.linear_2.bias_add", "layer_scale_2.input", "layer_scale_2.output",
        "residual_2.left", "residual_2.right", "residual_2.sum", "output",
    ]
    for view in range(shape.views):
        _add_prefixed(names, f"vision.view_{view}", vision_embedding)
        for index in range(shape.dino_layers):
            _add_prefixed(names, f"vision.view_{view}.block_{index:02d}", vision_block)

    fusion = [
        "visual.input", "text.input", "visual.norm.input", "visual.norm.mean", "visual.norm.variance",
        "visual.norm.inv_std", "visual.norm.normalized", "visual.norm.output", "text.norm.input",
        "text.norm.mean", "text.norm.variance", "text.norm.inv_std", "text.norm.normalized", "text.norm.output",
    ]
    for projection in ("q_visual", "k_text", "v_visual", "v_text"):
        fusion += [f"cross.{projection}.matmul", f"cross.{projection}.bias_add",
                   f"cross.{projection}.reshape", f"cross.{projection}.transpose"]
    for direction in ("visual_to_text", "text_to_visual"):
        fusion += [f"cross.{direction}.{name}" for name in (
            (["qk_matmul", "logits_raw", "global_max", "logits_global_shifted", "logits_clamped"] if direction == "visual_to_text"
             else ["logits_transposed", "row_max", "logits_shifted", "logits_clamped"])
            + ["mask", "masked_logits", "softmax.max", "softmax.shifted", "softmax.exp", "softmax.sum",
               "softmax.probs", "context_heads", "context_transpose", "context_merged", "output.matmul",
               "output.bias_add", "output"])]
    for side in ("visual", "text"):
        fusion += [f"{side}.{name}" for name in ("residual_source", "gamma", "delta", "scaled_delta",
                                                  "drop_path_output", "residual_sum", "after_fusion")]
    enhancer = ["input", "mask.original", "mask.repeated"]
    enhancer += [f"attn.{name}" for name in (
        "q_linear k_linear v_linear q_heads k_heads v_heads logits masked_logits softmax.max softmax.shifted "
        "softmax.exp softmax.sum softmax.probs context_heads context_merged output_projection".split())]
    for residual in (1, 2):
        enhancer += [f"residual_{residual}.{part}" for part in ("left", "right", "sum")]
        enhancer += [f"norm_{residual}.{part}" for part in ("input", "mean", "variance", "inv_std", "normalized", "output")]
    enhancer += ["ffn.linear_1.matmul", "ffn.linear_1.bias_add", "ffn.activation.input", "ffn.activation.output",
                 "ffn.linear_2.matmul", "ffn.linear_2.bias_add", "output"]
    for index in range(shape.interaction_layers):
        base = f"interaction.layer_{index:02d}"
        _add_prefixed(names, base, fusion)
        _add_prefixed(names, f"{base}.text_enhancer", enhancer)

    decoder = ["input"]
    for attention in ("self_attn", "cross_attn"):
        decoder += [f"{attention}.norm_input"]
        decoder += [f"{attention}.norm.{part}" for part in ("input", "mean", "variance", "inv_std", "normalized", "output")]
        decoder += [f"{attention}.{part}" for part in (
            "q_linear k_linear v_linear q_heads k_heads v_heads logits mask masked_logits softmax.max softmax.shifted "
            "softmax.exp softmax.sum softmax.probs context_heads context_merged output_projection".split())]
        decoder += [f"{attention}.residual.{part}" for part in ("left", "right", "sum")]
    decoder += ["ffn.norm_input"] + [f"ffn.norm.{part}" for part in ("input", "mean", "variance", "inv_std", "normalized", "output")]
    decoder += ["ffn.linear_1.matmul", "ffn.linear_1.bias_add", "ffn.activation.input", "ffn.activation.output",
                "ffn.linear_2.matmul", "ffn.linear_2.bias_add"]
    decoder += [f"ffn.residual.{part}" for part in ("left", "right", "sum")] + ["output"]
    for index in range(shape.action_decoder_layers):
        _add_prefixed(names, f"action.decoder.layer_{index:02d}", decoder)
    for index in range(3):
        names.update({f"action.mlp.linear_{index}.matmul", f"action.mlp.linear_{index}.bias_add"})
        if index < 2:
            names.update({f"action.mlp.relu_{index}.input", f"action.mlp.relu_{index}.output"})
    return names
