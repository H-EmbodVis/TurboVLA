from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..text.bert import BertModelWarper, generate_masks_with_special_tokens
from .configuration import TextEncoderConfig
from ..debug.model_trace import emit, trace_layer_norm, trace_linear, trace_softmax


def _load_pretrained_model(config: TextEncoderConfig):
    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
    }
    if config.attention_implementation:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                attn_implementation=config.attention_implementation,
                **kwargs,
            )
        except Exception as error:
            if config.attention_implementation == "flash_attention_2":
                try:
                    return AutoModel.from_pretrained(
                        config.model_name_or_path,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                except Exception:
                    pass
            print(
                f"[TurboVLA] text attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class TurboVLATextEncoder(nn.Module):
    """Online BERT encoder used by every TurboVLA forward pass."""

    def __init__(self, config: TextEncoderConfig, hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
            use_fast=True,
        )
        bert = _load_pretrained_model(config)
        self.bert = BertModelWarper(bert_model=bert)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.constant_(self.text_projection.bias, 0.0)
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        if config.frozen:
            self.bert.requires_grad_(False)
            if config.force_eval_when_frozen:
                self.bert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.frozen and self.config.force_eval_when_frozen:
            self.bert.eval()
        return self

    def _tokenize_group(
        self,
        instructions: Sequence[str],
        device: torch.device,
        padding_length: int | None,
    ):
        padding = "max_length" if padding_length is not None else "longest"
        effective_max_length = padding_length or self.config.max_length
        tokenized = self.tokenizer(
            [str(item) for item in instructions],
            padding=padding,
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        ).to(device)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens(
            tokenized,
            self.special_tokens,
            self.tokenizer,
        )
        return tokenized, text_self_attention_masks, position_ids

    def _encode_group(
        self,
        instructions: Sequence[str],
        device: torch.device,
        padding_length: int | None,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ):
        tokenized, text_self_attention_masks, position_ids = self._tokenize_group(
            instructions,
            device,
            padding_length,
        )
        if self.config.sub_sentence_present:
            bert_inputs = {key: value for key, value in tokenized.items() if key != "attention_mask"}
            bert_inputs["attention_mask"] = text_self_attention_masks
            bert_inputs["position_ids"] = position_ids
        else:
            bert_inputs = tokenized

        hook = None
        embedding_hook = None
        if dump is not None:
            group_length = tokenized.input_ids.shape[1]
            dump(f"text.group_{group_length}.input_ids", tokenized.input_ids)

            def dump_bert_input(_module, _inputs, output):
                dump(f"text.group_{group_length}.bert_input_embeddings", output)

            hook = self.bert.embeddings.register_forward_hook(dump_bert_input)

        if tracer is not None and tracer.active:
            valid_length = int(tokenized.attention_mask.sum(dim=1).max())
            token_type_ids = tokenized.get("token_type_ids", torch.zeros_like(tokenized.input_ids))
            tracer.tensor("text.tokenizer.input_ids_unpadded", tokenized.input_ids[:, :valid_length], layout="B,L", operation="tokenizer", required_level="op")
            tracer.tensor("text.tokenizer.token_type_ids_unpadded", token_type_ids[:, :valid_length], layout="B,L", operation="tokenizer", required_level="op")
            tracer.tensor("text.tokenizer.attention_mask_unpadded", tokenized.attention_mask[:, :valid_length].bool(), layout="B,L", operation="tokenizer", required_level="op")
            tracer.tensor("text.tokenizer.self_attention_mask_unpadded", text_self_attention_masks[:, :valid_length, :valid_length], layout="B,L,L", operation="tokenizer", required_level="op")
            tracer.tensor("text.tokenizer.position_ids_unpadded", position_ids[:, :valid_length], layout="B,L", operation="tokenizer", required_level="op")
            tracer.tensor("text.layout.input_ids_padded", tokenized.input_ids, layout="B,N", operation="padding", required_level="boundary")
            tracer.tensor("text.layout.attention_mask_padded", tokenized.attention_mask.bool(), layout="B,N", operation="padding", required_level="boundary")
            tracer.tensor("text.layout.key_padding_mask_padded", ~tokenized.attention_mask.bool(), layout="B,N", operation="invert", required_level="op")
            if text_self_attention_masks is not None:
                tracer.tensor("text.layout.self_attention_mask_padded", text_self_attention_masks, layout="B,N,N", operation="padding", required_level="boundary")
            tracer.tensor("text.layout.position_ids_padded", position_ids, layout="B,N", operation="padding", required_level="boundary")
            tracer.tensor("text.bert.embeddings.input_ids", tokenized.input_ids, layout="B,N", operation="input", required_level="op")

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        try:
            with grad_context:
                bert_output = self.bert(
                    **bert_inputs,
                    output_hidden_states=bool(tracer is not None and tracer.active),
                    output_attentions=False,
                )
        finally:
            if hook is not None:
                hook.remove()
            if embedding_hook is not None:
                embedding_hook.remove()

        if tracer is not None and tracer.active:
            self._trace_bert_math(
                tracer, tokenized.input_ids, tokenized.get("token_type_ids", torch.zeros_like(tokenized.input_ids)),
                position_ids, text_self_attention_masks if self.config.sub_sentence_present else tokenized.attention_mask.bool(),
                bert_output,
            )

        return (
            bert_output.last_hidden_state,
            tokenized.attention_mask.bool(),
            text_self_attention_masks,
        )

    def _trace_bert_math(self, tracer, input_ids, token_type_ids, position_ids, attention_mask, output):
        embeddings = self.bert.embeddings
        word = embeddings.word_embeddings(input_ids)
        token_type = embeddings.token_type_embeddings(token_type_ids)
        position = embeddings.position_embeddings(position_ids)
        summed = word + token_type + position
        emit(tracer, "text.bert.embeddings.word", word, "B,N,D", "embedding", "op")
        emit(tracer, "text.bert.embeddings.token_type", token_type, "B,N,D", "embedding", "op")
        emit(tracer, "text.bert.embeddings.position", position, "B,N,D", "embedding", "op")
        emit(tracer, "text.bert.embeddings.sum_before_norm", summed, "B,N,D", "add", "exhaustive")
        trace_layer_norm(tracer, "text.bert.embeddings.norm", summed, embeddings.LayerNorm,
                         layout="B,N,D", include_parameters=True)
        embedding_output = output.hidden_states[0]
        emit(tracer, "text.bert.embeddings.output", embedding_output, "B,N,D", "dropout", "layer")

        if attention_mask.dim() == 2:
            allowed = attention_mask[:, None, None, :].bool()
        else:
            allowed = attention_mask[:, None, :, :].bool()
        for index, layer in enumerate(self.bert.encoder.layer):
            prefix = f"text.bert.layer_{index:02d}"
            hidden = output.hidden_states[index]
            emit(tracer, f"{prefix}.input", hidden, "B,N,D", "layer_input", "layer")
            attention = layer.attention.self
            q = trace_linear(tracer, f"{prefix}.attn.q_linear", hidden, attention.query)
            k = trace_linear(tracer, f"{prefix}.attn.k_linear", hidden, attention.key)
            v = trace_linear(tracer, f"{prefix}.attn.v_linear", hidden, attention.value)
            batch, length = hidden.shape[:2]
            heads, head_dim = attention.num_attention_heads, attention.attention_head_size
            q_reshape = q.view(batch, length, heads, head_dim)
            k_reshape = k.view(batch, length, heads, head_dim)
            v_reshape = v.view(batch, length, heads, head_dim)
            qh, kh, vh = (item.permute(0, 2, 1, 3) for item in (q_reshape, k_reshape, v_reshape))
            for name, reshaped, transposed in (("q", q_reshape, qh), ("k", k_reshape, kh), ("v", v_reshape, vh)):
                emit(tracer, f"{prefix}.attn.{name}.reshape", reshaped, "B,N,H,Dh", "reshape", "exhaustive")
                emit(tracer, f"{prefix}.attn.{name}.transpose", transposed, "B,H,N,Dh", "transpose", "exhaustive")
            kt = kh.transpose(-1, -2)
            logits = torch.matmul(qh, kt)
            scale = torch.tensor(head_dim ** 0.5, dtype=hidden.dtype, device=hidden.device)
            scaled = logits / scale
            expanded_allowed = allowed.expand(batch, heads, length, length)
            mask = ~expanded_allowed
            masked = scaled.masked_fill(mask, torch.finfo(scaled.dtype).min)
            emit(tracer, f"{prefix}.attn.k_transposed", kt, "B,H,Dh,N", "transpose", "exhaustive")
            emit(tracer, f"{prefix}.attn.qk_matmul", logits, "B,H,N,N", "matmul", "op")
            emit(tracer, f"{prefix}.attn.scale", scale, "", "constant", "exhaustive")
            emit(tracer, f"{prefix}.attn.scaled_logits", scaled, "B,H,N,N", "divide", "op")
            emit(tracer, f"{prefix}.attn.mask", mask, "B,H,N,N", "mask", "op")
            emit(tracer, f"{prefix}.attn.masked_logits", masked, "B,H,N,N", "masked_fill", "op")
            probs = trace_softmax(tracer, f"{prefix}.attn.softmax", masked, layout="B,H,N,N", output_dtype=hidden.dtype)
            context_heads = torch.matmul(probs, vh)
            context_transpose = context_heads.permute(0, 2, 1, 3).contiguous()
            context = context_transpose.view(batch, length, -1)
            emit(tracer, f"{prefix}.attn.context_heads", context_heads, "B,H,N,Dh", "matmul", "op")
            emit(tracer, f"{prefix}.attn.context_transpose", context_transpose, "B,N,H,Dh", "transpose", "exhaustive")
            emit(tracer, f"{prefix}.attn.context_merged", context, "B,N,D", "reshape", "op")
            attn_out = trace_linear(tracer, f"{prefix}.attn.output", context, layer.attention.output.dense)
            emit(tracer, f"{prefix}.attn.output", attn_out, "B,N,D", "linear", "layer")
            emit(tracer, f"{prefix}.residual_1.left", hidden, "B,N,D", "identity", "exhaustive")
            emit(tracer, f"{prefix}.residual_1.right", attn_out, "B,N,D", "identity", "exhaustive")
            residual_1 = hidden + attn_out
            emit(tracer, f"{prefix}.residual_1.sum", residual_1, "B,N,D", "add", "op")
            trace_layer_norm(tracer, f"{prefix}.norm_1", residual_1, layer.attention.output.LayerNorm)
            norm_1 = layer.attention.output.LayerNorm(residual_1)
            linear_1 = trace_linear(tracer, f"{prefix}.ffn.linear_1", norm_1, layer.intermediate.dense)
            gelu = layer.intermediate.intermediate_act_fn(linear_1)
            emit(tracer, f"{prefix}.ffn.gelu.input", linear_1, "B,N,F", "gelu", "exhaustive")
            emit(tracer, f"{prefix}.ffn.gelu.output", gelu, "B,N,F", "gelu", "op")
            linear_2 = trace_linear(tracer, f"{prefix}.ffn.linear_2", gelu, layer.output.dense)
            emit(tracer, f"{prefix}.residual_2.left", norm_1, "B,N,D", "identity", "exhaustive")
            emit(tracer, f"{prefix}.residual_2.right", linear_2, "B,N,D", "identity", "exhaustive")
            residual_2 = norm_1 + linear_2
            emit(tracer, f"{prefix}.residual_2.sum", residual_2, "B,N,D", "add", "op")
            trace_layer_norm(tracer, f"{prefix}.norm_2", residual_2, layer.output.LayerNorm)
            emit(tracer, f"{prefix}.output", output.hidden_states[index + 1], "B,N,D", "layer_output", "layer")
        emit(tracer, "text.bert.last_hidden_state_unpadded", output.last_hidden_state, "B,N,D", "bert_output", "boundary")

    def encode_bert_hidden(
        self,
        instructions: Sequence[str],
        device: torch.device,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ):
        if not instructions:
            raise ValueError("instructions cannot be empty")
        normalized = [str(item) for item in instructions]
        output_length = self.config.padding_length
        layout = self.config.padding_length_by_instruction
        if not layout:
            return self._encode_group(normalized, device, output_length, dump=dump, tracer=tracer)

        if output_length is None:
            raise ValueError("text.padding_length is required when instruction-specific lengths are configured")
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index, instruction in enumerate(normalized):
            grouped_indices[int(layout.get(instruction, output_length))].append(index)

        hidden = None
        attention_mask = torch.zeros((len(normalized), output_length), device=device, dtype=torch.bool)
        self_attention = torch.eye(output_length, device=device, dtype=torch.bool).expand(
            len(normalized), -1, -1
        ).clone()
        for group_length, indices in grouped_indices.items():
            if group_length > output_length:
                raise ValueError(f"instruction padding length {group_length} exceeds output length {output_length}")
            group_instructions = [normalized[index] for index in indices]
            group_hidden, group_attention, group_self_attention = self._encode_group(
                group_instructions,
                device,
                group_length,
                dump=dump,
                tracer=tracer,
            )
            if hidden is None:
                hidden = group_hidden.new_zeros((len(normalized), output_length, group_hidden.shape[-1]))
            hidden[indices, :group_length] = group_hidden
            attention_mask[indices, :group_length] = group_attention
            self_attention[indices, :group_length, :group_length] = group_self_attention

        return hidden, attention_mask, self_attention

    def forward(
        self,
        instructions: Sequence[str],
        device: torch.device,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ):
        hidden, text_token_mask, text_self_attention_masks = self.encode_bert_hidden(
            instructions,
            device,
            dump=dump,
            tracer=tracer,
        )
        if dump is not None:
            dump("text.bert_hidden", hidden)
            dump("text.token_mask", text_token_mask)

        hidden = hidden.to(dtype=self.text_projection.weight.dtype)
        emit(tracer, "text.bert.hidden_padded", hidden, "B,N,D", "padding", "boundary")
        emit(tracer, "text.projection.input", hidden, "B,N,D", "input", "op")
        traced_projection = trace_linear(tracer, "text.projection", hidden, self.text_projection)
        emit(tracer, "text.projection.before_zero_fill", traced_projection, "B,N,D", "identity", "exhaustive")
        text_tokens = self.text_projection(hidden)
        text_key_padding_mask = ~text_token_mask
        emit(tracer, "text.projection.zero_fill_mask", text_key_padding_mask, "B,N", "mask", "exhaustive")
        if self.config.zero_padded_tokens:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        if dump is not None:
            dump("text.projected_tokens", text_tokens)
        if tracer is not None and tracer.active:
            tracer.tensor("text.projection.output", text_tokens, layout="B,N,D", operation="linear_projection")
        return text_tokens, text_key_padding_mask, text_self_attention_masks
