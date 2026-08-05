from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..text.bert import BertModelWarper, generate_masks_with_special_tokens
from .configuration import TextEncoderConfig


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
            tracer.tensor("text.input_ids", tokenized.input_ids, layout="B,N", operation="tokenizer")
            tracer.tensor("text.attention_mask", tokenized.attention_mask, layout="B,N", operation="tokenizer")
            if text_self_attention_masks is not None:
                tracer.tensor("text.text_self_attention_mask", text_self_attention_masks, layout="B,N,N", operation="tokenizer")
            tracer.tensor("text.position_ids", position_ids, layout="B,N", operation="tokenizer")

            def trace_embeddings(_module, _inputs, output):
                tracer.tensor("text.bert.embeddings", output, layout="B,N,D", operation="embedding")

            embedding_hook = self.bert.embeddings.register_forward_hook(trace_embeddings)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        try:
            with grad_context:
                bert_output = self.bert(**bert_inputs)
        finally:
            if hook is not None:
                hook.remove()
            if embedding_hook is not None:
                embedding_hook.remove()

        if tracer is not None and tracer.active:
            tracer.tensor("text.bert.last_hidden_state", bert_output.last_hidden_state, layout="B,N,D", operation="bert_output")

        return (
            bert_output.last_hidden_state,
            tokenized.attention_mask.bool(),
            text_self_attention_masks,
        )

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
        text_tokens = self.text_projection(hidden)
        text_key_padding_mask = ~text_token_mask
        if self.config.zero_padded_tokens:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        if dump is not None:
            dump("text.projected_tokens", text_tokens)
        if tracer is not None and tracer.active:
            tracer.tensor("text.projection.output", text_tokens, layout="B,N,D", operation="linear_projection")
        return text_tokens, text_key_padding_mask, text_self_attention_masks
