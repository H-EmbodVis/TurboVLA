from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from transformers import AutoModel

from .configuration import VisionEncoderConfig


def _load_pretrained_model(config: VisionEncoderConfig):
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
                f"[TurboVLA] vision attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class DINOv3VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _load_pretrained_model(config)
        self.hidden_size = self._hidden_size(self.backbone.config)
        self.patch_size = self._patch_size(self.backbone.config)
        self.prefix_tokens = self._prefix_tokens(self.backbone.config, default=5)
        self.num_patches = (config.image_size // self.patch_size) ** 2

        embeddings = getattr(self.backbone, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)
        if config.frozen:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _hidden_size(config) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("cannot infer DINOv3 hidden size")

    @staticmethod
    def _patch_size(config) -> int:
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("cannot infer DINOv3 patch size")

    @staticmethod
    def _prefix_tokens(config, default: int = 0) -> int:
        if hasattr(config, "num_register_tokens"):
            return int(config.num_register_tokens) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(config.vision_config.num_register_tokens) + 1
        return int(default)

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported DINOv3 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            with autocast_context:
                outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        tokens = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        if tokens.shape[1] == expected_patches + self.prefix_tokens:
            return tokens[:, self.prefix_tokens :, :]
        if tokens.shape[1] == expected_patches:
            return tokens
        raise RuntimeError(
            f"DINOv3 produced {tokens.shape[1]} tokens; expected {expected_patches} patches "
            f"or {expected_patches + self.prefix_tokens} tokens including prefixes"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        dump: Callable[[str, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            tokens = torch.stack(
                [self._encode_images(pixel_values[:, view_idx]) for view_idx in range(num_views)],
                dim=1,
            )
        else:
            flat = pixel_values.flatten(0, 1)
            tokens = self._encode_images(flat)
            tokens = tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2])
        if dump is not None:
            dump("vision.dinov3_patch_tokens", tokens)
        return tokens
