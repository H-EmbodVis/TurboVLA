#!/usr/bin/env python3
"""Export and validate the first static TurboVLA QNN graph set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

PORT_ROOT = Path(__file__).resolve().parents[1]
if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))

from tools.qcs8550_reference import (
    OneViewDINOv3,
    StaticBert,
    StaticPolicyCore,
    load_object_reference,
    pad_bert_hidden,
    prepare_static_text_inputs,
)


INSTRUCTIONS = {
    11: "put the bowl on the plate",
    14: "pick up the orange juice and place it in the basket",
    21: "put the white mug on the left plate and put the yellow and white mug on the right plate",
}


def numpy_value(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy()


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    denominator = np.linalg.norm(np.asarray(expected, dtype=np.float64))
    return float(numerator / max(denominator, 1e-12))


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    return {
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-12)),
        "max_abs": float(difference.abs().max()),
        "rmse": float(difference.square().mean().sqrt()),
    }


def export_model(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: Path,
    input_names: list[str],
    output_name: str,
    opset: int,
) -> None:
    module.eval()
    torch.onnx.export(
        module,
        inputs,
        output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=[output_name],
        dynamic_axes=None,
    )
    onnx.checker.check_model(str(output))


def run_ort(path: Path, feed: dict[str, np.ndarray]) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feed)
    if len(outputs) != 1:
        raise RuntimeError(f"expected one output from {path.name}, got {len(outputs)}")
    return np.asarray(outputs[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PORT_ROOT / "artifacts" / "object")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    sdpa_model = load_object_reference(device="cpu")
    model = load_object_reference(
        device="cpu",
        dino_attention_implementation="eager",
        bert_attention_implementation="eager",
    )
    # TransformerDecoder's eval fastpath becomes aten::_native_multi_head_attention,
    # which is not an ONNX operator. The portable path is action-identical here.
    torch.backends.mha.set_fastpath_enabled(False)
    dino = OneViewDINOv3(model).eval()
    bert = StaticBert(model).eval()
    core = StaticPolicyCore(model).eval()

    pixels = torch.randn((1, 2, 3, 256, 256), dtype=torch.float32)
    state = torch.randn((1, 8), dtype=torch.float32)
    core_instruction = INSTRUCTIONS[11]

    with torch.inference_mode():
        sdpa_action = sdpa_model([core_instruction], {"dinov3": pixels}, state)
        full_action = model([core_instruction], {"dinov3": pixels}, state)
        vision_view0 = dino(pixels[:, 0])
        vision_view1 = dino(pixels[:, 1])
        core_text = prepare_static_text_inputs(model, core_instruction, torch.device("cpu"))
        core_bert_hidden = bert(
            core_text["input_ids"],
            core_text["token_type_ids"],
            core_text["bert_attention_mask"],
            core_text["position_ids"],
        )
        core_bert_padded = pad_bert_hidden(core_bert_hidden)
        split_action = core(
            vision_view0,
            vision_view1,
            core_bert_padded,
            core_text["text_key_padding_mask"],
            core_text["text_self_attention_mask"],
            state,
        )

    if not torch.equal(full_action, split_action):
        max_abs = float((full_action - split_action).abs().max())
        raise RuntimeError(f"split reference changed full action, max_abs={max_abs}")
    eager_vs_sdpa = error_metrics(full_action, sdpa_action)
    if eager_vs_sdpa["relative_l2"] > 1e-5:
        raise RuntimeError(f"eager lowering changed reference action too much: {eager_vs_sdpa}")

    dino_path = onnx_dir / "dinov3_one_view_fp32.onnx"
    export_model(dino, (pixels[:, 0],), dino_path, ["pixel_values"], "patch_tokens", args.opset)

    bert_paths: dict[int, Path] = {}
    bert_reference: dict[int, dict[str, torch.Tensor]] = {}
    for length, instruction in INSTRUCTIONS.items():
        text = prepare_static_text_inputs(model, instruction, torch.device("cpu"))
        bert_inputs = (
            text["input_ids"],
            text["token_type_ids"],
            text["bert_attention_mask"],
            text["position_ids"],
        )
        with torch.inference_mode():
            hidden = bert(*bert_inputs)
        path = onnx_dir / f"bert_l{length}_fp32.onnx"
        export_model(
            bert,
            bert_inputs,
            path,
            ["input_ids", "token_type_ids", "text_self_attention_mask", "position_ids"],
            "bert_hidden",
            args.opset,
        )
        bert_paths[length] = path
        bert_reference[length] = {**text, "bert_hidden": hidden}

    core_path = onnx_dir / "policy_core_l21_fp32.onnx"
    core_inputs = (
        vision_view0,
        vision_view1,
        core_bert_padded,
        core_text["text_key_padding_mask"],
        core_text["text_self_attention_mask"],
        state,
    )
    core_input_names = [
        "vision_view0",
        "vision_view1",
        "bert_hidden_padded",
        "text_key_padding_mask",
        "text_self_attention_mask",
        "state",
    ]
    export_model(core, core_inputs, core_path, core_input_names, "normalized_action", args.opset)

    precision: dict[str, Any] = {}
    dino_input = {"pixel_values": numpy_value(pixels[:, 0])}
    precision["dinov3_one_view"] = {
        "relative_l2": relative_l2(run_ort(dino_path, dino_input), numpy_value(vision_view0)),
        "shape": list(vision_view0.shape),
    }
    for length, text in bert_reference.items():
        feed = {
            "input_ids": numpy_value(text["input_ids"]),
            "token_type_ids": numpy_value(text["token_type_ids"]),
            "text_self_attention_mask": numpy_value(text["bert_attention_mask"]),
            "position_ids": numpy_value(text["position_ids"]),
        }
        precision[f"bert_l{length}"] = {
            "relative_l2": relative_l2(run_ort(bert_paths[length], feed), numpy_value(text["bert_hidden"])),
            "shape": list(text["bert_hidden"].shape),
        }
    core_feed = dict(zip(core_input_names, (numpy_value(value) for value in core_inputs)))
    ort_action = run_ort(core_path, core_feed)
    precision["policy_core_l21"] = {
        "relative_l2": relative_l2(ort_action, numpy_value(full_action)),
        "shape": list(full_action.shape),
    }

    np.savez(
        output_dir / "reference_bundle_l11.npz",
        pixels=numpy_value(pixels),
        state=numpy_value(state),
        full_action=numpy_value(full_action),
        vision_view0=numpy_value(vision_view0),
        vision_view1=numpy_value(vision_view1),
        input_ids=numpy_value(core_text["input_ids"]),
        token_type_ids=numpy_value(core_text["token_type_ids"]),
        bert_attention_mask=numpy_value(core_text["bert_attention_mask"]),
        position_ids=numpy_value(core_text["position_ids"]),
        bert_hidden=numpy_value(core_bert_hidden),
        bert_hidden_padded=numpy_value(core_bert_padded),
        text_key_padding_mask=numpy_value(core_text["text_key_padding_mask"]),
        text_self_attention_mask=numpy_value(core_text["text_self_attention_mask"]),
    )

    report = {
        "source": {
            "checkpoint": str(model.config.name),
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "dino_feature": "outputs.hidden_states[-1]",
            "static_attention_lowering": "eager",
            "mha_fastpath_enabled": False,
        },
        "input_contract": {
            "pixels": [1, 2, 3, 256, 256],
            "state": [1, 8],
            "action": [1, 12, 7],
            "bert_lengths": sorted(INSTRUCTIONS),
        },
        "instruction_by_length": {str(key): value for key, value in INSTRUCTIONS.items()},
        "onnx": {
            "dinov3": str(dino_path),
            "bert": {str(key): str(value) for key, value in bert_paths.items()},
            "policy_core": str(core_path),
        },
        "precision": precision,
        "eager_vs_sdpa_action": eager_vs_sdpa,
        "reference_bundle": str(output_dir / "reference_bundle_l11.npz"),
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    main()
