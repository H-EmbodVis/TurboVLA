from __future__ import annotations

import json
from pathlib import Path
import torch
import numpy as np

class FixtureWriter:
    def __init__(self, output_dir: Path | str, fixture_id: str, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.fixture_id = fixture_id
        self.fixture_dir = self.output_dir / fixture_id
        
        if self.fixture_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Fixture directory {self.fixture_dir} already exists")
        else:
            self.fixture_dir.mkdir(parents=True, exist_ok=True)
            
    def write_metadata(self, metadata: dict) -> None:
        with open(self.fixture_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
    def write_instruction(self, instruction: str) -> None:
        with open(self.fixture_dir / "instruction.txt", "w", encoding="utf-8") as f:
            f.write(instruction)
            
    def write_tensor(self, name: str, tensor: torch.Tensor, dtype_suffix: str = "f32", binary_filename: str | None = None) -> None:
        tensor_cpu = tensor.detach().cpu().contiguous()
        np_arr = tensor_cpu.numpy()
        
        np.save(self.fixture_dir / f"{name}.npy", np_arr)
        
        if binary_filename:
            bin_path = self.fixture_dir / binary_filename
        elif dtype_suffix == "f32":
            base_name = name.removesuffix("_f32")
            bin_path = self.fixture_dir / f"{base_name}_f32le.bin"
        elif dtype_suffix == "bf16":
            base_name = name.removesuffix("_bf16")
            bin_path = self.fixture_dir / f"{base_name}_bf16le.bin"
        else:
            bin_path = self.fixture_dir / f"{name}.bin"

        if dtype_suffix == "f32":
            np_arr.astype('<f4', copy=False).tofile(bin_path)
        elif dtype_suffix == "bf16":
            if tensor_cpu.dtype == torch.bfloat16:
                tensor_cpu.view(torch.int16).numpy().tofile(bin_path)
            else:
                raise ValueError(f"Expected bfloat16 tensor for bf16 suffix, got {tensor_cpu.dtype}")
        else:
            np_arr.tofile(bin_path)
            
    def write_image(self, name: str, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        np.save(self.fixture_dir / f"{name}.npy", image)
        image.tofile(self.fixture_dir / f"{name}.bin")
        
    def finalize(self) -> Path:
        return self.fixture_dir
