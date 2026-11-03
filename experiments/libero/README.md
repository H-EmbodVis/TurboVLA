# LIBERO

This directory contains the released LIBERO recipe and public entry points for the root `turbovla` package. The configuration uses two DINOv3 views, an online BERT encoder, 7-D actions, and a chunk size of 12. The task and episode rollout protocol is the VLA-Adapter-derived implementation under `third_party/vla_adapter/`.

## Layout

```text
../../turbovla/             model, RLDS data, training, and policy code
configs/                    released mixed-suite normalization statistics
train.py                    public mixed-suite training entry point
evaluate.py                 public single-checkpoint evaluation entry point
../../scripts/libero/       data, statistics, and evaluation utilities
../../third_party/          VLA-Adapter rollout protocol and license
```

## Environment

Use the following Linux reference environment for the published checkpoint. The Python package pins are scoped to the `libero` extra in `pyproject.toml`; use a separate environment for RoboTwin.

| Component | Reference version |
| --- | --- |
| OS / Python | Ubuntu 20.04 / Python 3.9.25 |
| PyTorch / torchvision | 2.3.1+cu121 / 0.18.1 (CUDA 12.1 wheels) |
| transformers / tokenizers / timm | 4.56.0 / 0.22.2 / 1.0.25 |
| NumPy / Pillow | 1.26.4 / 11.3.0 |
| MuJoCo / robosuite | 2.3.7 / 1.4.0 |
| PyOpenGL / glfw | 3.1.10 / 2.10.0 |
| tensorflow-cpu / tensorflow-datasets | 2.20.0 / 4.9.3 |
| Renderer | OSMesa, Mesa 21.2.6, llvmpipe LLVM 12.0.0 |

Run the following from the repository root:

```bash
conda create -n turbovla-libero python=3.9.25 -y
conda activate turbovla-libero
export PYTHONNOUSERSITE=1

# Use the reference CUDA 12.1 PyTorch build before installing project dependencies.
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[libero]"
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) separately, then rerun `python -m pip install -e ".[libero]"` and `python -m pip check` to check the pinned dependencies. Keep the LIBERO source, scene assets, BDDL files, and initial states fixed across comparisons. DINOv3, BERT, and GroundingDINO assets must also be prepared separately; use the same local copies across runs.

OSMesa is a native library and is not installed or pinned by `pip`. Install a build matching **Mesa 21.2.6 / llvmpipe LLVM 12.0.0**. If it is outside the system library search path, set the following before starting Python, replacing the placeholder with the actual library path:

```bash
export OSMESA_LIBRARY=/absolute/path/to/libOSMesa.so.8
export LD_LIBRARY_PATH="$(dirname "$OSMESA_LIBRARY")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Set both `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` explicitly in the evaluation shell, as in the example below. Existing environment variables take precedence over the evaluator defaults. Selecting `osmesa` alone does not fix the Mesa/LLVM version.

## Data

```text
data/libero/
|-- libero_10_no_noops/1.0.0/
|-- libero_goal_no_noops/1.0.0/
|-- libero_object_no_noops/1.0.0/
`-- libero_spatial_no_noops/1.0.0/
```

The repository does not redistribute demonstrations. Use `scripts/libero/regenerate_libero_no_noops.py` with the official LIBERO data, then convert the derived trajectories to TFDS/RLDS. Released normalization statistics are in `experiments/libero/configs/libero_all4_stats.json`.

## Training

The released recipe uses full DINOv3 ViT-B unfreezing, global batch 128 on four GPUs (per-device batch size 8 with 4 gradient-accumulation steps), 40k optimizer steps, 10k warmup steps, learning rate `5e-5`, seed 42, FP32 policy parameters, and BF16 autocast for DINOv3.

```bash
torchrun --nproc_per_node=4 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path facebook/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path google-bert/bert-base-uncased \
  --allow_hf_download \
  --pretrained_init_ckpt /path/to/groundingdino_swint_ogc.pth \
  --batch_size 8 \
  --checkpoint_dir outputs/checkpoints
```

## Evaluation

One invocation evaluates one checkpoint on one suite. Checkpoint selection and multi-GPU scheduling are intentionally outside the evaluator.

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export USE_TORCH=1 USE_TF=0 USE_FLAX=0
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  --dinov3_path /path/to/dinov3-vitb \
  --bert_path /path/to/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --num_steps_wait 10 --env_img_res 256 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path outputs/evaluation/object.json
```


Use `--dry_run_model_load true` to validate checkpoint compatibility without a simulator rollout. Complete release checkpoints contain BERT, DINOv3, the vision-language interaction module, and the action head in one strict-loadable state dict. `configs/online_text_layout.json` stores tokenizer padding metadata for the 40 standard instructions; it contains no embeddings or cached model outputs.

The rollout protocol is derived from [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter/commit/23fa0c9c159e2aa04341cdd3e924f44061311060). See `third_party/licenses/VLA-Adapter.txt` for the retained license and attribution.
