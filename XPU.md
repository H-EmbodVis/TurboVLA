# Getting Started on XPU

Tested on the following hardware specification and software version.

Hardware Specification:
 - CPU: Intel® Core™ Ultra X7 Processor 358H
 - CPU Cores: 16 (4 Performance-cores, 8 Efficient-cores, and 4 Low Power Efficient-cores)
 - CPU Threads: 16
 - Memory: 32 GB
 - iGPU: Intel® Arc™ B390 GPU

Software Version:
 - Ubuntu 26.04 LTS
 - Intel Graphics Compute Runtime 26.40.40196.8
 - Python 3.10.22
 - PyTorch 2.14.1+xpu
 - Torchvision 0.29.2+xpu
 - LIBERO 0.1.1
 - MuJoCo 2.3.7

## LIBERO Environment

### Installation

```bash
$ sudo apt install libosmesa6
```

```bash
$ conda create -n turbovla-libero python=3.10 -y
$ conda activate turbovla-libero
```

Downgrade MuJoCo from version 3.0.0 to 2.3.7.

```bash
$ cd TurboVLA

$ export PYTHONNOUSERSITE=1
$ export CMAKE_POLICY_VERSION_MINIMUM=3.5

$ python -m pip install -e ".[libero_xpu]"
$ python -m pip install torch==2.14.1+xpu torchvision==0.29.2+xpu torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
$ python -m pip install mujoco==2.3.7
```

```bash
$ python -m pip freeze | grep mujoco
```
```
mujoco==2.3.7
```

### Dataset and Model Preparation

Download required models.

```bash
$ hf download H-EmbodVis/TurboVLA \
  --local-dir pretrained/TurboVLA

$ hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir pretrained/dinov3-vitb16-pretrain-lvd1689m

$ hf download google-bert/bert-base-uncased \
  --local-dir pretrained/bert-base-uncased
```

Download demo dataset.

```bash
$ mkdir -p ./data/libero
$ hf download --repo-type dataset yifengzhu-hf/LIBERO-datasets \
  libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
  --local-dir ./data/libero
```

```bash
$ ls -l ./data/libero/
```
```
total 4
drwxrwxr-x 2 eka eka 4096 Nov  8 18:52 libero_spatial
```

```bash
$ ls -l ./data/libero/libero_spatial
```
```
total 496860
-rw-rw-r-- 1 eka eka 508779600 Nov  8 18:52 pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

Convert dataset as required by TurboVLA.

```bash
$ python scripts/libero/regenerate_libero_no_noops.py \
  --libero_task_suite libero_spatial \
  --libero_raw_data_dir ./data/libero/libero_spatial \
  --libero_target_dir ./data/libero/libero_spatial_no_noops/1.0.0/
```

Please ignore the following error as the previous `hf` command only download 1 out of 10 demo files.

```
...
Total # episodes replayed: 50, Total # successes: 45 (90.0 %)
  Total # no-op actions filtered out: 9
Saved regenerated demos for task 'pick up the black bowl between the plate and the ramekin and place it on the plate' at: ./data/libero/libero_spatial_no_noops/1.0.0/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
 10%|█████                                             | 1/10 [03:39<32:51, 219.04s/it]
...
AssertionError: Cannot find raw data file ./data/libero/libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5.
...
```

```bash
$ ls -l ./data/libero/
```
```
total 8
drwxrwxr-x 2 eka eka 4096 Nov  8 18:52 libero_spatial
drwxrwxr-x 3 eka eka 4096 Nov  8 18:52 libero_spatial_no_noops
```

```bash
$ ls -l ./data/libero/libero_spatial_no_noops/1.0.0/
```
```
total 1727928
-rw-rw-r-- 1 eka eka     136314 Nov  8 18:56 metainfo.json
-rw-rw-r-- 1 eka eka 1769253800 Nov  8 18:56 pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

### Evaluation

__Note:__ XPU takes about 5 minutes to warm up for the first time only.

System utilization:
 - CPU: 13%
 - RAM: 10GB (including iGPU RAM)
 - iGPU: 4% average, 7% maximum
 - iGPU RAM: 2GB

```bash
$ export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
$ export USE_TORCH=1 USE_TF=0 USE_FLAX=0
$ export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$ export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

$ python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  --dinov3_path pretrained/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path pretrained/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --num_steps_wait 10 --env_img_res 256 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path outputs/evaluation/libero_object.json
```
```
/home/eka/miniconda3/envs/turbovla-libero/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[TurboVLAPolicy] model source: turbovla.models.turbovla
[TurboVLAPolicy] precision=bf16, dinov3_output_hidden_states=True
[TurboVLAPolicy] strict checkpoint load: pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth (672 tensors)
[TurboVLAPolicy] precision=bf16, model_parameter_dtype=bfloat16
[TurboVLAPolicy] suite stats: path=experiments/libero/configs/libero_all4_stats.json, key=libero_all4_no_noops, normalize_binary_gripper=False
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tasks:   0%|                                                      | 0/10 [00:00<?, ?it/s][robosuite WARNING] No private macro file found! (__init__.py:7)
2026-11-08 19:02:53,449 [WARNING] No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
2026-11-08 19:02:53,449 [WARNING] It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /home/eka/miniconda3/envs/turbovla-libero/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
2026-11-08 19:02:53,449 [WARNING] To setup, run: python /home/eka/miniconda3/envs/turbovla-libero/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
Local assets not found. Downloading from HuggingFace Hub...
Assets already downloaded at /home/eka/.cache/libero/assets
/home/eka/miniconda3/envs/turbovla-libero/lib/python3.10/site-packages/transformers/modeling_utils.py:1742: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
  warnings.warn(
2026-11-08 19:03:07,912 [INFO] task=0 episode=0 success=True running=1/1 100.0%
2026-11-08 19:03:19,395 [INFO] task=0 episode=1 success=True running=2/2 100.0%                        | 1/50 [00:12<10:16, 12.59s/it]
...
2026-11-08 20:31:24,063 [INFO] task=9 episode=48 success=True running=499/499 100.0%████████████████▊  | 48/50 [07:52<00:19,  9.66s/it]
2026-11-08 20:31:33,640 [INFO] task=9 episode=49 success=True running=500/500 100.0%█████████████████▉ | 49/50 [08:02<00:09,  9.83s/it]
2026-11-08 20:31:33,751 [INFO] task=9 success_rate=1.0000 (50/50) tasks: 100%|█████████████████████████| 10/10 [1:28:40<00:00, 532.03s/it]
2026-11-08 20:31:33,752 [INFO] Final success rate: 1.0000 (500/500)
2026-11-08 20:31:33,752 [INFO] Saved result json: outputs/evaluation/libero_object.json
```

```bash
$ cat outputs/evaluation/libero_object.json
```
```
{
  "script": "turbovla_libero_evaluation",
  "ckpt_path": "pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth",
  "task_suite_name": "libero_object",
  "requested_task_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "num_trials_per_task": 50,
  "num_open_loop_steps": 12,
  "seed": 7,
  "precision": "bf16",
  "dinov3_output_hidden_states": true,
  "tasks": [
    {
      "task_id": 0,
      "task_description": "pick up the alphabet soup and place it in the basket",
      "episodes": 50,
      "successes": 50,
      "success_rate": 1.0
    },
    ...
    {
      "task_id": 9,
      "task_description": "pick up the orange juice and place it in the basket",
      "episodes": 50,
      "successes": 50,
      "success_rate": 1.0
    }
  ],
  "total_episodes": 500,
  "total_successes": 500,
  "overall_success_rate": 1.0
}
```
