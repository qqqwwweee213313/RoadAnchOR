# Training RoadAnchOR

[← Repository overview](../README.md) · [Project Page](https://qqqwwweee213313.github.io/RoadAnchOR/)

Run the following commands from the repository root after cloning.

## Installation

Use Linux with an NVIDIA GPU and a CUDA development toolkit. The supplied environment uses Python 3.8.20, PyTorch 2.4.1+cu118, torchvision 0.19.1+cu118, NumPy 1.20.0, Numba 0.48.0, timm 1.0.22, and torchmetrics 0.11.4. This records the existing environment; a clean installation and full training run have not been revalidated for this release. Refer to the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/) for the corresponding wheels.

The code is a fork of an mmcv/mmdet3d-style stack with custom CUDA extensions,
so it needs to be built against your local PyTorch and CUDA toolkit.

```bash
conda create -n roadanchor python=3.8 -y
conda activate roadanchor

# PyTorch / torchvision versions observed in the supplied environment
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .            # builds mmcv._ext, iou3d_cuda, roiaware_pool3d_ext
```

If you prefer not to install the package into your environment (for example
because another checkout of a similar stack is already installed editable),
build the extensions in place and put the repository on `PYTHONPATH` instead:

```bash
python setup.py build_ext --inplace
export PYTHONPATH=/path/to/RoadAnchOR:$PYTHONPATH
```

> The compiled `mmcv/_ext*.so` embeds the absolute path of the directory it was
> built in and targets the compute capability of the build machine. Always
> rebuild it on the machine you train on; never copy a prebuilt `.so` between
> checkouts or between GPU generations (on a different architecture some ops,
> including deformable attention, are silently skipped).

---

## Data preparation

This repository does not ship data. Download the Bench2Drive sensor data and
the HD maps from the official release, generate the annotation pickles from
them, then arrange everything like this (or symlink it in):

```
data/
├── bench2drive/
│   ├── v1/                     # per-route sensor recordings
│   └── maps/                   # per-town HD map archives (*_HD_map.npz)
└── infos/
    ├── b2d_infos_train.pkl     # ~2.2 GB   234,769 training frames
    ├── b2d_infos_val.pkl       # ~150 MB    12,806 validation frames
    └── b2d_map_infos.pkl       # ~5.9 GB    per-town map index
```

All paths in the config are relative to the repository root, so run training
from there. Sizes are approximate and depend on the release you download.

Where each piece comes from:

- **Sensor recordings** (`data/bench2drive/v1/`) - the Bench2Drive tarballs at
  <https://huggingface.co/datasets/rethinklab/Bench2Drive>.
- **HD maps** (`data/bench2drive/maps/`) - the twelve `*_HD_map.npz` archives
  at <https://huggingface.co/datasets/rethinklab/Bench2Drive-Map>; extract them
  into that directory.
- **Annotation pickles** (`data/infos/`) - not a download.
  `b2d_infos_train.pkl`, `b2d_infos_val.pkl` and `b2d_map_infos.pkl` are
  generated from the raw
  dataset by `mmcv/datasets/prepare_B2D.py` of the upstream Bench2Drive model
  zoo (`python prepare_B2D.py --workers 16`, run from `mmcv/datasets`; about an
  hour for the base split). That preprocessing script is **not** part of this
  extraction, which carries the training path only - take it from the upstream
  release.

The map annotations released with the benchmark store the same
`(road_id, lane_id)` polyline an integer number of times, coordinates included,
so roughly half of the per-frame ground truth is a duplicate of the same
geometry. Because the Hungarian matcher is 1:1, that many queries end up
assigned to the same element. The config enables `map_gt_dedup=True`, which
keeps one instance per (class, Chamfer < `eps`) group. `eps` is not a tuning
knob: the distance distribution of duplicate pairs is bimodal — near 0 m for
copies, above 1.37 m for genuinely distinct elements — so anything in
0.05–1.0 m gives the same result.

---

## Pre-trained weights

```
ckpts/
├── resnet50-19c8e357.pth   # torchvision ImageNet ResNet-50 (image backbone init)
└── vad_b2d_base.pth        # perception pre-training from the upstream baseline
```

`vad_b2d_base.pth` is the perception (backbone / BEV encoder / detection / map)
pre-training released with the upstream Bench2Drive baseline. Both files are in
the upstream model zoo at <https://huggingface.co/rethinklab/Bench2DriveZoo>;
`resnet50-19c8e357.pth` is also the stock torchvision weight at
<https://download.pytorch.org/models/resnet50-19c8e357.pth>. Training starts
from it by default (`load_from` in the config).
Loading is non-strict, so the planning submodules — which do not exist in the
perception checkpoint — are initialised from scratch and reported as missing
keys. Measured against this configuration: the model has 1,256 tensors, the
checkpoint has 1,062, **1,000 match by name and shape**, 256 are missing (all
under `pts_bbox_head.ra_*`, `score_pt_mlp`, `score_head_mlp`) and 62 keys in
the checkpoint are unused. Because loading is non-strict, a key mismatch is
only a log line — check that number if you change the head.

Before sharing any checkpoint, strip its metadata: an mmcv checkpoint embeds
the full training config and the absolute paths of the machine that produced
it.

```bash
python tools/convert_legacy_checkpoint.py in.pth out.pth --strip-meta
```

---

## Training

Single GPU, plain Python:

```bash
python adzoo/vad/train.py \
    adzoo/vad/configs/roadanchor/roadanchor_b2d.py \
    --launcher none \
    --work-dir runs/roadanchor_b2d \
    --no-validate \
    --cfg-options load_from=ckpts/vad_b2d_base.pth \
    --seed 0 --deterministic
```

Multiple GPUs (four shown):

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    adzoo/vad/train.py \
    adzoo/vad/configs/roadanchor/roadanchor_b2d.py \
    --launcher pytorch \
    --work-dir runs/roadanchor_b2d \
    --no-validate \
    --cfg-options load_from=ckpts/vad_b2d_base.pth \
    --seed 0 --deterministic
```

To use another checkpoint, set `--cfg-options load_from=ckpts/your_checkpoint.pth`.
Use this option instead of `--load-from`: the runner reads `cfg.load_from`,
which can otherwise reload the default checkpoint after the initial model load.

With `samples_per_gpu=4` on four GPUs, one epoch of the full training split is
about 14.7 k iterations; the schedule runs for six epochs.

In-loop validation is disabled with `--no-validate` (open-loop planning metrics
are not computed during training). Removing the flag enables the validation hooks configured by the training runner; this does not provide a closed-loop evaluation harness.

### Optional Weights & Biases logging

Logging is off unless `WANDB_PROJECT` is set. No credentials are stored in this
repository; everything is taken from the usual environment variables.

```bash
pip install wandb                      # optional logging dependency
export WANDB_PROJECT=my_project
export WANDB_NAME=roadanchor_b2d          # optional, defaults to the work-dir name
export WANDB_API_KEY=...                  # or run `wandb login` beforehand
```

To also stream the scalar training logs, add `dict(type='WandbLoggerHook')` to
`log_config.hooks` in the config.

---

## Configuration

Everything lives in `adzoo/vad/configs/roadanchor/roadanchor_b2d.py`.

### The three flags that define this model

All three are **non-default**, and the head inherits `**kwargs` from a base
class that silently ignores unknown keys — so a typo does not raise, it just
trains a different model. Verify them against the *built* head:

```bash
python tools/check_config_flags.py adzoo/vad/configs/roadanchor/roadanchor_b2d.py
```

This check passed in the supplied environment for all three flags and the
`PlanCollisionLossV7` loss type. It checks model construction and configuration;
it does not run training or closed-loop evaluation.

| key | default | this config | effect |
|---|---|---|---|
| `score_target_use_cfar` | `True` | `False` | centerline importance target is `I = A · B` (drops `C_far`) |
| `plan_traj_cumsum` | `False` | `True` | decoder output is a per-step displacement, integrated in the head |
| `plan_col_v7` | `False` | `True` | forwards GT trajectory, object extents and headings to `PlanCollisionLossV7` |

### Other key hyper-parameters

| key | value | meaning |
|---|---|---|
| `point_cloud_range` | `[-15, -30, -2, 15, 30, 2]` | BEV extent (m) |
| `bev_h_` / `bev_w_` | `200` / `200` | BEV grid |
| `queue_length` | `4` | temporal frames per sample |
| `num_query` | `300` | object queries |
| `map_num_vec` | `100` | map polyline queries |
| `ego_fut_mode` | `6` | 6 road queries × 6 distance queries produce 36 trajectory candidates |
| `valid_fut_ts` | `6` | planning horizon (steps of 0.5 s → 3 s) |
| `score_cand_lines` | `40` | centerline candidates scored per frame |
| `score_lane_w` | `3.5` | `W`, the single scale constant of the score target |
| `loss_plan_col.loss_weight` | `0.5` | collision loss weight |
| `loss_plan_col.dis_thresh` | `3.0` | `D`, shared safety margin (m) |
| `total_epochs` | `6` | |
| `optimizer.lr` | `2e-4` | AdamW, `img_backbone` at `lr_mult=0.1` |
| `lr_config` | cosine | 500-iteration linear warm-up, `min_lr_ratio=1e-3` |

---
