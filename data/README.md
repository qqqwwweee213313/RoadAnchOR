# Datasets

This directory is ignored by git. See the [training guide](../docs/TRAINING.md#data-preparation) for the expected layout:

```
data/
├── bench2drive/          # sensor data released with the benchmark
│   └── maps/             # per-town HD map archives
└── infos/                # preprocessed annotation pickles
    ├── b2d_infos_train.pkl
    ├── b2d_infos_val.pkl
    └── b2d_map_infos.pkl
```
