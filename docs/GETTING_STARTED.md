# Getting Started

AutoReg3D model configs are in [tools/cfgs/autoreg3d_models](../tools/cfgs/autoreg3d_models).

## Dataset Preparation

This release uses **nuScenes**. Download the
[nuScenes 3D object detection dataset](https://www.nuscenes.org/download) and
place it at `data/nuscenes`:

```
autoreg3d
├── data
│   └── nuscenes
│       └── v1.0-trainval
│           ├── samples
│           ├── sweeps
│           ├── maps
│           └── v1.0-trainval
├── pcdet
└── tools
```

For full dataset preparation details, refer to the NuScenes section of the
[OpenPCDet getting-started guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#nuscenes-dataset).

## Training & Testing

See the [Training](../README.md#training) and [Evaluation](../README.md#evaluation)
sections of the README for AutoReg3D-specific commands. The general OpenPCDet
usage applies:

```bash
# train with multiple GPUs
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# train with a single GPU
python train.py --cfg_file ${CONFIG_FILE}

# test with a checkpoint
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}

# test with multiple GPUs
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```
