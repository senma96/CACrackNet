# CACrackNet

Official release code for **CACrackNet: Continuity-Aware Convolutional Network
for Lightweight Crack Segmentation**.

This repository accompanies the paper:

```text
CACrackNet: Continuity-Aware Convolutional Network for Lightweight Crack Segmentation
```

This release keeps only the modules used by the paper configuration:

- Continuity-Aware Bottleneck (CAB) encoder blocks.
- Spatial-Channel Gated Fusion (SCGF) decoder cascade.
- Decoder-side EDFFN refinement.
- DySample-based multi-scale feature alignment.

Experimental module switches from the research workspace were removed so that
the released model matches the paper checkpoint path.

## Environment

The verified server environment used during paper experiments was:

```text
Host: 192.168.18.171
Project: /data/users/hanyaojia/model/scsegamba-AGCF
Conda env: /data/users/hanyaojia/anaconda3/envs/SCSegamba
GPU used for FPS: Tesla V100-PCIE-32GB
```

Install dependencies in a new environment:

```bash
pip install -r requirements.txt
```

The code expects a PyTorch/CUDA installation compatible with your GPU.

## Dataset Layout

Each dataset root should use the original folder names:

```text
DatasetRoot/
  train_img/
  train_lab/
  test_img/
  test_lab/
```

Labels are binary masks. Images are resized to `512x512` by default.

## Weights

The model weights are distributed separately and are not tracked by Git.
Download them from Google Drive:

```text
https://drive.google.com/drive/folders/1kHzpeYTPCxkS9_w5pkPA3O5aHq6v6X3o?usp=sharing
```

After downloading, place the checkpoints under:

```text
weights/
  DeepCrack/checkpoint_best.pth
  Crack500/checkpoint_best.pth
  CrackMap/checkpoint_best.pth
```

See `weights/README.md` for metrics and hashes.

## Test

```bash
python main.py \
  --phase test \
  --dataset_path /path/to/DeepCrack \
  --test_checkpoint weights/DeepCrack/checkpoint_best.pth \
  --test_results_dir results/DeepCrack
```

Then recompute metrics:

```bash
python eval/evaluate.py --results_dir results/DeepCrack
```

## Predict Custom Images

```bash
python predict.py \
  --input /path/to/images_or_one_image \
  --output results/predict \
  --test_checkpoint weights/DeepCrack/checkpoint_best.pth
```

The saved masks are binarized with threshold `0.5` by default.

## Train

```bash
python main.py \
  --phase train \
  --dataset_path /path/to/DeepCrack \
  --batch_size_train 4 \
  --epochs 100
```

The released model fixes the paper architecture. Module-level switches such as
alternative GBC variants, deformable scanning, and non-paper fusion strategies
are intentionally not exposed.

## Complexity And FPS

```bash
python measure_complexity.py
python benchmark_fps.py \
  --test_checkpoint weights/DeepCrack/checkpoint_best.pth \
  --batch_size 1
```

Verified V100 FP32 results for the DeepCrack checkpoint:

```text
Params=2.36M, FLOPs=11.55G, model size=9.01MB
batch=1, input=512x512, mean latency=29.08 ms, FPS=34.39
```

## Paper Results

The selected paper artifacts use the same configuration across datasets:

```text
CAB encoder + early SCGF decoder + decoder EDFFN
```

| Dataset | ODS | OIS | P | R | F1 | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| DeepCrack | 0.9149 | 0.9217 | 0.9136 | 0.9446 | 0.9289 | 0.9200 |
| Crack500 | 0.7399 | 0.7512 | 0.7792 | 0.7865 | 0.7828 | 0.7897 |
| CrackMap | 0.7911 | 0.7970 | 0.7415 | 0.8511 | 0.7925 | 0.8212 |

The older research workspace contains additional checkpoints and ablation
experiments. They are not part of this clean release.
