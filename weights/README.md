# CACrackNet Paper Weights

This directory documents the checkpoints used by the current paper version.
The `.pth` files are distributed separately and should not be committed to Git.

Download:

```text
https://drive.google.com/drive/folders/1kHzpeYTPCxkS9_w5pkPA3O5aHq6v6X3o?usp=sharing
```

Expected layout:

```text
weights/
  DeepCrack/checkpoint_best.pth
  Crack500/checkpoint_best.pth
  CrackMap/checkpoint_best.pth
```

## Selected Results

| Dataset | Remote run | Best epoch | ODS | OIS | P | R | F1 | mIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepCrack | `2026_05_29_02:17:07_Dataset->DeepCrack` | 63 | 0.9149412119 | 0.9217295498 | 0.9136153338 | 0.9446265703 | 0.9288621868 | 0.9199894078 |
| Crack500 | `2026_06_03_07:27:24_Dataset->Crack500` | 20 | 0.7398571956 | 0.7512494590 | 0.7791525156 | 0.7864974059 | 0.7828077323 | 0.7896508400 |
| CrackMap | `2026_06_04_01:28:21_Dataset->CrackMap` | 27 | 0.7911313377 | 0.7969505773 | 0.7414537950 | 0.8510680859 | 0.7924885299 | 0.8211698228 |

## Original Archive Hashes

The original full training checkpoints included optimizer states and historical
argument objects. Release checkpoints are converted to model-state-only files.

```text
b5bac765d4cdbcebd73836a91bce823406f661d23bc1929176d459591da52670  original DeepCrack/checkpoint_best.pth
203480af881ad8c8b971dffded670067418db928521c199fa5ce28f2f7fab4ca  original Crack500/checkpoint_best.pth
2645c7f93ffa5bca9c4649ac6acd043795ee5b034670f468fb160913c708b19e  original CrackMap/checkpoint_best.pth
```

## Release Checkpoint Hashes

The local release directory was synchronized from the verified server package:

```text
/data/users/hanyaojia/model/CACrackNet_release_20260609
```

```text
fcc7b86b0e07e58ffe24e995dd1170b12daada2a27e3f560759de8864bde0148  weights/DeepCrack/checkpoint_best.pth
3a3d61a69294e46d99a02fd580e029ad0fe1f995066cd6ee8baa037ce9ba8d95  weights/Crack500/checkpoint_best.pth
9bb9a9dcc1244fa34ac4d5d7f2354198304af5651419f60b4583f87454ff44f8  weights/CrackMap/checkpoint_best.pth
```

## Release Re-Verification

The synchronized release was re-run on the server with the converted
model-state-only checkpoints.

```text
DeepCrack dataset: /data/users/hanyaojia/model/scsegamba-AGCF/data/DeepCrack
Crack500 dataset:  /data/users/zhangzj/data/datasets/Crack500
CrackMap dataset:  /data/users/zhangzj/data/datasets/CrackMap
```

| Dataset | Results dir | ODS | OIS | P | R | F1 | mIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepCrack | `results/release_DeepCrack_test` | 0.9149274753 | 0.9216782183 | 0.9141603146 | 0.9442309839 | 0.9289523622 | 0.9199846245 |
| Crack500 | `results/release_Crack500_test` | 0.7397804493 | 0.7510598555 | 0.7798494834 | 0.7855936165 | 0.7827110114 | 0.7896071498 |
| CrackMap | `results/release_CrackMap_test` | 0.7910607916 | 0.7969483052 | 0.7420849811 | 0.8504995631 | 0.7926021315 | 0.8211227727 |
