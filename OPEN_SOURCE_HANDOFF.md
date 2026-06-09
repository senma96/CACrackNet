# CACrackNet Open-Source Handoff

Date: 2026-06-09

## Local Release

```text
/Users/cosen/work/科研/裂缝检测/2026/cacrackNet/code/CACrackNet_release
```

This directory was synchronized from the verified server release:

```text
/data/users/hanyaojia/model/CACrackNet_release_20260609
```

The server tarball used for synchronization:

```text
/data/users/hanyaojia/model/CACrackNet_release_20260609_with_weights.tar.gz
```

## Server

```text
host: 192.168.18.171
user: hanyaojia
env:  /data/users/hanyaojia/anaconda3/envs/SCSegamba
```

Do not store the password in repository files. SSH/SCP template:

```bash
sshpass -p '<PASSWORD>' ssh -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  hanyaojia@192.168.18.171
```

## Verified Code Path

The release fixes the paper architecture and removes research-only switches:

- CAB encoder blocks.
- SCGF decoder cascade.
- Decoder-side EDFFN.
- DySample multi-scale alignment.
- No active deformable scan / SS2D path in the selected release checkpoints.

Historical names are kept only for checkpoint compatibility:

- SCGF class name remains `AGCF`.
- Encoder internal module name remains `freq_module`.

## Weights

```text
weights/DeepCrack/checkpoint_best.pth
weights/Crack500/checkpoint_best.pth
weights/CrackMap/checkpoint_best.pth
```

Release SHA256:

```text
fcc7b86b0e07e58ffe24e995dd1170b12daada2a27e3f560759de8864bde0148  weights/DeepCrack/checkpoint_best.pth
3a3d61a69294e46d99a02fd580e029ad0fe1f995066cd6ee8baa037ce9ba8d95  weights/Crack500/checkpoint_best.pth
9bb9a9dcc1244fa34ac4d5d7f2354198304af5651419f60b4583f87454ff44f8  weights/CrackMap/checkpoint_best.pth
```

## Server Verification

CUDA strict-load forward passed for all three checkpoints on:

```text
torch 1.13.1+cu116
GPU: Tesla V100-PCIE-32GB
output shape: (1, 1, 512, 512)
```

Full release inference and `eval/evaluate.py` recomputation were run in the
server release directory. The generated prediction folders were removed from
the clean release package after verification.

```text
DeepCrack: results/release_DeepCrack_test
Crack500:  results/release_Crack500_test
CrackMap:  results/release_CrackMap_test
```

Dataset roots:

```text
DeepCrack: /data/users/hanyaojia/model/scsegamba-AGCF/data/DeepCrack
Crack500:  /data/users/zhangzj/data/datasets/Crack500
CrackMap:  /data/users/zhangzj/data/datasets/CrackMap
```

Recomputed metrics:

```text
DeepCrack ODS 0.9149274753 / OIS 0.9216782183 / P 0.9141603146 / R 0.9442309839 / F1 0.9289523622 / mIoU 0.9199846245
Crack500  ODS 0.7397804493 / OIS 0.7510598555 / P 0.7798494834 / R 0.7855936165 / F1 0.7827110114 / mIoU 0.7896071498
CrackMap  ODS 0.7910607916 / OIS 0.7969483052 / P 0.7420849811 / R 0.8504995631 / F1 0.7926021315 / mIoU 0.8211227727
```
