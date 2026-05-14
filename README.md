# ELF: Embedded Language Flows (Unofficial PyTorch Reproduction)

> [!CAUTION]
>
> The results are not directly comparable with baselines ([MDLM](https://github.com/kuleshov-group/mdlm), [Duo](https://github.com/s-sahoo/duo), [FLM](https://github.com/david3684/flm), ...)
> due to tokenization and preprocessing differences used in the ELF paper.
>
> Specifically, ELF uses a custom preprocessed OpenWebText dataset (see [`openwebtext-t5`](https://huggingface.co/datasets/embedded-language-flows/openwebtext-t5)).
> This is tokenized with the T5 tokenizer, not the GPT-2 tokenizer which is used in the standard setting in the literature. In addition, the paper's preprocessing pipeline includes a custom packing scheme with full details not disclosed in the paper.

---

 **This is an unofficial PyTorch reproduction** of *ELF: Embedded Language Flows*.\
  The official JAX/TPU implementation is at <https://github.com/lillian039/ELF>, and the official checkpoints are in HuggingFace at [`embedded-language-flows`](https://huggingface.co/embedded-language-flows).

 This repository was developed using [Claude Code](https://claude.com/claude-code).

## Reproduction status

OpenWebText (unconditional), ELF-B (105M), 32-step SDE, γ=1.5, SC-CFG=3:

| Metric | Paper (TPU v5p-64) | Reproduction (8× B200 DDP, Lightning) |
| --- | --- | --- |
| Gen. PPL ↓ | 24.1 | **25.61** |
| Entropy | 5.15 | **5.20** |

Per-epoch results (32-step SDE, 256 samples):

| Epoch | Step | Gen. PPL | Entropy |
| --- | --- | --- | --- |
| 1 | 38 034  | 2.73¹  | 0.70¹ |
| 2 | 76 068  | 37.11  | 5.17 |
| 3 | 114 102 | 28.63  | 5.21 |
| 4 | 152 136 | 25.00  | 5.16 |
| 5 | 190 170 | 25.58  | 5.19 |
| 6 | 228 204 | 26.11  | 5.21 |

All samples used for the measurements can be found in
[`reproduction/elf_b-owt/eval1000/metrics.jsonl`](reproduction/elf_b-owt/eval1000/metrics.jsonl)
and [`reproduction/elf_b-owt/per_epoch/metrics.jsonl`](reproduction/elf_b-owt/per_epoch/metrics.jsonl).

## TODO
- [ ] Train ELF and/or some of the baselines ([MDLM](https://github.com/kuleshov-group/mdlm), [Duo](https://github.com/s-sahoo/duo), [FLM](https://github.com/david3684/flm), ...) in a directly comparable setting (https://huggingface.co/datasets/Skylion007/openwebtext).

## What's in this repo

- [`pytorch_lightning/`](pytorch_lightning/): model, training
  script (`train_lightning.py`), eval (`eval_lightning.py`), and
  utilities. 8-GPU CUDA DDP via PyTorch Lightning.
- [`reproduction/elf_b-owt/`](reproduction/elf_b-owt/): config snapshot, 1000 final
 samples, and per-epoch samples. The
  checkpoint is hosted separately (see [Quickstart](#quickstart-evaluate-the-reproduced-checkpoint)).

## Quickstart — evaluate the reproduced checkpoint

```bash
# 1. Environment (conda)
conda env create -f environment.yml -n elf-pytorch && conda activate elf-pytorch

# 2. Download the reproduced final EMA checkpoint (1.4 GB)
pip install huggingface_hub
huggingface-cli download Ugness/elf-torch last.ckpt \
    --local-dir reproduction/elf_b-owt/

# 3. Run the 1000-sample evaluation
cd pytorch_lightning/
torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
    --num_samples 1000
# Expected: Gen. PPL ≈ 25.6, sample entropy ≈ 5.20.
```

### Per-epoch checkpoints

The checkpoints are under this HF repo:
[`checkpoints/`](https://huggingface.co/Ugness/elf-torch/tree/main/checkpoints).
```bash
# Example: pull epoch 4 ckpt.
huggingface-cli download Ugness/elf-torch \
    checkpoints/checkpoint_epoch03_step00152136.ckpt \
    --local-dir reproduction/elf_b-owt/
```

## Quickstart — train from scratch

```bash
cd pytorch_lightning/
torchrun --nproc_per_node=8 --master_port=29501 train_lightning.py \
    --config configs/training_configs/train_owt_ELF-B.yml
```

## Reproduction details

- **Hardware:** 8× NVIDIA B200 (sm_100), CUDA 12.8.
  `broadcast_buffers=False`. See `pytorch_lightning/train_lightning.py`.
- **Wall-clock:** ~3 hours per epoch.


### Differences vs the paper run

| Aspect | Paper | This reproduction |
| --- | --- | --- |
| Hardware | TPU v5p-64 | 8× B200 DDP |
| Framework | JAX/Flax | PyTorch Lightning |
| Epochs | 5 | 6 (one extra to reach entropy ≈ 5.20) |
| Optimizer / objective | Muon + L2 denoise + CE decode (decoder_prob=0.2) | Unchanged |
| Schedule, noise scale, time schedule, SC, CFG | Unchanged | Unchanged |
