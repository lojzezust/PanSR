# PanSR — Panoptic Segmentation for Maritime Scenes

PanSR is a transformer-based panoptic segmentation model for maritime obstacle scenes,
trained and evaluated on the [LaRS](https://lojzezust.github.io/lars-dataset/) dataset. It is
built on [MaskDINO](https://github.com/IDEA-Research/MaskDINO) / Mask2Former and adds four
contributions that make it well suited to thin, small, and densely packed maritime obstacles.

## The four contributions (and where they live in the code)

The model entry point is [`pansr/pansr_model.py`](pansr/pansr_model.py). Each paper contribution
has its own clearly named module:

| # | Contribution | Module | What it does |
|---|--------------|--------|--------------|
| 1 | **Object-Centric Proposal (OCP)** | [`pansr/modeling/ocp.py`](pansr/modeling/ocp.py) | An FCOS-style dense head over the FPN that produces object-centric proposals used to initialize the decoder's object queries. |
| 2 | **Object-centric mask prediction** | [`pansr/modeling/object_centric_mask.py`](pansr/modeling/object_centric_mask.py) | Constrains each foreground mask to its predicted (dilated) box, suppressing mask leakage. |
| 3 | **Proposal-aware matching** | [`pansr/modeling/matching.py`](pansr/modeling/matching.py) (`proposal_aware_matching`) | Refines Hungarian assignment by removing low-IoU matches and recovering high-IoU false positives. |
| 4 | **Mask-conditioned queries** | [`pansr/modeling/mask_conditioned_queries.py`](pansr/modeling/mask_conditioned_queries.py) | Training-time auxiliary queries whose content is sampled *inside GT masks* and whose positions are noised GT boxes. |

Contributions (1), (2) and (4) are wired together in
[`pansr/modeling/transformer_decoder/pansr_decoder.py`](pansr/modeling/transformer_decoder/pansr_decoder.py);
(3) is applied in [`pansr/modeling/disjoint_criterion.py`](pansr/modeling/disjoint_criterion.py).

## Installation

Requirements:
- a CUDA 12.x toolkit (`nvcc`) and `gcc`,
- **Python 3.10 with development headers** (`Python.h`) — e.g. `sudo apt install python3.10-dev`.
  A `conda` Python already ships these headers.
- a venv tool: [`uv`](https://github.com/astral-sh/uv) (preferred) or `conda` or stdlib `venv`.

```bash
bash setup.sh                 # creates ./.venv, installs torch (cu124), detectron2, deps, builds the CUDA op
source .venv/bin/activate
```

`setup.sh` is configurable via env vars (`PYTHON_VERSION`, `TORCH_VERSION`, `TORCH_CUDA_INDEX`,
`TORCH_CUDA_ARCH_LIST`). The default `TORCH_CUDA_ARCH_LIST=8.6` targets Ampere GPUs (e.g. RTX A4500);
set it to your GPU's compute capability if different. To build the venv from a specific interpreter
(e.g. a conda Python that has headers), set `PANSR_PYTHON=/path/to/python`.

> **Notes**
> - detectron2 and the MSDeformAttn op are compiled from source against the installed PyTorch. The
>   most common failure is `fatal error: Python.h: No such file or directory` — install the Python
>   dev headers (above) or point `PANSR_PYTHON` at an interpreter that has them.
> - If the from-source detectron2 build is troublesome on your platform, fall back to a prebuilt
>   combo (e.g. `torch==2.1.2` + the matching detectron2 wheel) and re-run the op build.
> - If `torch.cuda.is_available()` is `False` despite GPUs being present, set `CUDA_VISIBLE_DEVICES`.

Point the code at your LaRS dataset (used for training, evaluation, and inference metadata):

```bash
export LARS_ROOT=/path/to/LaRS/split_v0.9.3
```

## Pretrained weights

The released weights are an internal "AnchorFormer" checkpoint. Remap them to the PanSR module
layout once (this only renames the OCP submodule keys, see
[`tools/remap_weights.py`](tools/remap_weights.py)):

```bash
python tools/remap_weights.py \
    --src /path/to/AF_J22_Swin_L_bs16_90k/model_final.pth \
    --dst weights/pansr_lars_swin_l.pth \
    --validate --config-file configs/lars/panoptic/pansr_Swin_L.yaml
```

`--validate` builds PanSR and strict-loads the remapped weights, asserting 0 missing / 0 unexpected keys.

## Inference

```bash
python predict.py \
    --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    --weights weights/pansr_lars_swin_l.pth \
    --input assets/sample.jpg \
    --output out.png --vis
```

This writes a LaRS-format panoptic mask (`out.png`) and, with `--vis`, a human-friendly overlay
(`out_vis.png`). `--input` also accepts a glob (e.g. `'images/*.jpg'`) with `--output <dir>`.

## HuggingFace Hub

PanSR weights and config can be shared through the Hub via
[`pansr/hub.py`](pansr/hub.py) (`PanSRHF`, a `PyTorchModelHubMixin`).

```python
from pansr.hub import PanSRHF
model = PanSRHF.from_pretrained("lojze/pansr-lars-swin-l")   # downloads config + weights
```

```bash
# Load with predict.py directly from the Hub:
python predict.py --hf-model lojze/pansr-lars-swin-l --input assets/sample.jpg --output out.png

# Export a local checkpoint to a Hub folder and/or push it:
python tools/export_to_hub.py \
    --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    --weights weights/pansr_lars_swin_l.pth \
    --save-dir hf_export/pansr-lars-swin-l \
    --push-to lojze/pansr-lars-swin-l        # requires `huggingface-cli login`
```

(Rebuilding the model from the Hub requires the LaRS metadata, i.e. `LARS_ROOT` pointing at the dataset.)

## Training

```bash
export LARS_ROOT=/path/to/LaRS/split_v0.9.3

# ResNet-50
python train_net.py --num-gpus 4 --config-file configs/lars/panoptic/pansr_R50.yaml

# Swin-L (set MODEL.WEIGHTS to the ImageNet-22k Swin-L backbone for from-scratch training)
python train_net.py --num-gpus 4 --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    MODEL.WEIGHTS /path/to/swin_large_patch4_window12_384_22k.pkl
```

Evaluation only:

```bash
python train_net.py --eval_only --num-gpus 4 \
    --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    MODEL.WEIGHTS weights/pansr_lars_swin_l.pth
```

Copy-paste augmentation expects pre-extracted objects under `$LARS_ROOT/train/objects_v2`
(configurable via `INPUT.COPY_PASTE.OBJECTS_DIR`). The Swin-L ImageNet-22k backbone weights
(`swin_large_patch4_window12_384_22k.pkl`) are only needed for from-scratch training.

## Configuration

Configs use the detectron2 YACS system; PanSR-specific defaults are defined in
[`pansr/config.py`](pansr/config.py) under the `cfg.MODEL.PanSR` namespace. The contribution knobs are:

- `MODEL.PanSR.OBJECT_CENTRIC_MASK.{ENABLED,DILATE_AMOUNT,MIN_DILATE_PX}` — contribution (2)
- `MODEL.PanSR.MASK_COND_QUERIES.{MODE,NUM,NOISE_SCALE,MAX_GROUP_SIZE}` — contribution (4)
- `MODEL.PanSR.TWO_STAGE` — use OCP proposals to initialize queries (contribution 1)

## Repository layout

```
pansr/                         # the model package
  pansr_model.py               # PanSR meta-architecture (detectron2 META_ARCH "PanSR")
  config.py                    # add_pansr_config + cfg.MODEL.PanSR defaults
  hub.py                       # HuggingFace PyTorchModelHubMixin wrapper
  modeling/
    ocp.py                     # (1) Object-Centric Proposal head/module
    object_centric_mask.py     # (2) object-centric mask prediction
    matching.py                # HungarianMatcher + (3) proposal_aware_matching
    mask_conditioned_queries.py# (4) mask-conditioned queries
    transformer_decoder/pansr_decoder.py
    pixel_decoder/             # PanSREncoder + MSDeformAttn CUDA op (ops/)
    meta_arch/pansr_head.py    # PanSRHead
    backbone/swin.py           # Swin-L backbone (ResNet-50 via detectron2)
  data/                        # LaRS registration + panoptic dataset mapper + copy-paste
configs/lars/panoptic/         # Base-LaRS.yaml, pansr_R50.yaml, pansr_Swin_L.yaml
tools/remap_weights.py         # original checkpoint -> PanSR keys (+ strict validation)
tools/export_to_hub.py         # build from a checkpoint and save/push to the Hub
train_net.py                   # training / evaluation entry point
predict.py                     # single-image inference example
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Built on MaskDINO, Mask2Former, DINO, Deformable-DETR and FCOS.
