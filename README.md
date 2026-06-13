# PanSR: An Object-Centric Mask Transformer for Panoptic Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2412.10589-b31b1b.svg)](https://arxiv.org/abs/2412.10589)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTITS.2026.3697512-blue.svg)](https://doi.org/10.1109/TITS.2026.3697512)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/collections/lojzezust/pansr)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-blue.svg)](#cite)

PanSR is a transformer-based panoptic segmentation model for maritime obstacle scenes,
trained and evaluated on the [LaRS](https://lojzezust.github.io/lars-dataset/) dataset. It is
built on [MaskDINO](https://github.com/IDEA-Research/MaskDINO) / Mask2Former and adds four
contributions that make it well suited to thin, small, and densely packed maritime obstacles.

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

## PanSR inference

**Inference with HuggingFace model (easiest)**
```bash
python predict.py \
    --hf-model lojzezust/pansr-lars-resnet50 \
    --input assets/sample.jpg \
    --output out.png --vis
```

This writes a LaRS-format panoptic mask (`out.png`) and, with `--vis`, a human-friendly overlay
(`out_vis.png`). `--input` also accepts a glob (e.g. `'images/*.jpg'`) with `--output <dir>`.

**Inference using local weights**

```bash
python predict.py \
    --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    --weights weights/pansr_lars_swin_l.pth \
    --input assets/sample.jpg \
    --output out.png --vis
```

## Pretrained weights

| Backbone | Weights (md5) | HuggingFace | PQ (LaRS) |
|----------|---------------|-------------|-----------|
| ResNet-50 | [link](https://box.vicos.si/pansr/pansr_lars_resnet50.pth) (`c2554a5803c217c453bad78205ea4a3f`) | [lojze/pansr-lars-r50](https://huggingface.co/lojze/pansr-lars-r50) | 54.2 |
| Swin-L | [link](https://box.vicos.si/pansr/pansr_lars_swin_l.pth) (`e3948f8084d1bc33a180dce7a4122bf7`) | [lojze/pansr-lars-swin-l](https://huggingface.co/lojze/pansr-lars-swin-l) | 57.3|


## Using PanSR in code

Inference uses detectron2's `DefaultPredictor`, which handles preprocessing (resize +
normalization) and takes a single **BGR** `uint8` image. Building the model requires the LaRS
metadata, so make sure `LARS_ROOT` is set (see [Installation](#installation)).

### Load the model

**From the HuggingFace Hub** (downloads config + weights, see [`pansr/hub.py`](pansr/hub.py)):

```python
import torch
from detectron2.engine.defaults import DefaultPredictor
from pansr.hub import PanSRHF

hf = PanSRHF.from_pretrained("lojzezust/pansr-lars-resnet50")   # or pansr-lars-swin-l
cfg = hf.cfg
predictor = DefaultPredictor(cfg)
predictor.model = hf.model.to(cfg.MODEL.DEVICE).eval()
```

**From a local config + weights:**

```python
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from pansr import add_pansr_config

cfg = get_cfg()
add_deeplab_config(cfg)
add_pansr_config(cfg)
cfg.merge_from_file("configs/lars/panoptic/pansr_Swin_L.yaml")
cfg.MODEL.WEIGHTS = "weights/pansr_lars_swin_l.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.freeze()

predictor = DefaultPredictor(cfg)
```

### Run on an image

```python
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image

img = read_image("assets/sample.jpg", format="BGR")   # HxWx3 uint8, BGR
with torch.no_grad():
    predictions = predictor(img)

# Panoptic result: a HxW id map + per-segment metadata
panoptic_seg, segments_info = predictions["panoptic_seg"]

# Optional: draw a colored overlay
from detectron2.utils.visualizer import Visualizer
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
vis = Visualizer(img[:, :, ::-1], metadata)
overlay = vis.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info)
overlay.save("out_vis.jpg")
```

See [`predict.py`](predict.py) for a complete script (including encoding the result as a
LaRS-format panoptic PNG) and the CLI usage shown under [Inference](#inference).

## Training

```bash
export LARS_ROOT=/path/to/LaRS

# ResNet-50
python train_net.py --num-gpus 4 --config-file configs/lars/panoptic/pansr_R50.yaml

# Swin-L (set MODEL.WEIGHTS to the ImageNet-22k Swin-L backbone for from-scratch training)
python train_net.py --num-gpus 4 --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    MODEL.WEIGHTS /path/to/swin_large_patch4_window12_384_22k.pkl
```

**Note:** Copy-paste augmentation expects pre-extracted objects under `$LARS_ROOT/train/objects_v2`
(configurable via `INPUT.COPY_PASTE.OBJECTS_DIR`). The Swin-L ImageNet-22k backbone weights
(`swin_large_patch4_window12_384_22k.pkl`) are only needed for from-scratch training.

## Contributions (and where they live in the code)

The model entry point is [`pansr/pansr_model.py`](pansr/pansr_model.py). Each contribution
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


## License

Apache-2.0 (see [LICENSE](LICENSE)). Built on MaskDINO, Mask2Former, DINO, Deformable-DETR and FCOS.

## Cite

If you use PanSR in your research, please cite our work:

```bibtex
@article{Zust2026PanSR,
  author={Žust, Lojze and Kristan, Matej},
  journal={T-ITS}, 
  title={PanSR: An Object-Centric Mask Transformer for Maritime Panoptic Segmentation}, 
  year={2026},
  doi={10.1109/TITS.2026.3697512}
}
```
