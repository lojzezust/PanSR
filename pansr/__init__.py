# ------------------------------------------------------------------------
# PanSR: Panoptic Segmentation of maritime scenes (release of "AnchorFormer").
# Built on MaskDINO (Apache-2.0). See README for the four paper contributions.
# ------------------------------------------------------------------------
from . import data       # register all datasets (LaRS)
from . import modeling   # register backbones / pixel decoder / head

# config
from .config import add_pansr_config

# dataset mapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import (
    COCOPanopticNewBaselineDatasetMapper,
)

# model (meta architecture registered with detectron2 as "PanSR")
from .pansr_model import PanSR

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator

# util
from .utils import box_ops, misc, utils

# HuggingFace Hub wrapper
from .hub import PanSRHF
