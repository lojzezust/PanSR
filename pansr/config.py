# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Modified from MaskDINO (https://github.com/IDEA-Research/MaskDINO)
# ------------------------------------------------------------------------
# Copyright (c) IDEA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN


def add_pansr_config(cfg):
    """
    Add config for PanSR.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_panoptic_lsj"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # Copy-paste augmentation
    cfg.INPUT.COPY_PASTE = CN()
    cfg.INPUT.COPY_PASTE.ENABLED = False
    cfg.INPUT.COPY_PASTE.OBJECTS_DIR = None
    cfg.INPUT.COPY_PASTE.NUM_RANGES = [
        [0, 1],
        [1, 6],
        [6,11]
    ]
    cfg.INPUT.COPY_PASTE.SCALE_RANGES = [
        [20, 32],
        [32, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, 1024]]
    cfg.INPUT.COPY_PASTE.MAX_SCALE = 1.2
    cfg.INPUT.COPY_PASTE.ROTATION_RANGE = [-30, 30]
    cfg.INPUT.COPY_PASTE.WATER_ID = 3

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # PanSR model config
    cfg.MODEL.PanSR = CN()
    cfg.MODEL.PanSR.LEARN_TGT = False

    # loss
    cfg.MODEL.PanSR.PANO_BOX_LOSS = False
    cfg.MODEL.PanSR.SEMANTIC_CE_LOSS = False
    cfg.MODEL.PanSR.DEEP_SUPERVISION = True
    cfg.MODEL.PanSR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.PanSR.CLASS_WEIGHT = 4.0
    cfg.MODEL.PanSR.DICE_WEIGHT = 5.0
    cfg.MODEL.PanSR.MASK_WEIGHT = 5.0
    cfg.MODEL.PanSR.BOX_WEIGHT = 5.
    cfg.MODEL.PanSR.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.PanSR.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.PanSR.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.PanSR.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.PanSR.COST_BOX_WEIGHT = 5.
    cfg.MODEL.PanSR.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.PanSR.NHEADS = 8
    cfg.MODEL.PanSR.DROPOUT = 0.1
    cfg.MODEL.PanSR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.PanSR.ENC_LAYERS = 0
    cfg.MODEL.PanSR.DEC_LAYERS = 6
    cfg.MODEL.PanSR.INITIAL_PRED = True
    cfg.MODEL.PanSR.PRE_NORM = False
    cfg.MODEL.PanSR.BOX_LOSS = True
    cfg.MODEL.PanSR.HIDDEN_DIM = 256
    cfg.MODEL.PanSR.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.PanSR.NUM_BG_QUERIES = 50

    cfg.MODEL.PanSR.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.PanSR.TWO_STAGE = True  # use OCP proposals to initialize object queries
    cfg.MODEL.PanSR.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']

    cfg.MODEL.PanSR.EVAL_FLAG = 1

    # Contribution (4): Mask-conditioned queries (training-time auxiliary queries).
    # MODE: 'no' disables them; 'standard' uses labels+boxes; 'seg' adds mask supervision.
    cfg.MODEL.PanSR.MASK_COND_QUERIES = CN()
    cfg.MODEL.PanSR.MASK_COND_QUERIES.MODE = "seg"
    cfg.MODEL.PanSR.MASK_COND_QUERIES.NOISE_SCALE = 0.4
    cfg.MODEL.PanSR.MASK_COND_QUERIES.NUM = 100
    cfg.MODEL.PanSR.MASK_COND_QUERIES.MAX_GROUP_SIZE = 25

    # Contribution (2): Object-centric mask prediction (limit FG masks to predicted boxes).
    cfg.MODEL.PanSR.OBJECT_CENTRIC_MASK = CN()
    cfg.MODEL.PanSR.OBJECT_CENTRIC_MASK.ENABLED = True
    cfg.MODEL.PanSR.OBJECT_CENTRIC_MASK.DILATE_AMOUNT = 0.1  # Dilate the bounding box a bit
    cfg.MODEL.PanSR.OBJECT_CENTRIC_MASK.MIN_DILATE_PX = 2    # Minimum dilation [px in output stride (4)]


    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # PanSR inference config
    cfg.MODEL.PanSR.TEST = CN()
    cfg.MODEL.PanSR.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.PanSR.TEST.SEMANTIC_ON = True
    cfg.MODEL.PanSR.TEST.INSTANCE_ON = False
    cfg.MODEL.PanSR.TEST.PANOPTIC_ON = False
    cfg.MODEL.PanSR.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.PanSR.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.PanSR.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.PanSR.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.PanSR.TEST.PANO_TEMPERATURE = 0.06
    # cfg.MODEL.PanSR.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.PanSR.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "PanSREncoder"

    # transformer module
    cfg.MODEL.PanSR.TRANSFORMER_DECODER_NAME = "PanSRDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.PanSR.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.PanSR.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.PanSR.IMPORTANCE_SAMPLE_RATIO = 0.75


    # Swin default config
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
