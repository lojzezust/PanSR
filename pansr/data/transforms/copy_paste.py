# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
import copy
import logging
from collections import namedtuple

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from panopticapi.utils import rgb2id, id2rgb

class CopyPaste:
    """
    Randomly copy-paste objects from a pool.
    """

    def __init__(self, objects_dir, *,
                 num_ranges=[(0,5)],
                 scale_ranges=[(20,512)],
                 max_scale=1.2,
                 rotation_range=(-30,30),
                 water_id=3):
        """
        Args:
            objects_dir (str): path to directory with objects
            num_range (tuple): range of number of objects to paste
            scale_range (tuple): range of scales to paste objects
            min_size (int): minimal size (diagonal px) of objects to paste
            rotation_range (tuple): range of rotations to paste objects
            water_id (int): id of the water class, used to determine the placing of objects
        """
        super().__init__()

        self.objects = self._get_objects(objects_dir)

        self.num_ranges = num_ranges
        self.scale_ranges = scale_ranges
        self.max_scale = max_scale
        self.rotation_range = rotation_range

        self.water_id = water_id

    @staticmethod
    def _get_objects(objects_dir):
        objects = []
        for p in Path(objects_dir).glob('**/*.png'):
            objects.append({
                'path': str(p),
                'filename': p.name,
                'class': int(p.parent.name),
            })
        return objects

    @staticmethod
    def _get_random_id(labels):
        while True:
            id = np.random.randint(0, 256**3)
            if id not in labels:
                return id

    @staticmethod
    def _add_patch_at(image, patch, mask, loc):
        """Adds patch to image with center at loc (x,y). Patch is masked by mask."""

        x,y = loc
        h,w = patch.shape[:2]
        x0,y0 = x - w//2, y - h//2
        x1,y1 = x0 + w, y0 + h

        # Crop patch
        if x0 < 0:
            patch = patch[:, -x0:]
            mask = mask[:, -x0:]
            x0 = 0
        if y0 < 0:
            patch = patch[-y0:, :]
            mask = mask[-y0:, :]
            y0 = 0
        if x1 > image.shape[1]:
            patch = patch[:, :-(x1 - image.shape[1])]
            mask = mask[:, :-(x1 - image.shape[1])]
            x1 = image.shape[1]
        if y1 > image.shape[0]:
            patch = patch[:-(y1 - image.shape[0]), :]
            mask = mask[:-(y1 - image.shape[0]), :]
            y1 = image.shape[0]

        # Paste patch
        image[y0:y1, x0:x1] = patch * mask + image[y0:y1, x0:x1] * ~mask
        bbox = [x0,y0,x1-x0,y1-y0]
        area = np.sum(mask)

        return image, bbox, area

    def __call__(self, image, pan_seg_gt, segments_info):
        pan_seg_id = rgb2id(pan_seg_gt)
        water_mask = pan_seg_id == self.water_id
        # If no water: skip
        if not np.any(water_mask):
            return image, pan_seg_gt, segments_info

        labels = set(np.unique(pan_seg_id))
        water_locs = np.stack(np.where(water_mask), axis=-1)

        # 1. determine number
        rng = np.random.default_rng()
        num_range = rng.choice(self.num_ranges, axis=0)
        num = np.random.randint(*num_range)

        objs = []
        for i in range(num):
            if len(water_locs) == 0:
                break

            # 2. Determine object, location, size, rotation, flip
            obj = np.random.choice(self.objects)

            path=obj['path']
            obj_img = Image.open(path).convert('RGBA')
            class_id=obj['class']
            loc_i = np.random.randint(water_locs.shape[0])
            cy,cx = water_locs[loc_i]
            scale_range=rng.choice(self.scale_ranges, axis=0)
            new_diag=np.random.uniform(*scale_range)
            rotation=np.random.uniform(*self.rotation_range)
            flip=np.random.rand() > 0.5

            # Limit minimum size of pasted object
            scale_f = new_diag / np.sqrt(obj_img.width**2 + obj_img.height**2)
            scale_f = min(scale_f, self.max_scale)

            # Flip, scale, rotate
            if flip:
                obj_img = obj_img.transpose(Image.FLIP_LEFT_RIGHT)
            obj_img = obj_img.resize((round(obj_img.width * scale_f), round(obj_img.height * scale_f)), Image.BILINEAR)
            obj_img = obj_img.rotate(rotation, expand=True)
            obj_img = np.array(obj_img)

            obj_mask = obj_img[:,:,3] > 0
            obj_img = obj_img[:,:,:3]

            # 3. Paste object to image and pan_seg_gt
            image, bbox, area = self._add_patch_at(image, obj_img, obj_mask[...,None], (cx,cy))

            obj_id = self._get_random_id(labels)
            pan_seg_id, _, _ = self._add_patch_at(pan_seg_id, obj_mask * obj_id, obj_mask, (cx,cy))

            water_mask = pan_seg_id == self.water_id
            water_locs = np.stack(np.where(water_mask), axis=-1)

            segments_info.append({
                'id': obj_id,
                'category_id': class_id,
                'isthing': 1,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'cls_weight': 0.
            })



        return image, id2rgb(pan_seg_id), segments_info
