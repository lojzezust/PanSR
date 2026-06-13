# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""Extracts 'thing' objects from the LaRS train split and saves them as RGBA cutouts.

These cutouts are consumed by the copy-paste augmentation during training
(``INPUT.COPY_PASTE.OBJECTS_DIR``). See the README's Training section.

    python -m pansr.data.utils.extract_objects \
        --dataset-dir $LARS_ROOT/train \
        --output-dir $LARS_ROOT/train/objects_v2
"""

import argparse
import os
import os.path as osp
import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


ANN_FILE = 'mmdet_annotations.json'
IMG_DIR = 'images'
MASK_DIR = 'panoptic_masks'

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

def mask2bbox(mask):
    ys,xs = np.where(mask)
    x0,y0 = xs.min(), ys.min()
    x1,y1 = xs.max() + 1, ys.max() + 1

    return x0,y0,x1,y1


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-dir', required=True,
                    help="LaRS train split dir (contains images/, panoptic_masks/, and the annotations file)")
    ap.add_argument('--output-dir', required=True,
                    help="Where to write the per-category object cutouts (e.g. $LARS_ROOT/train/objects_v2)")
    ap.add_argument('--ann-file', default=ANN_FILE, help=f"Annotations file name (default: {ANN_FILE})")
    ap.add_argument('--size-threshold', type=int, default=256,
                    help="Skip objects smaller than this many pixels (default: 256)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(osp.join(args.dataset_dir, args.ann_file), 'r') as f:
        annotations = json.load(f)

    id2img = {img['id']: img for img in annotations['images']}
    id2cat = {cat['id']: cat for cat in annotations['categories']}

    for annotation in tqdm(annotations['annotations']):
        filename = id2img[annotation['image_id']]['file_name']
        name = osp.splitext(filename)[0]
        image_path = os.path.join(args.dataset_dir, IMG_DIR, filename)
        image = np.array(Image.open(image_path))

        mask_filename = annotation['file_name']
        mask_path = os.path.join(args.dataset_dir, MASK_DIR, mask_filename)
        mask = rgb2id(np.array(Image.open(mask_path)))

        for segment_info in annotation['segments_info']:
            cat = id2cat[segment_info['category_id']]
            if cat['isthing'] == 0 or segment_info['iscrowd'] == 1:
                continue

            obj_mask = segment_info['id'] == mask

            if obj_mask.sum() < args.size_threshold:
                continue

            # Skip if object touches the border
            if obj_mask[0, :].sum() > 0 or obj_mask[-1, :].sum() > 0 or obj_mask[:, 0].sum() > 0 or obj_mask[:, -1].sum() > 0:
                continue

            x0,y0,x1,y1 = mask2bbox(obj_mask)

            obj_image = image[y0:y1, x0:x1]
            obj_mask = obj_mask[y0:y1, x0:x1][..., None]

            img_a = np.concatenate([obj_image * obj_mask, obj_mask * 255], axis=2).astype(np.uint8)

            out_dir = os.path.join(args.output_dir, '%d' % segment_info['category_id'])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            out_path = os.path.join(out_dir, f'{name}_{segment_info["id"]}.png')
            Image.fromarray(img_a, 'RGBA').save(out_path)


if __name__ == '__main__':
    main()
