# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
import os
import os.path as osp
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

# Root of the LaRS dataset. Set the LARS_ROOT environment variable to point at it.
LARS_ROOT = os.environ.get("LARS_ROOT", "datasets/LaRS")

# Define the dataset class
def get_lars_dataset(root_dir, split, annotation_file):
    # Load the annotations
    with open(osp.join(root_dir, split, annotation_file), 'r') as json_file:
        dataset_annotations = json.load(json_file)

    id2img = {img['id']: img for img in dataset_annotations["images"]}
    id2category = {cat['id']: cat for cat in dataset_annotations["categories"]}

    dataset_dicts = []
    for idx, ann_info in enumerate(dataset_annotations["annotations"]):
        record = {}
        image_info = id2img[ann_info['image_id']]

        # Set image file path
        filename = image_info["file_name"]
        record["file_name"] = os.path.join(root_dir, split, 'images', filename)

        # Set image ID
        record["image_id"] = ann_info['image_id']

        # Set image size
        record["height"] = image_info["height"]
        record["width"] = image_info["width"]

        # Set panoptic annotations
        record["pan_seg_file_name"] = osp.join(root_dir, split, 'panoptic_masks', ann_info['file_name'])
        record["segments_info"] = ann_info['segments_info']

        # Add thing/stuff information
        for seg_info in record["segments_info"]:
            seg_info["isthing"] = id2category[seg_info["category_id"]]["isthing"]
            seg_info["cls_weight"] = 1. # Used to disable classification loss for specific objects (e.g. copy-paste augmentation)

        dataset_dicts.append(record)

    return dataset_dicts


def get_metadata(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    categories = data['categories']

    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = ['void'] + [k["name"] for k in categories if k["isthing"] == 1]
    thing_colors = [[0,0,0]] + [k["color"] for k in categories if k["isthing"] == 1]
    stuff_classes = ['void'] + [k["name"] for k in categories]
    stuff_colors = [[0,0,0]] + [k["color"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i + 1
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    meta["category_id_map"] = {
        1: 11, # Boat/ship
        2: 12, # Row boat
        3: 13, # Paddle board
        4: 14, # Buoy
        5: 15, # Swimmer
        6: 16, # Animal
        7: 17, # Float
        8: 19, # Other
        9: 1, # Static Obstacle
        10: 3, # Water
        11: 5, # Sky
        12: 0 # VOID
    }

    return meta

def register_lars_dataset(root_dir, split, annotations_file='mmdet_annotations.json', name=None):
    if name is None:
        name = f'lars_{split}_panoptic'

    DatasetCatalog.register(name, lambda: get_lars_dataset(root_dir, split, annotations_file))

    metadata = get_metadata(osp.join(root_dir, split, annotations_file))

    MetadataCatalog.get(name).set(
        panoptic_root=osp.join(root_dir, split, 'panoptic_masks'),
        image_root=osp.join(root_dir, split, 'images'),
        panoptic_json=osp.join(root_dir, split, annotations_file),
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=256 * 256,
        **metadata,
    )

def _maybe_register(root_dir, split, annotations_file='mmdet_annotations.json', name=None):
    """Register a split only if its annotation file exists.

    Metadata is built from the annotation JSON, so a split can only be registered when the
    dataset is present (set LARS_ROOT). Missing splits are skipped with a warning rather than
    failing ``import pansr`` — handy for inference-only / HuggingFace use on machines without LaRS.
    """
    ann_path = osp.join(root_dir, split, annotations_file)
    if not osp.isfile(ann_path):
        import warnings
        warnings.warn(
            f"LaRS annotation file not found at {ann_path}; skipping registration of "
            f"'{name or f'lars_{split}_panoptic'}'. Set LARS_ROOT to enable it."
        )
        return
    register_lars_dataset(root_dir, split, annotations_file=annotations_file, name=name)


# Register the LaRS splits in the dataset catalog (skipped gracefully if the data is absent).
_maybe_register(LARS_ROOT, 'val')
_maybe_register(LARS_ROOT, 'train')
_maybe_register(LARS_ROOT, 'train', annotations_file='overfit_mmdet_annotations.json', name='lars_train_overfit_panoptic')
