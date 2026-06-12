#!/usr/bin/env python3
"""Run PanSR panoptic segmentation on a single image (or a glob of images).

Examples:
    # From a local (remapped) checkpoint:
    python predict.py --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
        --weights weights/pansr_lars_swin_l.pth --input assets/sample.jpg --output out.png

    # From the HuggingFace Hub (downloads weights + config):
    python predict.py --hf-model lojze/pansr-lars-swin-l --input assets/sample.jpg --output out.png
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer

from pansr import add_pansr_config


def build_cfg(config_file, weights, opts):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_pansr_config(cfg)
    cfg.merge_from_file(config_file)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    cfg.merge_from_list(opts or [])
    cfg.freeze()
    return cfg


def panoptic_to_lars_rgb(panoptic_seg, segments_info, metadata):
    """Encode a panoptic result as a LaRS-style RGB id mask (matches the original inference.py)."""
    from panopticapi.utils import id2rgb, rgb2id

    pan = panoptic_seg.cpu().numpy()
    out = np.full_like(pan, fill_value=255, dtype=np.int64)
    for ann in segments_info:
        cat_id = metadata.category_id_map[ann["category_id"]]
        if ann["isthing"]:
            new_id = rgb2id([cat_id, ann["id"] % 256, ann["id"] // 256])
        else:
            new_id = rgb2id([cat_id, 0, 0])
        out[pan == ann["id"]] = new_id
    return id2rgb(out).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", default="configs/lars/panoptic/pansr_Swin_L.yaml")
    ap.add_argument("--weights", default=None, help="Path to a (remapped) PanSR checkpoint")
    ap.add_argument("--hf-model", default=None, help="HuggingFace Hub repo id to load weights+config from")
    ap.add_argument("--input", nargs="+", required=True, help="Image path(s) or a glob pattern")
    ap.add_argument("--output", default=None, help="Output file (single image) or directory")
    ap.add_argument("--vis", action="store_true", help="Also save a human-friendly colored overlay (*_vis.png)")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Extra cfg KEY VALUE overrides")
    args = ap.parse_args()

    if args.hf_model:
        from pansr.hub import PanSRHF
        hf = PanSRHF.from_pretrained(args.hf_model)
        cfg = hf.cfg
        predictor = DefaultPredictor(cfg)
        predictor.model = hf.model.to(cfg.MODEL.DEVICE).eval()
    else:
        assert args.weights, "Provide --weights or --hf-model"
        cfg = build_cfg(args.config_file, args.weights, args.opts)
        predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    inputs = args.input
    if len(inputs) == 1:
        inputs = glob.glob(os.path.expanduser(inputs[0]))
    assert inputs, "No input images found"

    for path in tqdm(inputs, "Predicting & saving"):
        img = read_image(path, format="BGR")
        with torch.no_grad():
            predictions = predictor(img)
        panoptic_seg, segments_info = predictions["panoptic_seg"]

        rgb = panoptic_to_lars_rgb(panoptic_seg, segments_info, metadata)

        if args.output and len(inputs) > 1 or (args.output is None):
            out_dir = args.output or "."
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
        else:
            out_path = args.output
        Image.fromarray(rgb).save(out_path)

        if args.vis:
            vis = Visualizer(img[:, :, ::-1], metadata)
            vis_out = vis.draw_panoptic_seg(panoptic_seg.to("cpu"), segments_info)
            vis_path = os.path.splitext(out_path)[0] + "_vis.jpg"
            vis_out.save(vis_path)

if __name__ == "__main__":
    main()
