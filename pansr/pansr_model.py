# ------------------------------------------------------------------------
# PanSR meta-architecture. Built on MaskDINO / Mask2Former (Apache-2.0).
# ------------------------------------------------------------------------
# The four PanSR paper contributions live in dedicated, named modules:
#   (1) Object-Centric Proposal (OCP) ........ pansr/modeling/ocp.py
#   (2) Object-centric mask prediction ....... pansr/modeling/object_centric_mask.py
#   (3) Proposal-aware matching .............. pansr/modeling/matching.py (proposal_aware_matching)
#   (4) Mask-conditioned queries ............. pansr/modeling/mask_conditioned_queries.py
# (1), (2) and (4) are wired together inside pansr/modeling/transformer_decoder/pansr_decoder.py;
# (3) is applied in pansr/modeling/disjoint_criterion.py.
# ------------------------------------------------------------------------
from typing import Tuple
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.disjoint_criterion import DisjointSetCriterion
from .modeling.matching import HungarianMatcher
from .utils import box_ops
from torchvision.ops import nms


@META_ARCH_REGISTRY.register()
class PanSR(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.PanSR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.PanSR.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.PanSR.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.PanSR.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.PanSR.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.PanSR.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.PanSR.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.PanSR.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.PanSR.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.PanSR.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.PanSR.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.PanSR.GIOU_WEIGHT  #
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.PanSR.TRAIN_NUM_POINTS,
        )

        weight_dict_base = {"loss_ce": class_weight}
        weight_dict_base.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict_base.update({"loss_bbox":box_weight,"loss_giou":giou_weight})

        weight_dict = {}
        weight_dict.update({k + f'_fg': v for k, v in weight_dict_base.items()})
        weight_dict.update({k + f'_bg': v for k, v in weight_dict_base.items()})

        # two stage is the query selection scheme
        if cfg.MODEL.PanSR.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict_base.items()})
            weight_dict.update(interm_weight_dict)

        # mask-conditioned queries (contribution 4)
        dn = cfg.MODEL.PanSR.MASK_COND_QUERIES.MODE
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict_base.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict_base.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.PanSR.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_fg_{i}": v for k, v in weight_dict_base.items()})
                aux_weight_dict.update({k + f"_bg_{i}": v for k, v in weight_dict_base.items()})
                aux_weight_dict.update({k + f"_dn_{i}": v for k, v in weight_dict_base.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.PanSR.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]

        bg_losses = ["labels", "masks"]

        # building criterion
        criterion = DisjointSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.PanSR.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.PanSR.OVERSAMPLE_RATIO,
            bg_losses=bg_losses,
            num_bg_queries=cfg.MODEL.PanSR.NUM_BG_QUERIES,
            importance_sample_ratio=cfg.MODEL.PanSR.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.PanSR.MASK_COND_QUERIES.MODE,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.PanSR.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.PanSR.TEST.SEMANTIC_ON and cfg.MODEL.PanSR.SEMANTIC_CE_LOSS and not cfg.MODEL.PanSR.TEST.PANOPTIC_ON,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.PanSR.NUM_OBJECT_QUERIES + cfg.MODEL.PanSR.NUM_BG_QUERIES,
            "object_mask_threshold": cfg.MODEL.PanSR.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.PanSR.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.PanSR.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.PanSR.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.PanSR.TEST.PANOPTIC_ON
                or cfg.MODEL.PanSR.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.PanSR.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.PanSR.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.PanSR.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.PanSR.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.PanSR.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.PanSR.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.PanSR.TEST.SEMANTIC_ON and cfg.MODEL.PanSR.SEMANTIC_CE_LOSS and not cfg.MODEL.PanSR.TEST.PANOPTIC_ON
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            targets_fg = [{k: v[tgt['isthing']] for k,v in tgt.items()} for tgt in targets]
            outputs,mask_dict = self.sem_seg_head(features,targets=targets_fg)
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets,mask_dict)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)


            if 'ocp_losses' in outputs:
                ocp_losses = {'ocp_' + k: v for k,v in outputs['ocp_losses'].items()}
                losses.update(ocp_losses)

            return losses
        else:
            outputs, _ = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            ocp_outputs = outputs["ocp_aux"]

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                if self.panoptic_on or self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r

                processed_results[-1]["ocp_aux"] = ocp_outputs

            return processed_results

    def _get_levels(self, boxes: torch.Tensor):
        """Get target level based on box size"""
        buckets = [32, 64, 128, 256]
        buckets = torch.tensor(buckets, device=boxes.device)
        boxes = box_ops.box_xyxy_to_cxcywh(boxes)
        diags = boxes[:, 2:].norm(p=2, dim=1)

        levels = buckets.shape[0] - torch.bucketize(diags, buckets)

        return levels

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "level": self._get_levels(targets_per_image.gt_boxes.tensor),
                    "isthing": targets_per_image.gt_isthing,
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy,
                    "cls_weights": targets_per_image.gt_cls_weight
                }
            )
        return new_targets

    def prepare_targets_detr(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "levels": self._get_levels(targets_per_image.gt_boxes.tensor),
                    "isthing": targets_per_image.gt_isthing,
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy,
                    "cls_weights": targets_per_image.gt_cls_weight
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred, box_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.


        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)

        # NMS for foreground classes
        thing_cls = torch.tensor(list(self.metadata.thing_dataset_id_to_contiguous_id.values()), device=labels.device)
        isthing = torch.isin(labels, thing_cls)
        nms_mask = torch.isin(torch.arange(box_pred.shape[0], device=box_pred.device), nms(box_pred, scores, 0.75))
        nms_mask = nms_mask | ~isthing

        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold) & nms_mask
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        isthing = isthing[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            bg_prob_masks = cur_prob_masks * (~isthing)[:, None, None]
            # BG (stuff) argmax
            bg_mask_ids = bg_prob_masks.argmax(0)

            # FG (things) argmax
            fg_prob_masks = cur_prob_masks * isthing[:, None, None]
            fg_mask_ids = fg_prob_masks.argmax(0)

            m = fg_prob_masks.max(0)[0] > prob

            # Merge (overlay fg on top of bg)
            cur_mask_ids = ~m * bg_mask_ids + m * fg_mask_ids
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):

        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)

        # NMS for foreground classes
        thing_cls = torch.tensor(list(self.metadata.thing_dataset_id_to_contiguous_id.values()), device=labels.device)
        isthing = torch.isin(labels, thing_cls)
        nms_mask = torch.isin(torch.arange(box_pred.shape[0], device=box_pred.device), nms(box_pred, scores, 0.75))
        nms_mask = nms_mask & isthing

        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold) & nms_mask
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        isthing = isthing[keep]
        cur_boxes = box_pred[keep]

        image_size = mask_pred.shape[-2:]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (cur_masks > 0).float()
        result.pred_boxes = Boxes(cur_boxes)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (cur_masks.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = cur_scores * mask_scores_per_image
        result.pred_classes = cur_classes
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


