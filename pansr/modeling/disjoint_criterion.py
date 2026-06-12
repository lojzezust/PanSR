"""Disjoint criterion for stuff and thing classes."""

import torch
import torch.nn.functional as F
from torch import nn
from detectron2.utils.comm import get_world_size
import numpy as np

from .criterion import SetCriterion
from .matching import proposal_aware_matching  # PanSR contribution (3)
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from pansr.utils.box_ops import box_iou, box_cxcywh_to_xyxy


class DisjointSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, bg_losses, num_bg_queries, importance_sample_ratio, **args):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio, **args)

        self.bg_losses = bg_losses
        self.num_bg_queries = num_bg_queries

    def split_predictions(self, outputs):
        keys = ["pred_logits", "pred_masks", "pred_boxes"]

        fg_outputs = {k: v[:, self.num_bg_queries:] for k, v in outputs.items() if k in keys}
        bg_outputs = {k: v[:, :self.num_bg_queries] for k, v in outputs.items() if k in keys}

        return fg_outputs, bg_outputs

    def forward(self, outputs, targets, mask_dict=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """


        # Separate FG and BG targets
        targets_fg = [{k: v[tgt['isthing']] for k,v in tgt.items()} for tgt in targets]
        targets_bg = [{k: v[~tgt['isthing']] for k,v in tgt.items()} for tgt in targets]

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn != "no" and mask_dict is not None:
            dn_outputs = mask_dict['dn_outputs']
            indices_dn = mask_dict['indices_dn']
            num_dn_tgts = sum(len(tgt_i) for _,tgt_i in indices_dn)

        outputs_fg, outputs_bg = self.split_predictions(outputs)
        indices_fg, iou_matrices = self.matcher(outputs_fg, targets_fg, return_iou=True)
        indices_fg = proposal_aware_matching(indices_fg, outputs_fg, targets_fg, iou_matrices)
        indices_bg = self.matcher(outputs_bg, targets_bg, cost=["cls", "mask"])

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks_fg = sum(len(t["labels"]) for t in targets_fg)
        num_masks_fg = torch.as_tensor(
            [num_masks_fg], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks_bg = sum(len(t["labels"]) for t in targets_bg)
        num_masks_bg = torch.as_tensor(
            [num_masks_bg], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_fg)
            torch.distributed.all_reduce(num_masks_bg)
        num_masks_fg = torch.clamp(num_masks_fg / get_world_size(), min=1).item()
        num_masks_bg = torch.clamp(num_masks_bg / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # FG
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs_fg, targets_fg, indices_fg, num_masks_fg)
            l_dict = {k + f'_fg': v for k, v in l_dict.items()}
            losses.update(l_dict)
        for loss in self.bg_losses:
            l_dict = self.get_loss(loss, outputs_bg, targets_bg, indices_bg, num_masks_bg)
            l_dict = {k + f'_bg': v for k, v in l_dict.items()}
            losses.update(l_dict)

        if self.dn != "no" and mask_dict is not None:
            l_dict={}
            for loss in self.dn_losses:
                l_dict.update(self.get_loss(loss, dn_outputs, targets_fg, indices_dn, num_dn_tgts))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if self.dn == "seg":
                l_dict['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs_fg, aux_outputs_bg = self.split_predictions(aux_outputs)
                indices_fg, iou_matrices = self.matcher(aux_outputs_fg, targets_fg, return_iou=True)
                indices_fg = proposal_aware_matching(indices_fg, aux_outputs_fg, targets_fg, iou_matrices)
                indices_bg = self.matcher(aux_outputs_bg, targets_bg, cost=["cls", "mask"])

                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs_fg, targets_fg, indices_fg, num_masks_fg)
                    l_dict = {k + f"_fg_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                for loss in self.bg_losses:
                    l_dict = self.get_loss(loss, aux_outputs_bg, targets_bg, indices_bg, num_masks_bg)
                    l_dict = {k + f"_bg_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None:
                        out_=dn_outputs['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(
                                self.get_loss(loss, out_, targets_fg, indices_dn, num_dn_tgts))
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg":
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs_fg = outputs['interm_outputs']
            indices_fg, iou_matrices = self.matcher(interm_outputs_fg, targets_fg, return_iou=True)
            indices_fg = proposal_aware_matching(indices_fg, interm_outputs_fg, targets_fg, iou_matrices)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs_fg, targets_fg, indices_fg, num_masks_fg)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
