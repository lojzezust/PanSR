# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Modified from MaskDINO (https://github.com/IDEA-Research/MaskDINO)
# ------------------------------------------------------------------------
# Copyright (c) IDEA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from pansr.utils.box_ops import box_iou, box_cxcywh_to_xyxy


# =====================================================================================
# PanSR contribution (3): Proposal-aware matching
# -------------------------------------------------------------------------------------
# Refines the Hungarian assignment between object proposals and ground-truth objects by
# (1) dropping matched pairs whose box IoU is too low (false negatives for that query),
# and (2) recovering unmatched proposals ("false positives") that nevertheless overlap a
# GT object strongly enough, re-assigning them to that GT. This makes supervision
# consistent with the object-centric proposals produced by the OCP module.
# =====================================================================================
def proposal_aware_matching(indices, outputs, targets, iou_matrices, min_iou=0.25, min_fp_iou=0.75):
    """Refine Hungarian ``indices`` using proposal/GT box IoU.

    Args:
        indices: list (len = batch) of ``(proposal_idx, target_idx)`` tuples from the matcher.
        outputs: dict with ``pred_boxes`` of shape ``(B, Np, 4)`` (cxcywh, normalized).
        targets: list of per-image target dicts with ``boxes`` of shape ``(Nt, 4)``.
        iou_matrices: list (len = batch) of ``(Np, Nt)`` proposal-vs-GT IoU matrices.
        min_iou: matched pairs with IoU below this are removed (false-negative removal).
        min_fp_iou: unmatched proposals with max-GT IoU above this are recovered (false-positive recovery).

    Returns:
        list of refined ``(proposal_idx, target_idx)`` tuples.
    """
    new_indices = []
    for bi in range(len(indices)):
        idx_pred, idx_tgt = indices[bi]
        if len(idx_pred) == 0:
            new_indices.append((idx_pred, idx_tgt))
            continue

        iou_matrix = iou_matrices[bi]

        # Step 1: Remove bad matches (low-IoU pairs).
        matches_iou = iou_matrix[idx_pred, idx_tgt]
        valid_matches = (matches_iou > min_iou).cpu()
        idx_pred = idx_pred[valid_matches]
        idx_tgt = idx_tgt[valid_matches]

        # Step 2: Recover false positives that strongly overlap a GT object.
        n_preds = outputs['pred_boxes'][bi].shape[0]
        fp_idx = torch.tensor(np.nonzero(np.logical_not(np.in1d(np.arange(n_preds), idx_pred)))[0])
        fp_iou, fp_tgt_idx = iou_matrix[fp_idx, :].max(dim=1)
        valid_fps = fp_iou > min_fp_iou

        idx_pred = torch.cat([idx_pred, fp_idx[valid_fps.cpu()]])
        idx_tgt = torch.cat([idx_tgt, fp_tgt_idx[valid_fps].cpu()])

        new_indices.append((idx_pred, idx_tgt))

    return new_indices


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 cost_box: float = 0, cost_giou: float = 0, panoptic_on: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou

        self.panoptic_on = panoptic_on

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost=["cls", "box", "mask"]):
        """More memory-friendly matching. Change cost to compute only certain loss in matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        iou_matrices = []

        # Iterate through batch size
        for b in range(bs):
            out_bbox = outputs["pred_boxes"][b]
            if 'box' in cost:
                tgt_bbox=targets[b]["boxes"]
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = torch.tensor(0).to(out_bbox)
                iou = torch.tensor(0).to(out_bbox)

            cost_iou = -iou
            iou_matrices.append(iou)

            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]
            # focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-7).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-7).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]
            if 'mask' in cost:
                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # If there's no annotations
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        # Compute the dice loss betwen masks
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            else:
                cost_mask = torch.tensor(0).to(out_bbox)
                cost_dice = torch.tensor(0).to(out_bbox)

            # Final cost matrix
            if self.panoptic_on:
                # TODO: Why fixed to 80? Does this code run at any time?
                isthing = tgt_ids<80
                cost_bbox[:, ~isthing] = cost_bbox[:, isthing].mean()
                cost_iou[:, ~isthing] = cost_iou[:, isthing].mean()
                cost_bbox[cost_bbox.isnan()] = 0.0
                cost_iou[cost_iou.isnan()] = 0.0

            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_box*cost_bbox
                + self.cost_giou*cost_iou
            )
            C = C.reshape(num_queries, -1).cpu()

            try:
                indices.append(linear_sum_assignment(C))
            except ValueError as e:
                print('Something wrong happened')
                raise e

        matches = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

        return matches, iou_matrices

    @torch.no_grad()
    def forward(self, outputs, targets, cost=["cls", "box", "mask"], return_iou=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        matches, iou_matrices = self.memory_efficient_forward(outputs, targets, cost)

        if return_iou:
            return matches, iou_matrices

        return matches

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
