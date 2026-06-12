# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""PanSR contribution (1): Object-Centric Proposal (OCP) module and head.

A lightweight, FCOS-style dense detection branch that runs on the FPN levels of the pixel
decoder and produces object-centric proposals (objectness + center + box) used to initialize
the decoder's object queries (two-stage query selection). ``OCPHead`` is the per-level
prediction head; ``OCPModule`` orchestrates the multi-level heads, generates the top-K
proposals, and computes the OCP training losses.

Originally an FCOS-style head; adapted by lojzezust from https://github.com/tianzhi0549/FCOS
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.ops import sigmoid_focal_loss

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class OCPHead(torch.nn.Module):
    def __init__(self, in_channels, num_convs, stride, norm_reg_targets=False, prior_prob=0.01, use_dcn_in_tower=False, norm_scale=1024):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OCPHead, self).__init__()
        self.num_classes = 1 # Objectness
        self.stride = stride
        self.norm_reg_targets = norm_reg_targets
        self.use_dcn_in_tower = use_dcn_in_tower

        cls_tower = []
        bbox_tower = []
        center_tower = []
        for i in range(num_convs):
            if self.use_dcn_in_tower and i == num_convs - 1:
                # conv_func = DFConv2d
                raise NotImplementedError("DFConv currently not supported.")
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

            center_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            center_tower.append(nn.GroupNorm(32, in_channels))
            center_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('center_tower', nn.Sequential(*center_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.xyhw_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.center_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )


        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.center_tower,
                        self.cls_logits, self.xyhw_pred, self.center_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scale = Scale(init_value=norm_scale/self.stride)

    def forward(self, features):
        cls_tower = self.cls_tower(features)
        box_tower = self.bbox_tower(features)
        center_tower = self.center_tower(features)

        cls_logits = self.cls_logits(cls_tower)
        center_logits = self.center_logits(center_tower)

        xyhw_reg = self.scale(self.xyhw_pred(box_tower)) # No activation (to allow positive and negative values)

        return cls_logits, xyhw_reg, center_logits


def generate_center_gt(shape, centers, sigma=1.0):
    """Generate center GT for FCOS. A 2d gaussian centered at each center with std sigma."""
    h,w = shape
    shifts_x = torch.arange(0, w, dtype=torch.float32, device=centers.device)
    shifts_y = torch.arange(0, h, dtype=torch.float32, device=centers.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

    centers = centers.unsqueeze(2).unsqueeze(3)
    centers = centers.repeat(1,1,h,w)

    dist = torch.sqrt((shift_x - centers[:,0])**2 + (shift_y - centers[:,1])**2)
    dist = dist / sigma
    dist = dist ** 2
    dist = -dist / 2
    dist = torch.exp(dist)

    dist = dist.max(0).values

    return dist


class OCPModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, hidden_dim, num_layers, fpn_strides, return_proposals=True, num_proposals=250, weight_dict={}):
        super(OCPModule, self).__init__()

        self.fpn_strides = fpn_strides
        heads = [OCPHead(hidden_dim, num_layers, stride) for stride in fpn_strides]
        self.heads = nn.ModuleList(heads)

        self.weight_dict = weight_dict
        self.return_proposals = return_proposals
        self.num_proposals = num_proposals

    def forward_heads(self, fpn):
        cls_logits_all = []
        xyhw_reg_all = []
        center_logits_all = []
        for l, feature in enumerate(fpn):
            cls_logits, xyhw_reg, center_logits = self.heads[l](feature)

            cls_logits_all.append(cls_logits)
            xyhw_reg_all.append(xyhw_reg)
            center_logits_all.append(center_logits)

        return cls_logits_all, xyhw_reg_all, center_logits_all

    def forward(self, features, features_out=None, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            features_out (list[Tensor]): features used to build the queries from proposals
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, xyhw_regression, center_logits = self.forward_heads(features)

        aux_out = {
            'box_cls': box_cls,
            'xyhw_reg': xyhw_regression,
            'center_logits': center_logits
        }

        losses = None
        if self.training and targets is not None:
            losses = self.loss(box_cls, xyhw_regression, center_logits, targets)

        # Objectness probs
        # obj_probs = [cls_log.sigmoid() for cls_log in box_cls]

        if self.return_proposals:
            proposals = self.generate_proposals(features_out, box_cls, xyhw_regression, center_logits)
            return losses, aux_out, proposals

        return losses, aux_out

    def loss_single_level(self, box_cls, xyhw_reg, center_logits, targets, targets_unsel):

        targets_xyhw_reg = []
        targets_masks = []
        targets_centers = []

        # Prepare center and mask targets
        for target_i in targets:
            if target_i['labels'].shape[0] == 0:
                targets_xyhw_reg.append(torch.zeros_like(xyhw_reg[0]))
                targets_masks.append(torch.zeros_like(box_cls[0]))
                targets_centers.append(torch.zeros_like(box_cls[0]))
                continue

            masks = F.interpolate(target_i['masks'].unsqueeze(1).float(), xyhw_reg.shape[-2:], mode='area') # i, 1, h, w
            masks = (masks > 0.5).float()

            t,_,h,w = masks.shape
            shifts_x = torch.arange(0, w, dtype=torch.float32, device=xyhw_reg.device)
            shifts_y = torch.arange(0, h, dtype=torch.float32, device=xyhw_reg.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

            # Scale with w,h to make size independet
            target_xyhw = target_i['boxes'] * torch.tensor([w,h,w,h], dtype=torch.float32, device=xyhw_reg.device)
            target_centers = target_xyhw[:, :2]
            target_hw = target_xyhw[:, 2:]

            coords = torch.stack([shift_x, shift_y], dim=0).unsqueeze(0).repeat(t,1,1,1) # i, 2, h, w
            target_xy_reg = (target_centers.unsqueeze(2).unsqueeze(3) - coords) * masks
            target_xy_reg = target_xy_reg.sum(0)

            target_hw_reg = target_hw.unsqueeze(2).unsqueeze(3) * masks
            target_hw_reg = target_hw_reg.sum(0)

            target_xyhw_reg = torch.cat([target_xy_reg, target_hw_reg], dim=0)

            target_centers_gauss = generate_center_gt((h,w), target_centers).unsqueeze(0)

            targets_xyhw_reg.append(target_xyhw_reg)
            targets_masks.append(masks.sum(0).clamp_max(1.0))
            targets_centers.append(target_centers_gauss)

        # Prepare ignore masks (ignore non-selected targets in loss)
        valid_masks = []
        for target_unsel_i in targets_unsel:
            if target_unsel_i['labels'].shape[0] == 0:
                valid_masks.append(torch.ones_like(box_cls[0]))
                continue

            masks = F.interpolate(target_unsel_i['masks'].unsqueeze(1).float(), xyhw_reg.shape[-2:], mode='area') # i, 1, h, w
            masks = (masks > 0.5).float()

            valid_masks.append(1 - masks.sum(0).clamp_max(1.0))

        targets_xyhw_reg = torch.stack(targets_xyhw_reg)
        targets_masks = torch.stack(targets_masks)
        targets_centers = torch.stack(targets_centers)
        targets_valid = torch.stack(valid_masks)

        # 1. L1 loss for xy_center regression
        xy_reg_loss = (F.l1_loss(xyhw_reg, targets_xyhw_reg, reduction='none') * targets_masks * targets_valid).mean((2,3)).sum()

        # 2. Cross entropy / focal loss for box_cls
        cls_loss = (sigmoid_focal_loss(box_cls, targets_masks, alpha=0.25, gamma=2.0, reduction='none') * targets_valid).mean((2,3)).sum()

        # 3. Cross entropy / focal loss for center_logits
        center_loss = (sigmoid_focal_loss(center_logits, targets_centers, alpha=0.25, gamma=2.0, reduction='none')).mean((2,3)).sum()

        return {
            'center_reg_loss': self.weight_dict['center_reg_loss'] * xy_reg_loss,
            'cls_loss': self.weight_dict['cls_loss'] * cls_loss,
            'center_loss': self.weight_dict['center_loss'] * center_loss
        }


    def loss(self, box_cls, xyhw_reg, center_logits, targets):
        losses = {}
        for box_cls_l, xyhw_reg_l, center_logits_l, stride in zip(box_cls, xyhw_reg, center_logits, self.fpn_strides):
            # Select relevant targets (depending on size)
            targets_l, targets_unsel_l = self.filter_targets_for_stride(targets, stride)

            losses_l = self.loss_single_level(box_cls_l, xyhw_reg_l, center_logits_l, targets_l, targets_unsel_l)

            losses_l = {k + '_s{}'.format(stride): v for k, v in losses_l.items()}
            losses.update(losses_l)


        return losses

    @staticmethod
    def _get_abs_xy(xy_reg, normalize=True):
        _,h,w = xy_reg.shape
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=xy_reg.device)
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=xy_reg.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        centers = torch.stack([shift_x, shift_y], dim=0)
        centers = centers + xy_reg[:2]

        if normalize: # Normalize coordinates to range [0, 1]
            centers[0] /= w
            centers[1] /= h

        return centers

    @staticmethod
    def _get_obj_probs(xyhw_reg, obj_centers, obj_probs):
        abs_xy = OCPModule._get_abs_xy(xyhw_reg, normalize=True)

        # Add dummy 0 object
        obj_centers = F.pad(obj_centers, (0,0,1,0), mode='constant', value=0)

        # Distance to proposals
        T = 0.02 # rel px
        diff = (abs_xy - obj_centers[..., None, None]).square().sum(dim=1)
        obj_diff, obj_ids = diff.min(0)
        obj_ids_m = obj_ids * (obj_diff < T**2)

        # Get object masks
        obj_masks = F.one_hot(obj_ids_m, num_classes=len(obj_centers)).permute(2,0,1)
        obj_masks = obj_masks[1:] * obj_probs

        return obj_masks

    def generate_proposals(self, features, box_cls, xyhw_regression, center_logits, threshold=0.1):

        center_probs = [center_log.detach().sigmoid() for center_log in center_logits]
        peaks_all = []
        for probs in center_probs:
            # Non-maximum suppression
            maxima = F.max_pool2d(probs, kernel_size=5, stride=1, padding=2)
            peak_mask = (probs == maxima) & (probs > threshold)
            peaks_all.append(peak_mask)

        # Process each batch item separately
        probs_all = []
        boxes_all = []
        feats_all = []
        levels_all = []
        padding_masks = []
        for bi in range(center_logits[0].shape[0]):
            boxes_all_i = []
            probs_all_i = []
            feats_all_i = []
            levels_all_i = []
            for lvl, (feats, probs, xyhw_reg, peak_mask) in enumerate(zip(features, center_probs, xyhw_regression, peaks_all)):
                _,ys,xs = torch.where(peak_mask[bi])

                boxes_p = xyhw_reg[bi,:,ys,xs].detach()
                probs_p = probs[bi,:,ys,xs].detach()
                feats_p = feats[bi,:,ys,xs]

                # Add center coordinates
                boxes_p[:2] += torch.stack([xs, ys], dim=0)
                boxes_p = boxes_p.clamp(min=1e-5)

                # Normalize
                h,w = feats.shape[-2:]
                boxes_p /= torch.tensor([w,h,w,h], dtype=torch.float32, device=feats.device).view(4,1)

                boxes_all_i.append(boxes_p)
                probs_all_i.append(probs_p)
                feats_all_i.append(feats_p)
                levels_all_i.append((torch.ones_like(probs_p)[0] * lvl).long())

            boxes_all_i = torch.cat(boxes_all_i, dim=1)
            probs_all_i = torch.cat(probs_all_i, dim=1)
            feats_all_i = torch.cat(feats_all_i, dim=1)
            levels_all_i = torch.cat(levels_all_i)

            # Sort and select top K
            idx_s = probs_all_i.argsort(dim=1, descending=True)[0,:self.num_proposals]
            boxes_all_i = boxes_all_i[:,idx_s]
            probs_all_i = probs_all_i[:,idx_s]
            feats_all_i = feats_all_i[:,idx_s]
            levels_all_i = levels_all_i[idx_s]

            # Extract features for each proposal for each level
            for lvl, (feats, xyhw_reg, obj_probs_unsig) in enumerate(zip(features, xyhw_regression, box_cls)):
                idxs = levels_all_i == lvl
                obj_centers_l = boxes_all_i[:2, idxs]
                feats_center_l = feats_all_i[:, idxs]
                obj_probs = obj_probs_unsig.sigmoid()
                obj_masks = self._get_obj_probs(xyhw_reg[bi], obj_centers_l.permute(1,0), obj_probs[bi]) # N, H, W

                feats_p = feats[bi]
                obj_masks = obj_masks

                # Probability-weighted pooling (mask + 1 pixel from center)
                feats_p = (torch.einsum('chw,nhw->nc', feats_p, obj_masks) + feats_center_l.permute(1,0)) / (obj_masks.sum(dim=(1,2)).unsqueeze(1) + 1) # N, C
                feats_all_i[:,idxs] = feats_p.permute(1,0)

            # Pad to num_proposals
            padding_mask_i = torch.zeros(len(idx_s), dtype=torch.bool, device=feats.device)
            if (len(idx_s) < self.num_proposals):
                diff = self.num_proposals - len(idx_s)
                boxes_all_i = F.pad(boxes_all_i, (0, diff))
                probs_all_i = F.pad(probs_all_i, (0, diff))
                feats_all_i = F.pad(feats_all_i, (0, diff))
                levels_all_i = F.pad(levels_all_i, (0, diff))
                padding_mask_i = F.pad(padding_mask_i, (0, diff), value=True)

            boxes_all.append(boxes_all_i)
            probs_all.append(probs_all_i)
            feats_all.append(feats_all_i)
            levels_all.append(levels_all_i)
            padding_masks.append(padding_mask_i)

        boxes_all = torch.stack(boxes_all)
        probs_all = torch.stack(probs_all)
        feats_all = torch.stack(feats_all)
        levels_all = torch.stack(levels_all)
        padding_masks = torch.stack(padding_masks)

        return boxes_all, probs_all, feats_all, levels_all, padding_masks


    def filter_targets_for_stride(self, targets, stride):
        minsize_for_stride = {
            4: (0, 64),
            8: (32, 128),
            16: (64, 256),
            32: (128, 512),
            64: (256, 10_000),
        }

        mins, maxs = minsize_for_stride[stride]

        targets_sel = []
        targets_unsel = []
        for target_i in targets:
            # Select only things between min and max allowed size
            diag2 = (target_i['boxes'][:, 2]**2 + target_i['boxes'][:, 3]**2).sqrt()
            diag2 = diag2 * target_i['masks'].shape[-1]

            valid_mask = (target_i['isthing'] == 1) & \
                         (diag2 > mins) & (diag2 < maxs)
            target_i_sel = {k: v[valid_mask] for k, v in target_i.items()}
            target_i_unsel = {k: v[~valid_mask] for k, v in target_i.items()}
            targets_sel.append(target_i_sel)
            targets_unsel.append(target_i_unsel)

        return targets_sel, targets_unsel
