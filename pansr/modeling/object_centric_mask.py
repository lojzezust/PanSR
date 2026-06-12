# ------------------------------------------------------------------------
# Copyright (c) 2026 University of Ljubljana. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""PanSR contribution (2): Object-centric mask prediction.

Foreground (thing) mask logits are constrained to lie within each query's predicted
bounding box (slightly dilated). This couples the mask head to the object-centric box
prediction, suppressing mask leakage outside the object extent. Background (stuff) masks
are left unconstrained.

Behaviour is identical to the original "seg bbox limit": pixels outside the (dilated) box
are set to a large negative logit so they vanish after the sigmoid.
"""
import torch

# Large negative logit used to mask out pixels outside the box.
# (-inf is avoided because it produces NaNs in downstream ops; -300 saturates the sigmoid.)
_OUTSIDE_BOX_LOGIT = -300.0


def boxes_to_box_masks(pred_boxes, shape, dilate=0.1, min_px=2):
    """Build boolean masks that are True inside each (dilated) predicted box.

    Args:
        pred_boxes: ``(B, N, 4)`` boxes in normalized ``cxcywh`` format.
        shape: ``(h, w)`` of the mask grid.
        dilate: fractional box dilation applied to width/height.
        min_px: minimum dilation in pixels (in the mask grid's resolution).

    Returns:
        Boolean tensor ``(B, N, h, w)``, True where a pixel falls inside the box.
    """
    pred_boxes = pred_boxes.detach()
    h, w = shape

    x = torch.linspace(0, 1, w, device=pred_boxes.device).half()
    y = torch.linspace(0, 1, h, device=pred_boxes.device).half()
    yy, xx = torch.meshgrid(y, x)

    dx = xx[None, None] - pred_boxes[:, :, 0][..., None, None]
    dy = yy[None, None] - pred_boxes[:, :, 1][..., None, None]

    min_hw = torch.tensor([min_px / w, min_px / h], device=pred_boxes.device, dtype=pred_boxes.dtype)
    hw_delta = (dilate * pred_boxes[:, :, 2:4]).clamp(min_hw)
    boxes_hw = pred_boxes[:, :, 2:4] + hw_delta

    bx = dx.abs() <= 0.5 * boxes_hw[:, :, 0, None, None]
    by = dy.abs() <= 0.5 * boxes_hw[:, :, 1, None, None]

    return (bx * by) > 0


def limit_masks_to_boxes(mask_logits, pred_boxes, shape, dilate=0.1, min_px=2):
    """Constrain ``mask_logits`` to the predicted boxes (contribution 2).

    Args:
        mask_logits: ``(B, N, h, w)`` foreground mask logits.
        pred_boxes: ``(B, N, 4)`` boxes in normalized ``cxcywh`` format.
        shape: ``(h, w)`` of the mask grid.
        dilate, min_px: box dilation parameters (see :func:`boxes_to_box_masks`).

    Returns:
        Mask logits with out-of-box pixels set to a large negative value.
    """
    box_masks = boxes_to_box_masks(pred_boxes, shape, dilate=dilate, min_px=min_px)
    return torch.where(box_masks, mask_logits, _OUTSIDE_BOX_LOGIT)
