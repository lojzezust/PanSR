"""PanSR contribution (4): Mask-conditioned queries.

A training-time auxiliary query group (the evolution of DN-DETR denoising queries) whose
*content* embeddings are sampled from backbone/FPN features **inside the ground-truth masks**
(hence "mask-conditioned"), and whose *positional* embeddings are noised GT boxes. Each GT
object is repeated into groups; an attention mask keeps groups from seeing each other while
letting them attend to the real matching/background queries. These queries provide a dense,
mask-aware learning signal and are discarded at inference time.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..utils.utils import inverse_sigmoid


def randint(low, high, size):
    """Random integer generator that works with tensor inputs."""
    return torch.randint(2 ** 63 - 1, size=size, device=low.device) % (high - low) + low


def build_mask_conditioned_queries(decoder, targets, fpn, tgt=None, refpoint_emb=None, batch_size=None):
    """Build the mask-conditioned query group for one forward pass.

    Args:
        decoder: the :class:`PanSRDecoder` (provides config + the shared ``enc_output`` projection).
        targets: list of per-image GT dicts with ``labels``, ``masks``, ``boxes``, ``level``.
        fpn: list of FPN feature maps to sample content embeddings from.
        tgt, refpoint_emb, batch_size: only used in the (unused at train time) eval branch.

    Returns:
        ``(inputs_tgt, inputs_bbox_embed, attn_masks, padding_masks, mask_dict)``.
    """
    if decoder.training:
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]

        if max(known_num) == 0:
            return None, None, None, None, None

        inputs_bbox_embed = []
        inputs_tgt = []
        attn_masks = []
        padding_masks = []
        indices = []
        for bi in range(len(targets)):
            dn_max_group_size = decoder.dn_max_group_size
            dn_num = decoder.dn_num

            num_gt = len(targets[bi]['labels'])
            if num_gt == 0:
                inputs_bbox_embed.append(torch.zeros(dn_num, 4, dtype=fpn[0].dtype, device=fpn[0].device))
                inputs_tgt.append(torch.zeros(dn_num, fpn[0].shape[1], dtype=fpn[0].dtype, device=fpn[0].device))
                q_n = dn_num + decoder.num_bg_queries + decoder.num_queries
                attn_masks.append(torch.zeros(q_n, q_n, dtype=torch.bool, device=fpn[0].device))  # allow all
                padding_masks.append(torch.ones(dn_num, dtype=torch.bool, device=fpn[0].device))  # but ignore all dn queries
                indices.append((torch.tensor([], dtype=torch.int64, device=fpn[0].device), torch.tensor([], dtype=torch.int64, device=fpn[0].device)))
                continue

            gt_idx = np.arange(num_gt)
            group_size = min(num_gt, dn_max_group_size)
            num_repetitions = dn_num // group_size

            # Create groups
            gt_idxs = []
            group_ids = []
            for i in range(num_repetitions):
                sel_idx = gt_idx
                # Randomly sample a subset if there are too many GT objects
                if group_size != num_gt:
                    sel_idx = np.random.choice(gt_idx, group_size, replace=False)
                gt_idxs.append(sel_idx)
                group_ids.append(np.full(group_size, i))

            gt_idxs = np.concatenate(gt_idxs)
            group_ids = np.concatenate(group_ids)

            l_low = (targets[bi]['level'][gt_idxs] - 1).clip(min=0)
            l_high = (targets[bi]['level'][gt_idxs] + 2).clip(max=len(fpn))
            levels = randint(l_low, l_high, l_low.shape)

            # Get random points inside GT masks
            sample_points = torch.zeros(len(gt_idxs), 2, dtype=torch.float, device=levels.device)
            _, H, W = targets[bi]['masks'].shape
            for obj_i in np.unique(gt_idxs):
                mask = targets[bi]['masks'][obj_i]
                ys, xs = torch.where(mask)
                m = gt_idxs == obj_i
                n_obj = m.sum()
                ixs = np.random.randint(0, len(xs), n_obj)
                sample_points[m] = torch.stack([ys[ixs] / H, xs[ixs] / W], dim=1)

            # Sample features from FPN at selected points and levels (mask-conditioned content)
            input_tgt = torch.zeros(len(gt_idxs), fpn[0].shape[1], dtype=fpn[0].dtype, device=fpn[0].device)
            for i, feats in enumerate(fpn):
                level_mask = levels == i
                if level_mask.sum() == 0:
                    continue

                H, W = feats.shape[-2:]
                points = sample_points[level_mask]
                points = (points * torch.tensor([H, W], device=points.device)).round().long()
                points[:, 0] = points[:, 0].clamp(0, H - 1)
                points[:, 1] = points[:, 1].clamp(0, W - 1)

                feats_sel = feats[bi, :, points[:, 0], points[:, 1]].permute(1, 0)
                input_tgt[level_mask] = feats_sel.to(input_tgt.dtype)

            gt_boxes = targets[bi]['boxes'][gt_idxs]

            # Add noise to bounding boxes (positional component)
            noise_scale = decoder.noise_scale
            noise_boxes = gt_boxes
            if noise_scale > 0:
                diff = torch.zeros_like(gt_boxes)
                diff[:, :2] = gt_boxes[:, 2:] / 2
                diff[:, 2:] = gt_boxes[:, 2:]
                noise_boxes = gt_boxes + torch.mul((torch.rand_like(gt_boxes) * 2 - 1.0),
                                                   diff).to(gt_boxes.device) * noise_scale
                noise_boxes = noise_boxes.clamp(min=0.0, max=1.0)

            input_bbox_embed = inverse_sigmoid(noise_boxes)

            pad_size = dn_num - len(gt_idxs)
            padding_mask = torch.zeros(len(gt_idxs), dtype=torch.bool, device=levels.device)
            input_bbox_embed = F.pad(input_bbox_embed, (0, 0, 0, pad_size))
            input_tgt = F.pad(input_tgt, (0, 0, 0, pad_size))
            padding_mask = F.pad(padding_mask, (0, pad_size), value=True)  # Ignore padded

            inputs_bbox_embed.append(input_bbox_embed)
            inputs_tgt.append(input_tgt)

            # Prepare attention masks
            num_queries = dn_num + decoder.num_bg_queries + decoder.num_queries
            attn_mask = torch.ones(num_queries, num_queries, dtype=bool).to('cuda')

            # Every query can see real queries
            attn_mask[:, dn_num:] = False

            # Mask-conditioned queries can only see inside their own group
            for i in range(num_repetitions):
                attn_mask[group_size * i:group_size * (i + 1), group_size * i:group_size * (i + 1)] = False

            attn_masks.append(attn_mask)
            padding_masks.append(padding_mask)
            pred_idxs = torch.arange(len(gt_idxs), dtype=torch.int64, device=levels.device)
            gt_idxs_t = torch.as_tensor(gt_idxs, dtype=torch.int64, device=levels.device)
            indices.append((pred_idxs, gt_idxs_t))  # query -> GT matching

        padding_masks = torch.stack(padding_masks, dim=0)
        attn_masks = torch.stack(attn_masks, dim=0)
        inputs_tgt = torch.stack(inputs_tgt, dim=0)
        inputs_bbox_embed = torch.stack(inputs_bbox_embed, dim=0)

        # Use the same embedding projection as the matching part
        inputs_tgt = decoder.enc_output_norm(decoder.enc_output(inputs_tgt))

        mask_dict = {
            'indices_dn': indices,
            'pad_size': dn_num,
        }
    else:
        if refpoint_emb is not None:
            inputs_tgt = tgt.repeat(batch_size, 1, 1)
            inputs_bbox_embed = refpoint_emb.repeat(batch_size, 1, 1)
        else:
            inputs_tgt = None
            inputs_bbox_embed = None
        attn_masks = None
        padding_masks = None
        mask_dict = None

    return inputs_tgt, inputs_bbox_embed, attn_masks, padding_masks, mask_dict
