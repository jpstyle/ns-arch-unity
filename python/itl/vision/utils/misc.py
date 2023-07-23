"""
Miscellaneous utility methods that don't classify into other files in utils
"""
from itertools import product
import numpy as np


def masks_bounding_boxes(masks):
    """
    Axis-aligned bounding boxes (in xyxy format) from min/max indices for nonzero
    value in masks
    """
    boxes = []

    for msk in masks:
        nz_inds_y, nz_inds_x = msk.nonzero()
        x_min, x_max = nz_inds_x.min(), nz_inds_x.max()
        y_min, y_max = nz_inds_y.min(), nz_inds_y.max()
        boxes.append([x_min, y_min, x_max, y_max])

    return boxes

def mask_iou(masks_1, masks_2):
    """
    Compute pairwise mask-IoUs between two lists of segmentation masks (torchvision.ops
    only has box-IoU). Masks_1 & 2 are two list-likes of segmentation masks of same
    size ((N1, H, W), (N2, H, W)), returns N * M array of float values between 0 and 1.
    """
    intersections = [np.minimum(m1, m2) for m1, m2 in product(masks_1, masks_2)]
    unions = [np.maximum(m1, m2) for m1, m2 in product(masks_1, masks_2)]
    ious = np.array([i.sum() / u.sum() for i, u in zip(intersections, unions)])
    ious = ious.reshape(len(masks_1), len(masks_2))

    return ious


def mask_nms(masks, scores, iou_thres):
    """
    Non-maximum suppression by mask-IoU (torchvision.ops only has NMS based
    on box-IoU). Return list of indices of kept masks, sorted.
    """
    # Return value
    kept_indices = []

    # Index queue sorted by score
    queue = np.argsort(scores)

    while len(queue) > 0:
        # Pop index of highest score from queue and add to return list
        ind = queue[-1]
        queue = queue[:-1]
        kept_indices.append(ind)

        # Can break if no more remaining items in queue
        if len(queue) == 0: break

        # Compute mask IoUs between the popped mask vs. remaining masks
        ious = mask_iou(masks[ind, None], masks[queue])[0]

        # Filter out masks with IoUs higher than threshold
        queue = queue[ious < iou_thres]

    return kept_indices


def flatten_cfg(cfg_entry, prefix=None):
    """ For flattening nested config dict using '.' as separator """
    if isinstance(cfg_entry, dict):
        return {
            f"{prefix}.{in_k}" if prefix is not None else in_k: in_v
            for k, v in cfg_entry.items()
            for in_k, in_v in flatten_cfg(v, k).items()
        }
    else:
        return { prefix: cfg_entry }
