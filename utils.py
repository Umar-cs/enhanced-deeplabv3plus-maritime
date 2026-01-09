import torch
import numpy as np


def calculate_pixel_accuracy(output, mask, ignore_index: int = 255):
    """
    Pixel accuracy ignoring pixels with label == ignore_index.
    Works with either a dict output ({"main": logits}) or a raw tensor.
    """
    if isinstance(output, dict):
        output = output["main"]

    with torch.no_grad():
        # output: (B, C, H, W), mask: (B, H, W)
        preds = torch.argmax(output, dim=1)

        valid = (mask != ignore_index)          # <-- ignore 255
        if valid.sum() == 0:
            return 0.0

        correct = (preds == mask) & valid
        acc = correct.sum().float() / valid.sum().float()
        return acc.item()


def calculate_iou(output, mask, num_classes: int = 3, ignore_index: int = 255):
    """
    Per-class IoU, ignoring pixels with label == ignore_index.
    Returns a Python list of IoUs for each class (may contain NaN).
    """
    if isinstance(output, dict):
        output = output["main"]

    with torch.no_grad():
        preds = torch.argmax(output, dim=1)     # (B, H, W)
        iou_per_class = []

        # Flatten for easier logic
        preds_f = preds.view(-1)
        mask_f = mask.view(-1)
        valid = (mask_f != ignore_index)

        for cls in range(num_classes):
            pred_inds = (preds_f == cls) & valid
            target_inds = (mask_f == cls) & valid

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union == 0:
                iou = float('nan')  # no samples for this class
            else:
                iou = (intersection / union).item()

            iou_per_class.append(iou)

        return iou_per_class
