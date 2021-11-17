import numpy as np


def dr_score(gt, preds):
    """
    :param gt: ndarray of [B, C, D, H, W] shape
    :param preds: ndarray of [B, C, D, H, W] shape
    :return: int dice ratio.
    """
    assert gt.shape == preds.shape, "input arrays must be same size"
    intersection = (gt * preds).sum()
    union = gt.sum() + preds.sum()
    dice_ratio = 2 * intersection / union
    return dice_ratio


def iou_score(target, pred):
    target = target.flatten()
    pred = pred.flatten()

    intersect = np.dot(target, pred)
    union = (target + pred).sum() - intersect
    if union == 0:
        return 1
    return intersect / union
