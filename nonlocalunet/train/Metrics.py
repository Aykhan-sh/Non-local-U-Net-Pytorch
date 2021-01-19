import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def mhd(gt, preds):
    """
    :param gt: ndarray of shape [B, C, D, H, W]
    :param preds: ndarray of shape [B, C, D, H, W]
    :return: ndarray of shape [B, C] 3D-Modified Hausdorff Distance
    """
    assert gt.shape == preds.shape, "input arrays must be same size"
    assert np.isin(np.unique(gt), [0, 1], invert=True).sum() == 0 and \
           np.isin(np.unique(preds), [0, 1], invert=True).sum() == 0, "values of input arrays must be in [0,1]"
    transpose_policy = [(1, 2, 0), (0, 1, 2), (0, 2, 1)]
    result = []
    for batch in range(gt.shape[0]):
        result_by_batch = []
        for channel in range(gt.shape[1]):
            result_by_channel = 0
            for i in transpose_policy:
                a = gt[batch, channel].transpose(*i)
                a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))
                b = preds[batch, channel].transpose(*i)
                b = b.reshape((b.shape[0], b.shape[1] * b.shape[2]))
                result_by_channel += max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
            result_by_channel /= len(transpose_policy)
            result_by_batch.append(result_by_channel)
        result.append(result_by_batch)
    return result


def dr(gt, preds):
    """
    :param gt: ndarray of [B, C, D, H, W] shape
    :param preds: ndarray of [B, C, D, H, W] shape
    :return: int dice ratio.
    """
    assert gt.shape == preds.shape, "input arrays must be same size"
    assert np.isin(np.unique(gt), [0, 1], invert=True).sum() == 0 and \
           np.isin(np.unique(preds), [0, 1], invert=True).sum() == 0, "values of input arrays must be in [0,1]"
    result = []
    for batch in range(gt.shape[0]):
        result_by_batch = []
        for channel in range(gt.shape[1]):
            intersection = (gt * preds).sum()
            union = gt.sum() + preds.sum()
            if union == 0:
                dice_ratio = 1
            else:
                dice_ratio = 2 * intersection / union
            result_by_batch.append(dice_ratio)
        result.append(result_by_batch)
    return np.array(result)
