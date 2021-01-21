import nibabel as nib
import numpy as np


def get_nii(path):
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img


def get_mask(path):
    if type(path) is not str:
        path = f'../input/liver-tumor-segmentation/segmentations/segmentation-{path}.nii'
    return get_nii(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MetricCounter:
    def __init__(self, metric_dict, target_metric, maximize):
        """
        :param metric_dict: dictionary where key is string - metric name and value is metric function
        function must take two numpy arrays (gt, preds) and return a number
        :param logger: torch.utils.tensorboard.SummaryWriter
        :param target_metric: metric name to compare
        :param maximize: bool. True if target_metric needed to be maximized.
        """
        self.metric_dict = metric_dict
        self.target_metric = target_metric
        self.maximize = maximize

    def __call__(self, gt, preds, other_metrics=None):
        """
        :param gt: ground truth
        :param preds: predictions
        :return: dictionary where key is metric name and value is score
        """
        result = {}
        for name, func in self.metric_dict.items():
            result[name] = func(to_numpy(gt), to_numpy(preds))
        result.update(other_metrics)
        return result

    @staticmethod
    def print_metrics(metric_dict, ceil=4):
        """
        :param metric_dict: return of __call__ function
        :param ceil: ceil metrics to nth number
        :return: None. only print mertics
        """
        for key, val in metric_dict.items():
            if key == 'epoch' or key == 'era':
                continue
            if type(val) is str:
                print(f'{key}: {val}', end='; ')
            else:
                print(f'{key}: {val:.{ceil}f}', end='; ')

    def compare_target(self, a, b):
        if self.maximize:
            if a[self.target_metric] >= b[self.target_metric]:
                return True
            else:
                return False
        else:
            if a[self.target_metric] <= b[self.target_metric]:
                return True
            else:
                return False
