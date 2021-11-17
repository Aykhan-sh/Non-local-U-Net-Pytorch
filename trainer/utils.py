import SimpleITK as sitk
import numpy as np
import os
import torch
from prettytable import PrettyTable
import cv2
from torch.utils.tensorboard import SummaryWriter
from trainer.metrics import dr_score, iou_score

label_names = ['Background', 'Liver', 'Bladder', 'Lungs', 'Kidneys', 'Bone', 'Brain']


def split_mask(mask, num_of_classes):
    mask = np.stack([mask == i for i in range(num_of_classes)])
    return mask


def open_ct(volume, path='data/raw data/volumes'):
    ct = sitk.ReadImage(os.path.join(path, f'volume-{volume}.nii.gz'))
    ct = sitk.GetArrayFromImage(ct)
    return ct


def open_mask(volume, path='data/raw data/masks'):
    ct = sitk.ReadImage(os.path.join(path, f'labels-{volume}.nii.gz'))
    ct = sitk.GetArrayFromImage(ct)
    return ct


def count_metrics(gt, preds_raw, post_preds, mode):
    """
    :param gt: ndarray of [B, C, D, H, W] shape
    :param preds_raw: ndarray of [B, C, D, H, W] shape
    :param preds: ndarray of [B, C, D, H, W] shape binarized
    :mode: string [Train, Val]
    """
    metrics = {f'{mode} Dice': {}, f'{mode} IOU': {}}
    for idx, i in enumerate(label_names):
        metrics[f'{mode} Dice'][i] = dr_score(gt[:, idx, :, :, :], preds_raw[:, idx, :, :, :])
        metrics[f'{mode} IOU'][i] = iou_score(gt[:, idx, :, :, :], post_preds[:, idx, :, :, :])
    return metrics


def sum_metrics(metric_dict, metric_dict2):
    for metric in metric_dict:
        for label in metric_dict[metric]:
            metric_dict[metric][label] += metric_dict2[metric][label]
    return metric_dict


def divide_metrics(metric_dict, divide_by):
    for metric in metric_dict:
        for label in metric_dict[metric]:
            metric_dict[metric][label] /= divide_by
    return metric_dict


def update_metrics(metric_dict, metric_dict2):
    for metric in metric_dict:
        if metric == 'Loss':
            metric_dict[metric].update(metric_dict2[metric])
        if metric in label_names:
            for label in metric_dict[metric]:
                metric_dict[metric][label].update(metric_dict2[metric][label])
    return metric_dict


def min_max(array):
    array -= array.min()
    array_max = array.max()
    if array_max != 0:
        array = array / array_max
    return array


def to_uint(array):
    array = min_max(array)
    array *= 255
    array = array.astype('uint8')
    return array


def unsplit_binary_mask(mask):
    """
    :param mask: binary mask of shape [B, C, D, H, W] or [B, C, H, W]
    :return: mask of shape [B, D, H, W] or [B, H, W] with integers of corresponding classes
    """
    img_size = mask.shape[-2:]
    mask = np.insert(mask, 0, np.zeros(img_size), axis=1)
    for i in range(mask.shape[1]):  # assign to each channel its class int
        mask[:, i, :, :] *= i
    mask = mask.argmax(axis=1)
    return mask


def to_numpy(array):
    """
    :param array: numpy, list, tensor
    :return: numpy
    """
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    elif type(array) is list:
        return np.array(array)
    else:
        return array


def img_with_masks(img, masks, alpha, return_colors=False):
    '''
    returns image with masks,
    img - numpy array of image
    masks - list of masks. Maximum 6 masks. only 0 and 1 allowed
    alpha - int transparency [0:1]
    return_colors returns list of names of colors of each mask
    '''
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [102, 51, 0]
    ]
    color_names = [
        'Red',
        'Green',
        'Blue',
        'Yellow',
        'Light',
        'Brown'
    ]
    img = to_uint(img)
    img = np.repeat(img, 3, -3)
    for c, mask in enumerate(masks):
        mask = to_numpy(mask)
        if mask.dtype == bool:
            mask.astype('uint8')
        else:
            mask = min_max(mask)
        if len(mask.shape):
            mask = np.dstack((mask, mask, mask))
        else:
            mask = np.repeat(mask, 3, -3)
        mask = mask.swapaxes(-1, -3)
        mask = mask * np.array(colors[c])
        mask = mask.swapaxes(-1, -3)
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(mask, alpha, img, 1, 0.)
    if not return_colors:
        return img
    else:
        return img, color_names[0:len(masks)]


# Log Utils
def create_logger(log_name=None, log_path='runs'):
    path = os.path.join(log_path, log_name)
    os.makedirs(log_path, exist_ok=True)
    path = path_uniquify(path)
    log = SummaryWriter(log_dir=path)
    os.mkdir(os.path.join(log.log_dir, 'weights'))
    return log


def path_uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


# Model Utils
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
