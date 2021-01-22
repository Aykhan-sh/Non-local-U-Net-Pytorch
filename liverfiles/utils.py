import nibabel as nib
import numpy as np
import os
import torch
from prettytable import PrettyTable


def get_nii(path):
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img


def get_mask(path):
    if type(path) is not str:
        path = f'data/segmentations/segmentation-{path}.nii'
    return get_nii(path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def path_uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


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
