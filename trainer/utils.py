import numpy as np
import SimpleITK as sitk
import os


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
