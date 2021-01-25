import torch
import numpy as np
from collections.abc import Iterable
from numbers import Number


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


@torch.no_grad()
def infer(model, img, inp_s, out_classes, window, batch_size, device):  # C, W, H, D
    """
    :param self:
    :param model:
    :param img:
    :param inp_s: model input shapes
    :param out_classes:
    :param window:
    :param batch_size:
    :param device:
    :return:
    """
    if isinstance(window, int):
        window = (window, window, window)
    is_valid = True
    for i in range(3):
        is_valid = is_valid and window[i] <= inp_s[i]
    assert is_valid, "Window must be <= than the lowest dimension of models input shape"
    model.to(device)
    ors = img.shape[-3:]

    pad = (window - (np.array(img.shape[-3:]) % window)) % window  # calculation padding for image
    pad = [(0, 0)] + [(0, i) for i in pad]
    img = np.pad(img, pad, constant_values=img.min())  # padding the image
    result = np.zeros((out_classes, *img.shape[-3:]))  # creating empty mask of shape [W, H, D]
    result_cnt = np.zeros(img.shape[-3:])  # counter of intersections
    slides = []  # how many times we have to iterate over each dimension
    for i in range(3):
        slides.append(img.shape[-3:][i] // window[i])
    start_coordinates = []  # create list of all start coordintaes for patches
    for i in range(slides[0]):
        for j in range(slides[1]):
            for k in range(slides[2]):
                ii = i * window[0]  # start coordinates of patch
                jj = j * window[1]
                kk = k * window[2]
                start_coordinates.append((ii, jj, kk))
    for idx, (i, j, k) in enumerate(start_coordinates):  # predicting loop
        batch = []
        batch_coords = []
        while len(batch) != batch_size:  # collecting batch
            temp_image = img[:, i:i + inp_s[0], j:j + inp_s[1], k:k + inp_s[2]]
            batch.append(temp_image)
            batch_coords.append((i, j, k))
        batch = torch.tensor(batch)  # converting to torch
        preds = model(batch.to(device).float())  # making prediciton
        for idx_batch, (ii, jj, kk) in enumerate(batch_coords):  # iteration over the batch
            result[:, ii:ii + inp_s[0], jj:jj + inp_s[1], kk:kk + inp_s[2]] = to_numpy(preds[idx_batch])
            result_cnt[ii:ii + inp_s[0], jj:jj + inp_s[1], kk:kk + inp_s[2]] += 1
    result = result[:, :ors[0], :ors[1], :ors[2]]
    result_cnt = result_cnt[:ors[0], :ors[1], :ors[2]]
    result = result / result_cnt
    return result
