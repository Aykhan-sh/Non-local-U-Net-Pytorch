import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


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


class InferenceDataset(Dataset):
    def __init__(self, img, start_coordinates, input_size):
        self.img = img
        self.sc = start_coordinates
        self.ins = input_size

    def __getitem__(self, idx):
        i, j, k = self.sc[idx]
        patch = self.img[:, i:i + self.ins[0], j:j + self.ins[1], k:k + self.ins[2]]
        return patch

    def __len__(self):
        return len(self.sc)


@torch.no_grad()
def infer(model, img, ins, out_classes, window, batch_size, num_workers, device):  # C, W, H, D
    """
    :param model:
    :param img:
    :param ins: model input shapes
    :param out_classes:
    :param window:
    :param batch_size:
    :param num_workers:
    :param device:
    :return:
    """
    if isinstance(window, int):
        window = (window, window, window)
    is_valid = True
    for i in range(3):
        is_valid = is_valid and window[i] <= ins[i]
    assert is_valid, "Window must be <= than the lowest dimension of models input shape"
    model.eval()
    model.to(device)
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    ors = img.shape[-3:]  # original shape

    pad = (window - (np.array(img.shape[-3:]) % window)) % window  # calculation padding for image
    pad = [(0, 0)] + [(0, i) for i in pad]
    img = np.pad(img, pad, constant_values=img.min(), mode="constant")  # padding the image
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
    ds = InferenceDataset(img, start_coordinates, ins)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
    for idx, patch in enumerate(dl):
        preds = model(patch.to(device).float())
        for patch_idx in range(batch_size):  # iteration over the batch
            current_idx = idx * batch_size + patch_idx
            if current_idx == len(start_coordinates):
                break
            i, j, k = start_coordinates[current_idx]
            result[:, i:i + ins[0], j:j + ins[1], k:k + ins[2]] += to_numpy(preds[patch_idx])
            result_cnt[i:i + ins[0], j:j + ins[1], k:k + ins[2]] += 1
    result = result[:, :ors[0], :ors[1], :ors[2]]
    result_cnt = result_cnt[:ors[0], :ors[1], :ors[2]]
    result = result / result_cnt
    return result
# dolboyebskiy inference ya by skazal
