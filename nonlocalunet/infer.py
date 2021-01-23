import torch
import numpy as np
from collections.abc import Iterable
from numbers import Number


@torch.no_grad()
def infer(self, model, img, input_shape, window, batch):  # C, W, H, D
    if isinstance(window, Number):
        window = (window, window, window)
    is_valid = True
    for i in range(3):
        is_valid = is_valid and window[i] < input_shape[i]
    assert is_valid, "Window must be <= than the lowest dimension of models input shape"
