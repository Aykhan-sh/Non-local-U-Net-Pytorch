import random
from liverfiles.utils import get_mask, get_nii, split_mask
import numpy as np
from torch.utils.data import Dataset


class Ds(Dataset):
    def __init__(self, df, crop_shape, transforms, nonzero_prob=0.5):
        self.df = df
        self.transforms = transforms
        self.shape = crop_shape
        self.nonzero_prob = nonzero_prob

    def __getitem__(self, idx):
        try:
            path, img_id = self.df.iloc[idx].values
            img, mask = get_nii(path), get_mask(img_id)
            img, mask = self.random_crop3d(img, mask)
            img = np.transpose(img, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            mask = split_mask(mask).astype('int16')
            return img, mask
        except:
            print(self.df.path[idx])

    def min_max(self, img):
        img = img - img.min()
        if img.max() != 0:
            img = img / img.max()
        return img

    def crop(self, img, p):
        return img[p[0]: p[0] + self.shape[0],
               p[1]: p[1] + self.shape[1],
               p[2]: p[2] + self.shape[2]]

    def random_crop3d(self, img, mask):
        if type(self.nonzero_prob) is bool:
            nonzero = self.nonzero_prob
        else:
            nonzero = random.random() > self.nonzero_prob
        if nonzero:
            p = np.argwhere(mask == 1)
            idx = np.random.randint(0, len(p))
            p = p[idx]
            p = p - (self.shape - np.clip(mask.shape - p, 0, self.shape))
            new = self.crop(img, p)
            new = self.min_max(new)
        else:
            p = []  # initial points
            for i in range(3):
                p.append(np.random.randint(0, img.shape[i] - self.shape[i]))
            new = self.crop(img, p)
            new = self.min_max(new)
            # if new.sum() < 0.3 * np.prod(self.shape):
            #     return self.random_crop3d(img, mask, False)
        return new, self.crop(mask, p)

    def __len__(self):
        return len(self.df)
