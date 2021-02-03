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
        # TODO add validation

    def __getitem__(self, idx):
        try:
            path, img_id = self.df.iloc[idx].values
            img, mask = get_nii(path), get_mask(img_id)
            img, mask = self.random_crop3d(img, mask)
            img = self.img_preprocess(img)
            img = np.transpose(img, (2, 1, 0))
            mask = np.transpose(mask, (2, 1, 0))
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            mask = split_mask(mask).astype('int16')
            return img, mask
        except:
            print(self.df.path[idx])

    @staticmethod
    def img_preprocess(img):
        img = img - img.min()
        if img.max() != 0:
            img = img / img.max()
        return img

    # Cropping functions
    def crop(self, img, p):
        return img[p[0]: p[0] + self.shape[0],
               p[1]: p[1] + self.shape[1],
               p[2]: p[2] + self.shape[2]]

    def random_crop(self, img, mask):
        p = []  # initial points
        for i in range(3):
            p.append(np.random.randint(0, img.shape[i] - self.shape[i]))
        new_img = self.crop(img, p)
        new_mask = self.crop(mask, p)
        return new_img, new_mask

    def random_nonzero_crop(self, img, mask):
        p = np.argwhere(mask == 1)
        if len(p) > 0:
            idx = np.random.randint(0, len(p))
            p = p[idx]
            p = p - (self.shape - np.clip(mask.shape - p, 0, self.shape))
            new_img = self.crop(img, p)
            new_mask = self.crop(mask, p)
        else:
            new_img, new_mask = self.random_crop(img, mask)
        return new_img, new_mask

    def random_crop3d(self, img, mask):
        print(img.shape, mask.shape)
        if type(self.nonzero_prob) is bool:
            nonzero = self.nonzero_prob
        else:
            nonzero = random.random() > self.nonzero_prob
        if nonzero:
            new_img, new_mask = self.random_nonzero_crop(img, mask)
        else:
            new_img, new_mask = self.random_crop(img, mask)
        return new_img, new_mask

    def __len__(self):
        return len(self.df)
