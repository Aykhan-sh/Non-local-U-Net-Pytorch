import numpy as np
from torch.utils.data import Dataset, DataLoader
from trainer.utils import split_mask, open_ct, open_mask


class TailDs(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df

    def __getitem__(self, idx):
        ct = np.load(self.df.iloc[idx, 1])
        mask = np.load(self.df.iloc[idx, 2])
        ct = np.expand_dims(ct, 0)
        mask = split_mask(mask, 7)
        ct = ct.astype('float')
        mask = mask.astype('float')
        return ct, mask

    def __len__(self):
        return len(self.df)


class Ds(Dataset):
    def __init__(self, indexes, shape, transforms=None):
        self.indexes = indexes
        self.shape = shape

    def __getitem__(self, idx):
        ct = open_ct(self.indexes[idx])
        mask = open_mask(self.indexes[idx])
        ct, mask = self.random_crop(ct, mask)
        ct = np.expand_dims(ct, 0)
        mask = split_mask(mask, 7)
        ct = ct.astype('float')
        mask = mask.astype('float')
        return ct, mask

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

    def __len__(self):
        return len(self.indexes)
