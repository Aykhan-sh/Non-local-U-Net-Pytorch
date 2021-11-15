import shutil
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from tqdm.notebook import tqdm
import cv2

# masks
path = 'data/labels and README'
mask_path = 'data/processed/masks'
images_path = 'data/processed/images'
os.makedirs(mask_path, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
labels = os.listdir(path)
labels = list(filter(lambda l: l.startswith('label'), labels))
df = []
for i in tqdm(range(len(labels))):
    vol = labels[i][7:-7]
    ct = sitk.ReadImage(os.path.join(path, labels[i]))
    ct = sitk.GetArrayFromImage(ct)
    for slc_idx, slc in enumerate(ct):
        label, counts = np.unique(slc, return_counts=True)
        pixel_count = np.zeros(7)
        pixel_count[list(label.astype('int'))] = counts
        pixel_count = list(pixel_count) + [vol, slc_idx]
        df.append(pixel_count)
        if (label == [0]).all():
            pass
        else:
            cv2.imwrite(f'data/processed/masks/{vol}_{slc_idx}.png', slc)
df = pd.DataFrame(df, columns=['Background', 'Liver', 'Bladder', 'Lungs', 'Kidneys', 'Bone', 'Brain', 'volume', 'slice'])
df = df[['volume', 'slice', 'Background', 'Liver', 'Bladder', 'Lungs', 'Kidneys', 'Bone', 'Brain']]
df = df.sort_values(by=['volume', 'slice']).reset_index(drop=True)
df.to_csv('data/df.csv', index=False)

# remove README
os.rename('data/raw data/labels and README', 'data/raw data/masks')
shutil.move('data/raw data/masks/README.txt', 'data/raw data/README.txt')

# images
os.makedirs('data/raw data/volumes', exist_ok=True)
for path in ['data/raw data/volumes 0-49', 'data/raw data/volumes 100-139', 'data/raw data/volumes 50-99']:
    vols = os.listdir(path)
    for vol in vols:
        shutil.move(os.path.join(path, vol), os.path.join('data/raw data/volumes', vol))
    os.remove(path)