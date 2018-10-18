import ipdb
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import h5py
import random
import torchvision.transforms as transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 2)


def get_transform():
    transform_list = []
    osize = [64, 64]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    return transforms.Compose(transform_list)

if 'DOCKER' in os.environ.keys() and os.environ['DOCKER'] == "1":
    DOCKERMODE = True
    data_dir = '/grasping_deltaaction_targeteep_h5rec/'
else:
    data_dir = '/home/dineshj/Documents/Data/frederik_data/grasping_deltaaction_targeteep_h5rec/'

class picker(data.Dataset):
    def __init__(self,
                 phase,
                 limit_files=-1,
                 ):
        self.data_dir = data_dir + '/train/'  # 'directory containing data_files.' ,
        self.train_val_split = 0.95
        self.video_transform = get_transform()

        self.filenames = sorted(glob.glob(self.data_dir + '*'))
        if not self.filenames:
            raise RuntimeError('No filenames found')

        random.seed(1)
        random.shuffle(self.filenames)
        if limit_files > 0 and limit_files < len(self.filenames):
            self.filenames = self.filenames[:limit_files]

        index = int(np.floor(self.train_val_split * len(self.filenames)))
        if phase == 'train':
            self.filenames = self.filenames[:index]
        elif phase == 'val':
            self.filenames = self.filenames[index:]

    def __getitem__(self, index):
        path = self.filenames[index]

        tg = []
        with h5py.File(path, 'r') as F:
            for t in range(15):
                a = '{}/image_view0/encoded'.format(int(t))
                tg.append(Image.fromarray(F[a][:]))
                if self.video_transform is not None:
                    tg[-1] = self.video_transform(tg[-1])
                # convert to NCHW tg[-1]
                tg[-1] = np.transpose(tg[-1], (2, 0, 1))

        tg = np.stack(tg, axis=0)/255.

        return tg

    def __len__(self):
        return len(self.filenames)
