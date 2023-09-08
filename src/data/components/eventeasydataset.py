#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = int(redux)
        self.crop_size = int(crop_size)
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs) - 1


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_type='gaussian', noise_param = 50., seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.root_dir += "preprocessed/"
        root_dir += "preprocessed/"
        self.imgs = os.listdir(root_dir)
        self.imgs.sort()
        # print(self.imgs)
        redux = int(redux)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_type
        self.noise_param = float(noise_param)
        self.seed = int(seed)
        if self.seed:
            np.random.seed(self.seed)


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        target_path = os.path.join(self.root_dir, self.imgs[index + 1])
        # print(img_path)
        img =  Image.open(img_path).convert('RGB')
        target =  Image.open(target_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        source = tvF.to_tensor(img)
        target = tvF.to_tensor(target)

        return source, target