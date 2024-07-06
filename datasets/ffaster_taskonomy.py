import torch.utils.data as data

from PIL import Image, ImageOps
import os
import os.path
import zipfile as zf
import io
import logging
import random
import copy
import numpy as np
import time
import torch
import warnings
# import torchvision.transforms as transforms
import pickle

from multiprocessing import Manager
import tqdm
import time
from torchvision.transforms import v2
from .datasets import register
import glob

@register("ffaster_taskonomy")
class FFasterTaskonomy(data.Dataset):
    def __init__(self,
                 data_dir,
                 label_set=None,
                 model_whitelist=None,
                 model_limit=None,
                 return_filename=False,
                 augment=False,
                 **kwargs):
        super().__init__()
        self.root = data_dir
        if model_limit is None or (isinstance(model_limit, str) and len(model_limit) == 0):
            self.model_limit = None
        else:
            self.model_limit = model_limit
        self.records = []
        if model_whitelist is None:
            self.model_whitelist = None
        else:
            self.model_whitelist = set()
            with open(model_whitelist) as f:
                for line in f:
                    self.model_whitelist.add(line.strip())

        preprocessing_begin_time = time.time()
        if self.model_limit is not None:
            counter = {model: 0 for model in self.model_whitelist}
        pkl_files = os.listdir(self.root)
        for f in pkl_files:
            model = f.split("_")[0]
            if self.model_whitelist is not None and model in self.model_whitelist:
                if self.model_limit is not None and counter[model] < self.model_limit:
                    self.records.append(os.path.join(self.root, f))
                    counter[model] += 1
                elif self.model_limit is None:
                    self.records.append(os.path.join(self.root, f))
            elif self.model_whitelist is None:
                self.records.append(os.path.join(self.root, f))
        preprocessing_end_time = time.time()
        print(f"preprocessing time is: {preprocessing_end_time - preprocessing_begin_time}")

        self.label_set = label_set if label_set is not None else kwargs["tasks"]
        self.return_filename = return_filename
        self.augment = augment

        if augment == "aggressive":
            print('Data augmentation is on (aggressive).')
        elif augment:
            print('Data augmentation is on (flip).')
        else:
            print('no data augmentation')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is an uint8 matrix of integers with the same width and height.
        If there is an error loading an image or its labels, return None
        """
        file_name = self.records[index]
        flip_lr = (random.randint(0, 1) > .5 and self.augment)
        flip_ud = (random.randint(0, 1) > .5 and (self.augment == "aggressive"))

        to_load = ['rgb'] + self.label_set
        if len(set(['edge_occlusion', 'normal', 'reshading', 'principal_curvature']).intersection(
                self.label_set)) != 0:
            to_load.append('mask')

        flip_fun = []
        if flip_lr: flip_fun.append(v2.functional.horizontal_flip)
        if flip_ud: flip_fun.append(v2.functional.vertical_flip)

        with open(file_name, "rb") as fr:
            ys = pickle.load(fr)
        for key in ys.keys():
            tmp = torch.from_numpy(ys[key])
            if len(flip_fun) > 0:
                for aug in flip_fun:
                    tmp = aug(tmp)
            ys[key] = tmp

        if "normal" in to_load:
            if flip_lr:
                ys["normal"][0] *= -1.0
            if flip_ud:
                ys["normal"][1] *= -1.0

        ys['inp'] = ys.pop('rgb')

        if self.return_filename:
            return ys, file_name
        else:
            return ys

    def __len__(self):
        return (len(self.records))
