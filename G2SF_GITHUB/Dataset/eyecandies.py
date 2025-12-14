import os
import gc

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T
import glob
from torch.utils.data import Dataset
from .mvtec3d_util import *
import numpy as np
import cv2 as cv

EYECANDIES_CLASS_NAMES = ['CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear',
                          'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy', ]

SIZE = 224
PATCH_SIZE = 56


class BaseDataset(Dataset):
    def __init__(self, args, split):
        self.IMG_MEAN = [0.485, 0.456, 0.406]
        self.IMG_STD = [0.229, 0.224, 0.225]
        self.dataset_path = args.dataset_path
        self.class_name = args.class_name
        self.dir = os.path.join(args.dataset_path, self.class_name, split)
        self.image_transform = T.Compose([T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                                          T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD)])
        self.gt_transform = T.Compose([T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
        self.patch_transform = T.Compose([T.Resize((PATCH_SIZE, PATCH_SIZE), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
        self.morph_size, self.morph_epoch = 3, 3
        self.normalize = T.Compose([T.Normalize(self.IMG_MEAN, self.IMG_STD)])

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class EyecandiesDataset(BaseDataset):
    def __init__(self, args, is_train=True):
        super().__init__(args, split='train' if is_train else 'test')
        self.is_train = is_train
        self.image, self.pcd, self.label, self.mask, self.defect_type = self.load_dataset_folder()

    def load_dataset_folder(self):
        image, pcd, label, mask, types = [], [], [], [], []
        defect_types = sorted(os.listdir(self.dir))
        for defect in defect_types:
            img_dir = os.path.join(self.dir, str(defect), 'rgb')
            pcd_dir = os.path.join(self.dir, str(defect), 'xyz')
            if not os.path.isdir(img_dir):
                continue

            img_fpath_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
            pcd_fpath_list = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith('.tiff')])
            image.extend(img_fpath_list)
            pcd.extend(pcd_fpath_list)

            if defect == 'good':
                label.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                label.extend([1] * len(img_fpath_list))
                gt_dir = os.path.join(self.dir, str(defect), 'gt')
                gt_fpath_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
                mask.extend(gt_fpath_list)
                types.extend([defect] * len(img_fpath_list))

        assert len(image) == len(pcd), 'number of images and point clouds should be same'
        assert len(image) == len(label), 'number of images and labels should be same'
        return list(image), list(pcd), list(label), list(mask), list(types)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img_path, pcd_path, label, mask, defect_type = self.image[idx], self.pcd[idx], self.label[idx], self.mask[idx], self.defect_type[idx]

        img_original = Image.open(img_path).convert('RGB')
        pcd = read_tiff(pcd_path)

        new_img_original = np.zeros_like(img_original)
        nonzero = np.nonzero(np.all(pcd != 0, axis=-1))
        new_img_original[nonzero] = np.asarray(img_original)[nonzero]

        img = self.image_transform(Image.fromarray(new_img_original))

        resized_pcd = resize_organized_pcd(pcd, height=SIZE, width=SIZE)
        resized_pcd = resized_pcd.clone().detach().float()

        if label == 0:
            mask = torch.zeros([1, SIZE, SIZE])
        else:
            mask = Image.open(mask).convert('L')
            mask = self.gt_transform(mask)
            mask = torch.where(mask > 0.5, 1., .0)

        indicator = np.zeros((SIZE, SIZE), dtype=np.uint8)
        nonzero_ = np.nonzero(np.all(resized_pcd.numpy() != 0, axis=0))
        indicator[nonzero_] = 1
        for i in range(self.morph_epoch):
            indicator = cv.dilate(indicator, np.ones((self.morph_size, self.morph_size), np.uint8))
        indicator = self.patch_transform(Image.fromarray(indicator))

        if self.is_train:
            return img, resized_pcd, label, mask, indicator
        else:
            return img, resized_pcd, label, mask, os.path.basename(img_path[:-4]), defect_type, indicator
