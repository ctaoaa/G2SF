from .mvtec3d import BaseDataset
from PIL import Image
from torchvision import transforms as T
from .mvtec3d_util import *
import numpy as np
import cv2 as cv
import os
import random
import albumentations as A
import matplotlib.pyplot as plt
from .perlin import rand_perlin_2d_np
import imgaug.augmenters as iaa
from scipy import ndimage
from Util import *
from .cut_paste import cut_paste_scc
import json
from .mvtec3d_util import pad_crop

SIZE = 224
PATCH_SIZE = 56


def normal_augmentation(class_name):
    if class_name in ['dowel', 'tire', 'foam', 'rope', 'cable_gland']:
        augmentor = T.Compose([T.RandomRotation(degrees=(-10, 10)), T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=0)])
    else:
        augmentor = T.Compose([T.RandomRotation(degrees=(-45, 45)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                               T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.25), shear=0)])
    return augmentor


class MVTecPseudoDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args, split='train')
        self.args = args
        self.num_anomalies = args.num_anomalies
        self.max_component = 1600
        self.min_component = 100
        source_file_name = os.path.join(args.source_path, 'anomaly_source_path.json')
        with open(source_file_name, 'r') as load_f:
            self.anomaly_source = json.load(load_f)

        """ We only load the train data, so defect type is avoided
        """
        self.n_imgs, self.n_pcds, self.n_labels, self.n_masks, _ = self.load_dataset_folder()

        self.a_labels = [1 for _ in range(self.num_anomalies)]
        self.labels = np.array(self.n_labels + self.a_labels)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.anomaly_idx = np.argwhere(self.labels == 1).flatten()

        self.normal_augmentor = normal_augmentation(args.class_name)
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        self.augmenters = [iaa.AverageBlur(k=(50, 50)), iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.7), add=(-50, 50)),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.pillike.Autocontrast(), iaa.ElasticTransformation(alpha=(2.5, 7.5), sigma=5),
                           iaa.Invert(0.05, per_channel=True), iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
                                                                           saturation=2.0, edge_prevalence=0.01)]

    def find_patch_indicator(self, resized_pcd):
        indicator = np.zeros((SIZE, SIZE), dtype=np.uint8)
        nonzero_ = np.nonzero(np.all(resized_pcd.numpy() != 0, axis=0))
        indicator[nonzero_] = 1
        for i in range(self.morph_epoch):
            indicator = cv.dilate(indicator, np.ones((self.morph_size, self.morph_size), np.uint8))
        indicator = self.patch_transform(Image.fromarray(indicator))
        return indicator

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        return aug

    def __len__(self):
        return self.args.num_normal + self.args.num_anomalies

    def __getitem__(self, idx):
        if idx >= self.args.num_normal:
            idx_ = np.random.randint(len(self.n_imgs))
            img_path, pcd_path = self.n_imgs[idx_], self.n_pcds[idx_]
            img, pcd, mask = self.generate_pseudo_anomaly(img_path, pcd_path)

            img, mask = Image.fromarray(img), Image.fromarray(mask)
            label = 1
            img = self.image_transform(img)
            resized_pcd = resize_organized_pcd(pcd, height=SIZE, width=SIZE)
            resized_pcd = resized_pcd.clone().detach().float()
            mask = self.gt_transform(mask)
            mask = torch.where(mask > 0.5, 1., .0)
            indicator = self.find_patch_indicator(resized_pcd)
            return img, resized_pcd, label, mask, indicator
        else:
            idx_ = idx % len(self.n_imgs)
            img_path, pcd_path, label, mask_path = self.n_imgs[idx_], self.n_pcds[idx_], self.n_labels[idx_], self.n_masks[idx_]

        img_original = cv.imread(img_path)
        img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
        pcd = read_tiff(pcd_path)
        img_original, pcd, _ = self.augment_normal_data(img_original, pcd, np.zeros_like(img_original))

        img = self.image_transform(Image.fromarray(img_original))

        resized_pcd = resize_organized_pcd(pcd, height=SIZE, width=SIZE)
        resized_pcd = resized_pcd.clone().detach().float()
        mask = torch.zeros([1, SIZE, SIZE])

        indicator = self.find_patch_indicator(resized_pcd)
        return img, resized_pcd, label, mask, indicator

    @staticmethod
    def get_foreground_mask(image):
        fg_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        nonzero_ = np.nonzero(np.all(image != 0, axis=-1))
        fg_mask[nonzero_] = 255
        return fg_mask

    def restrict_component(self, mask):
        labels, n_subs = ndimage.label(mask)
        new_mask = np.zeros_like(mask)
        for i in range(1, n_subs):
            component = (labels == i).astype(np.uint8)
            if self.min_component <= np.sum(component) <= self.max_component:
                new_mask[component == 1] = 1
        return new_mask

    def random_perlin_mask(self, fg_mask):

        resize_fg_mask = cv.resize(fg_mask, (512, 512), interpolation=cv.INTER_NEAREST)

        perlin_scale = 5
        min_perlin_scale = 0
        while True:
            perlin_scale_x = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scale_y = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((512, 512), (perlin_scale_x, perlin_scale_y))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.4
            perlin_mask = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

            intersect_mask = np.logical_and(resize_fg_mask == 255, perlin_mask == 1).astype(np.uint8)
            intersect_mask = self.restrict_component(intersect_mask)

            if np.sum(intersect_mask) >= 300:
                break
        intersect_mask = cv.resize(intersect_mask, fg_mask.shape, interpolation=cv.INTER_NEAREST)
        assert intersect_mask.max() == 1

        intersect_mask[intersect_mask == 1] = 255
        return intersect_mask

    def augment_normal_data(self, image, pcd, pcd_full):

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        pcd = torch.from_numpy(pcd).permute(2, 0, 1)
        pcd_full = torch.from_numpy(pcd_full).permute(2, 0, 1)
        data = torch.cat([image, pcd, pcd_full])
        aug_data = self.normal_augmentor(data)
        aug_image = aug_data[:3].numpy().transpose(1, 2, 0).astype(np.uint8)

        aug_pcd = aug_data[3:6].numpy().transpose(1, 2, 0)
        aug_pcd_full = aug_data[6:].numpy().transpose(1, 2, 0)
        return aug_image, aug_pcd, aug_pcd_full

    def generate_pseudo_anomaly(self, img_path, pcd_path):

        n_img = cv.imread(img_path)
        n_img = cv.cvtColor(n_img, cv.COLOR_BGR2RGB)
        n_pcd = read_tiff(pcd_path)
        full_pcd_path = str(pcd_path).replace(self.args.dataset_path, self.args.full_dataset_path)
        n_pcd_full = read_tiff(full_pcd_path)
        n_pcd_full = pad_crop(n_pcd_full)

        """ Augmentation
        """
        n_img, n_pcd, n_pcd_full = self.augment_normal_data(n_img, n_pcd, n_pcd_full)
        n_fg_mask = self.get_foreground_mask(n_pcd)

        while True:
            a_idx = np.random.randint(len(self.n_pcds))
            a_img_path = self.n_imgs[a_idx]
            if a_img_path != img_path:
                break

        a_pcd = read_tiff(self.n_pcds[a_idx])
        a_img = cv.imread(a_img_path)
        a_img = cv.cvtColor(a_img, cv.COLOR_BGR2RGB)

        aug = self.randAugmenter()
        a_img_aug = aug(image=a_img)

        a_fg_mask = self.get_foreground_mask(a_pcd)

        source_idx = np.random.randint(len(self.anomaly_source))
        pcd_source = read_tiff(self.anomaly_source[source_idx])
        pcd_source = resize_organized_pcd(pcd_source, height=a_pcd.shape[0], width=a_pcd.shape[1]).permute(1, 2, 0).numpy()

        source_mask = self.get_foreground_mask(pcd_source)

        a_fg_mask = np.logical_and(a_fg_mask == 255, source_mask == 255).astype(np.uint8)
        a_fg_mask[a_fg_mask == 1] = 255

        a_mask = self.random_perlin_mask(a_fg_mask)
        kernel = np.ones((3, 3), np.uint8)
        a_mask = cv.dilate(a_mask, kernel)

        labels, n_subs = ndimage.label(a_mask)
        final_mask = np.zeros_like(a_mask)
        """ For each iteration, we only add one connected component in the source a_mask
        """
        for i in range(1, n_subs + 1):
            sub_mask = (labels == i).astype(np.uint8)
            if np.sum(sub_mask) <= self.min_component:
                continue
            else:
                sub_mask[sub_mask == 1] = 255
                n_img, n_pcd, cur_mask = cut_paste_scc(n_img, n_pcd, n_pcd_full, n_fg_mask, sub_mask, a_img_aug, pcd_source)
                final_mask[cur_mask == 255] = 255
        return n_img, n_pcd, final_mask

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
