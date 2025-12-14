import time

import torch
import numpy as np
import os
from tqdm import tqdm
from Dataset.mvtec3d_util import *
from Model import Model, interpolating_points
from Util import *
from torchvision import transforms as T
from PIL import Image
import pickle
from torch.utils.data import TensorDataset, DataLoader


class Features(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        init_seeds(0)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(device=self.device, rgb_backbone_name=args.rgb_backbone_name,
                                            xyz_backbone_name=args.xyz_backbone_name,
                                            group_size=args.group_size, num_group=args.num_group)
        self.deep_feature_extractor.to(self.device)

        self.args = args
        self.image_size = args.img_size

        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.patch_transform = T.Compose([T.Resize((56, 56), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

        self.n_patch_lib = {'IMG': list(), 'PCD': list()}
        self.mean, self.std, self.coreset_idx = [{'IMG': None, 'PCD': None} for _ in range(3)]
        self.coresets = {'IMG': None, 'PCD': None}

    def get_features(self, sample):
        pcd = sample[1]
        pcd = pcd.squeeze().permute(1, 2, 0).numpy()
        unorganized_pcd = organized_to_unorganized(pcd)
        nonzero_idx = np.nonzero(np.all(unorganized_pcd != 0, axis=1))[0]
        unorganized_pcd = torch.tensor(unorganized_pcd[nonzero_idx, :]).unsqueeze(dim=0).permute(0, 2, 1)
        unorganized_pcd = unorganized_pcd.to(self.device)

        with torch.no_grad():
            rgb_feats, xyz_feats, center, _, _ = self.deep_feature_extractor(sample[0].to(self.device), unorganized_pcd)
            rgb_feats, xyz_feats, center = rgb_feats.detach(), xyz_feats.detach(), center.detach()
            interp_feats = interpolating_points(unorganized_pcd.double(), center.permute(0, 2, 1).double(), xyz_feats.double()).float()

        rgb_feats = [fmap.to("cpu") for fmap in [rgb_feats]]
        feature = dict()

        """ point cloud feature patch
        """
        xyz_patch_full = torch.zeros((1, interp_feats.shape[1], self.image_size * self.image_size), device=self.device)
        xyz_patch_full[:, :, nonzero_idx] = interp_feats
        xyz_patch_full = xyz_patch_full.view(1, interp_feats.shape[1], self.image_size, self.image_size)

        with torch.no_grad():
            xyz_patch_full = self.resize(self.average(xyz_patch_full)).detach().cpu()

        feature['PCD'] = xyz_patch_full.reshape(xyz_patch_full.shape[1], -1).T

        """ image feature patch
        """
        rgb_patch = torch.cat(rgb_feats, 1)

        rgb_patch_resize = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        feature['IMG'] = rgb_patch_resize.cpu()
        torch.cuda.empty_cache()
        return feature

    def construct_memory_bank(self, dataloader, normalize=True):
        for i, (data) in tqdm(enumerate(dataloader), desc='Extract features from dataloader'):
            img, pcd, label, mask, indicator = data
            feature = self.get_features((img, pcd))

            for key in self.n_patch_lib.keys():
                self.n_patch_lib[key].append(feature[key])

        print("Feature extraction has done")

        for key in self.n_patch_lib.keys():
            self.n_patch_lib[key] = torch.cat(self.n_patch_lib[key], 0)

        if normalize:
            self.normalize_memory_bank()

    def normalize_memory_bank(self):
        for key in self.n_patch_lib.keys():
            self.mean[key], self.std[key] = torch.mean(self.n_patch_lib[key]), torch.std(self.n_patch_lib[key])
            """ Following Wang et.al. CVPR2023, we do not normalize the point cloud feature
            """
            if key == 'PCD':
                self.mean[key], self.std[key] = torch.Tensor([0.]), torch.Tensor([1.])
            self.n_patch_lib[key] = (self.n_patch_lib[key] - self.mean[key]) / self.std[key]

    def get_coreset(self):
        for key in self.n_patch_lib.keys():
            print("Run coreset of {}".format(key))
            num_lib = self.n_patch_lib[key].shape[0]
            if self.args.coreset_mode == 'sparse':
                self.coreset_idx[key] = get_coreset_idx(self.n_patch_lib[key], n=int(self.args.f_coreset * num_lib),
                                                        eps=self.args.coreset_eps, print_=True)
            elif self.args.coreset_mode == 'nn':
                self.coreset_idx[key] = get_coreset_idx_nn(self.n_patch_lib[key], project_dim=self.args.proj_dim,
                                                           n=int(self.args.f_coreset * num_lib), print_=True)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        n_lib_name = os.path.join(save_path, 'n_patch_lib')
        np.savez_compressed(n_lib_name, IMG=self.n_patch_lib['IMG'], PCD=self.n_patch_lib['PCD'])

        coreset_idx_name = os.path.join(save_path, 'coreset_idx.pickle')
        with open(coreset_idx_name, 'wb') as file:
            pickle.dump(self.coreset_idx, file)
            file.close()

        mean_name = os.path.join(save_path, 'mean.pickle')
        with open(mean_name, 'wb') as file:
            pickle.dump(self.mean, file)
            file.close()

        std_name = os.path.join(save_path, 'std.pickle')
        with open(std_name, 'wb') as file:
            pickle.dump(self.std, file)
            file.close()

        print("Features have saved")

    def load_model(self, save_path, is_patch_lib=True):
        n_lib_name = os.path.join(save_path, 'n_patch_lib.npz')
        coreset_idx_name = os.path.join(save_path, 'coreset_idx.pickle')
        mean_name = os.path.join(save_path, 'mean.pickle')
        std_name = os.path.join(save_path, 'std.pickle')

        n_lib = np.load(n_lib_name)

        for key in self.n_patch_lib.keys():
            self.n_patch_lib[key] = torch.from_numpy(n_lib[key]).float()

        with open(coreset_idx_name, 'rb') as file:
            self.coreset_idx = pickle.load(file)

        with open(mean_name, 'rb') as file:
            self.mean = pickle.load(file)

        with open(std_name, 'rb') as file:
            self.std = pickle.load(file)

        for key in self.coresets.keys():
            self.coresets[key] = self.n_patch_lib[key][self.coreset_idx[key]]

        if not is_patch_lib:
            self.n_patch_lib = {'IMG': list(), 'PCD': list()}
