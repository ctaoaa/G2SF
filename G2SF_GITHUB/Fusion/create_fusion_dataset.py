import os
import torch
from Util import *
from tqdm import tqdm
import numpy as np
import cv2 as cv
import tifffile as tiff
from .util import feature_interpolation
from torchvision import transforms as T
from collections import defaultdict


class FusionDataCollector(object):
    def __init__(self, args):
        init_seeds(0)

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask_transformer = T.Resize((56, 56), interpolation=T.InterpolationMode.NEAREST)

        self.patch_lib, self.mean, self.std = [{'IMG': [], 'PCD': []} for _ in range(3)]
        self.modality = ['IMG', 'PCD']

        self.scores = {key: [] for key in self.modality}
        self.per_labels = []
        self.idx2coreset = {key: [] for key in self.modality}

        self.feat_dim = (56, 56)
        self.img_feat_dim = (28, 28)
        self.img_list, self.pcd_list, self.mask_list = [], [], []

    @staticmethod
    def get_anomaly_idx(mask, transformer):

        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask = cv.dilate(mask.squeeze().numpy(), kernel)
        new_mask = torch.from_numpy(new_mask).reshape(mask.shape)

        new_mask = transformer(new_mask).squeeze()
        anomaly_idx = torch.nonzero(new_mask.reshape(-1, ))[:, 0]
        normal_idx = torch.where(new_mask.reshape(-1, ) == 0)[0]
        return anomaly_idx, normal_idx

    def collect(self, fm, em, data_loader, remove_bg=True):

        for i, (img, pcd, label, mask, indicator) in tqdm(enumerate(data_loader), desc='Collecting anomaly score maps of pseudo anomaly dataset'):

            self.mask_list.append(t2np(mask.squeeze()))
            self.img_list.append(de_normalizer(t2np(img.squeeze())))
            self.pcd_list.append(t2np(pcd.squeeze().permute(1, 2, 0)))

            anomaly_idx, normal_idx = self.get_anomaly_idx(mask, self.mask_transformer)
            if label[0] == 0:
                assert anomaly_idx.shape[0] == 0, 'Normal sample should have zero anomaly mask'
            else:
                assert anomaly_idx.shape[0] != 0, 'Anomalous sample should have nonzero anomaly mask'

            feature = fm.get_features((img, pcd))
            bg = (feature['PCD'].sum(axis=-1) == 0)

            per_label = torch.zeros((1, self.feat_dim[0] * self.feat_dim[1])).long()
            per_label[0, normal_idx] = 0
            per_label[0, anomaly_idx] = 1
            if remove_bg:
                self.per_labels.append(per_label[0, ~bg])
            else:
                self.per_labels.append(per_label)

            for key in feature.keys():
                feature[key] = (feature[key] - fm.mean[key]) / fm.std[key]
                cur_feature = feature[key]
                if key == 'IMG':
                    cur_feature = feature_interpolation(cur_feature, self.img_feat_dim, self.feat_dim)
                if remove_bg:
                    self.patch_lib[key].append(cur_feature[~bg])
                else:
                    self.patch_lib[key].append(cur_feature)

            for key in self.modality:
                result = em.calculate_scores_per_data(feature, key, fm)
                if remove_bg:
                    self.scores[key].append(torch.flatten(result[2], start_dim=1).reshape(-1, 1)[~bg])
                    self.idx2coreset[key].append(result[3][~bg])
                else:
                    self.scores[key].append(torch.flatten(result[2], start_dim=1).reshape(-1, 1))
                    self.idx2coreset[key].append(result[3])

        """ Concatenation
                """
        for key in self.patch_lib.keys():
            self.patch_lib[key] = torch.cat(self.patch_lib[key], dim=0)

        for key in self.modality:
            self.scores[key] = torch.cat(self.scores[key], dim=0).flatten().reshape(-1, 1)
            self.idx2coreset[key] = torch.cat(self.idx2coreset[key], dim=0)

        self.per_labels = torch.cat(self.per_labels, dim=0).flatten()

        assert self.patch_lib['IMG'].shape[0] == self.scores['IMG'].shape[0]
        assert self.scores['IMG'].shape[0] == self.idx2coreset['IMG'].shape[0]

    def save_dataset(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(self.patch_lib.keys())
        lib_name = os.path.join(save_path, 'patch_lib')
        feature = torch.cat([self.patch_lib['IMG'], self.patch_lib['PCD']], dim=1)
        score_name = os.path.join(save_path, 'score')
        score = torch.cat([self.scores['IMG'], self.scores['PCD']], dim=1)
        per_label_name = os.path.join(save_path, 'per_label')
        idx2coreset = torch.cat([self.idx2coreset['IMG'], self.idx2coreset['PCD']], dim=1)
        idx2coreset_name = os.path.join(save_path, 'idx2coreset')

        print("Size of feature {} and score {}".format(feature.size(), score.size()))

        np.savez_compressed(lib_name, feature=feature.numpy().astype(np.float32))
        # np.save(lib_name, feature.numpy().astype(np.float32))
        np.save(score_name, score.numpy())
        np.save(per_label_name, self.per_labels.numpy())
        np.save(idx2coreset_name, idx2coreset.numpy())

        """ We also want to save images, point clouds and masks for better visualization
        """
        rgb_name = os.path.join(save_path, 'rgb')
        gt_name = os.path.join(save_path, 'gt')
        xyz_name = os.path.join(save_path, 'xyz')

        for name in [rgb_name, gt_name, xyz_name]:
            if not os.path.exists(name):
                os.makedirs(name)

        for i in tqdm(range(len(self.img_list)), desc='Save data used in fusion'):
            cur_img = self.img_list[i]
            cur_img = cv.cvtColor(cur_img, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(rgb_name, str(i) + '.png'), cur_img)

            cur_mask = self.mask_list[i].astype(np.uint8)
            cur_mask[cur_mask == 1] = 255

            cv.imwrite(os.path.join(gt_name, str(i) + '.png'), cur_mask)
            tiff.imwrite(os.path.join(xyz_name, str(i) + '.tiff'), self.pcd_list[i])

        print("Fusion dataset has saved")
