import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from Util import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


class Validator(object):
    def __init__(self, args):
        init_seeds(0)

        self.args = args
        self.device = args.device
        self.feat_dim = (56, 56)
        self.img_feat_dim = (28, 28)
        self.size = args.img_size
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3

        """ Default: all
        """
        self.modality = ['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD']
        self.evaluate_modality = ['IMG', 'PCD']

        self.scores, self.seg_scores, self.seg_scores_atr, self.aucs, self.seg_aucs, self.aupros = [{key: [] for key in self.modality} for _ in range(6)]
        self.image_list, self.gt_label_list, self.file_name_list = [], [], []
        self.fg_seg_scores, self.fg_masks = [], []

        self.gt_mask_list = []

    def evaluate(self, fm, data_loader, fuser=None,  remove_bg=True):
        running_time = 0
        num = 0

        for (img, pcd, label, mask, name, defect_type, indicator) in tqdm(data_loader, desc='Evaluate anomaly score'):
            num += 1
            self.gt_label_list.append(t2np(label[0]))
            self.gt_mask_list.append(t2np(mask.squeeze()))
            self.image_list.append(t2np(img.squeeze()))
            file_name = os.path.join(defect_type[0], name[0])
            self.file_name_list.append(file_name)

            xyz = pcd.squeeze().permute(1, 2, 0).numpy()
            xyz_mask = (xyz.sum(axis=-1) == 0)
            self.fg_masks.append(t2np(mask.squeeze())[~xyz_mask])

            start_time = time.perf_counter()

            feature = fm.get_features((img, pcd))
            feature_level_scores = {key: None for key in self.evaluate_modality}

            bg_mask = (feature['PCD'].sum(axis=-1) == 0)

            for key in feature.keys():
                feature[key] = (feature[key] - fm.mean[key]) / fm.std[key]

            """ Single modalities and cross modalities
            """
            for key in self.evaluate_modality:
                result = self.calculate_scores_per_data(feature, key, fm)
                self.scores[key].append(result[0])
                self.seg_scores[key].append(result[1])

                feature_level_scores[key] = torch.flatten(result[2], start_dim=1)

            """ Start fusion
            """
            img_feature = self.feature_interpolation(feature['IMG'], self.img_feat_dim, self.feat_dim)
            concat_feature = torch.cat([img_feature, feature['PCD']], dim=1)
            concat_score = torch.cat(list(feature_level_scores.values()), dim=0).transpose(1, 0)

            """ Supervised fusion
            """
            if 'FUS' in self.modality:
                self.compute_supervised_fusion_score(concat_feature, concat_score, bg_mask, fuser, remove_bg, xyz_mask)
            running_time += (time.perf_counter() - start_time)

        """ Normalize scores into [0,1] interval
        """

        print("Average time for processing one data: ", running_time / num)

        for key in self.modality:
            self.scores[key] = torch.cat(self.scores[key], dim=0).numpy()

            self.seg_scores[key] = torch.cat(self.seg_scores[key], dim=0).numpy()
            self.seg_scores[key] = np.expand_dims(self.seg_scores[key], axis=1)
            self.seg_scores_atr[key] = (self.seg_scores[key].min(), self.seg_scores[key].max())
            self.seg_scores[key] = (self.seg_scores[key] - self.seg_scores_atr[key][0]) / (self.seg_scores_atr[key][1] - self.seg_scores_atr[key][0])

        result = self.calculate_metrics()
        return result


    def compute_supervised_fusion_score(self, concat_feature, concat_score, bg_mask, fuser, remove_bg, xyz_mask):
        res = fuser.predict_with_bg(concat_feature, concat_score, bg_mask)
        fus_scores = {'FUS': res[0], 'WEI_IMG': res[1], 'WEI_PCD': res[2], 'ALL': res[3]}

        for key in ['FUS', 'WEI_IMG', 'WEI_PCD']:
            """ 
            """
            self.scores[key].append(fus_scores[key].max().view(1, ))
            fus_seg_score = self.postprocess(fus_scores[key], self.feat_dim)[0]
            fus_seg_score = fus_seg_score.unsqueeze(0)
            self.seg_scores[key].append(fus_seg_score)

            if key == 'FUS':
                self.fg_seg_scores.append(fus_seg_score.squeeze().numpy()[~xyz_mask])

    @staticmethod
    def feature_interpolation(feature, source_size, target_size):
        dim = feature.shape[1]
        interp_feat = feature.reshape(*source_size, dim)
        interp_feat = F.interpolate(interp_feat.permute(2, 0, 1).unsqueeze(0), size=target_size, mode='nearest')
        interp_feat = interp_feat.squeeze().permute(1, 2, 0).reshape(-1, dim)
        return interp_feat

    def calculate_scores_per_data(self, feature, key, fm):
        if key == 'PCD':
            result = self.single_score_map(feature['PCD'], fm.coresets['PCD'])
        else:
            interp_img_feat = self.feature_interpolation(feature['IMG'], self.img_feat_dim, self.feat_dim)
            if key == 'IMG':
                result = self.single_score_map(interp_img_feat, fm.coresets['IMG'])
            else:
                raise NotImplementedError
        return result

    def single_score_map(self, feature, coreset):
        feature = feature.to(self.args.device)
        coreset = coreset.to(self.args.device)

        s, min_val, min_idx = self.sample_score(feature, coreset)
        s_map, s_map0 = self.postprocess(min_val, self.feat_dim)
        return s.cpu().view(1, ), s_map.unsqueeze(0), s_map0.unsqueeze(0), min_idx.cpu()

    def sample_score(self, feature, coreset):
        dist = torch.cdist(feature, coreset)

        _, anchor_idx = torch.topk(dist, dim=1, k=20, largest=False)
        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        m_test = feature[s_idx].unsqueeze(0)
        m_star = coreset[min_idx[s_idx]].unsqueeze(0)
        w_dist = torch.cdist(m_star, coreset)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)
        m_star_knn = torch.linalg.norm(m_test - coreset[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(feature.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star
        anchor_idx[:, 0] = min_idx
        return s, min_val, anchor_idx

    """ interp_s_map:
    """

    def postprocess(self, score, feat_dim):
        s_map = score.flatten()
        s_map = s_map.view(1, 1, *feat_dim)
        interp_s_map = torch.nn.functional.interpolate(s_map.cpu(), size=(224, 224), mode='bilinear')
        interp_s_map = self.blur(interp_s_map)
        return interp_s_map.squeeze(), s_map.cpu().squeeze()

    def calculate_metrics(self):
        gt_label = np.asarray(self.gt_label_list)
        gt_mask = np.asarray(self.gt_mask_list)
        auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05 = [], [], [], [], [], []

        self.fg_masks = [m.reshape(-1, 1) for m in self.fg_masks]
        self.fg_masks = np.vstack(self.fg_masks).flatten()
        self.fg_seg_scores = [m.reshape(-1, 1) for m in self.fg_seg_scores]
        self.fg_seg_scores = np.vstack(self.fg_seg_scores).flatten()

        assert self.fg_masks.shape[0] == self.fg_seg_scores.shape[0]
        np.save(os.path.join(self.args.save_path, self.args.class_name + '_score.npy'), self.fg_seg_scores)
        np.save(os.path.join(self.args.save_path, self.args.class_name + '_gt.npy'), self.fg_masks)

        for key in self.modality:
            self.aucs[key] = roc_auc_score(gt_label, self.scores[key].flatten())
            self.seg_aucs[key] = roc_auc_score(gt_mask.flatten(), self.seg_scores[key].flatten())
            self.aupros[key], _ = calculate_au_pro(gt_mask, self.seg_scores[key][:, 0, :, :])
            print("Modal {}:  sample-level auc {:.3f}, pixel-level auc {:.3f}, "
                  "aupro {:.3f}, aupro_0p01 {:.3f}, aupro_0p1 {:.3f}, aupro_0p05 {:.3f}".format(key, self.aucs[key], self.seg_aucs[key],
                                                                                                self.aupros[key][0], self.aupros[key][-1],
                                                                                                self.aupros[key][1], self.aupros[key][2]))
            auc.append(self.aucs[key])
            seg_auc.append(self.seg_aucs[key])
            aupro.append(self.aupros[key][0])  # aupro 30%
            aupro_0p01.append(self.aupros[key][-1])
            aupro_0p1.append(self.aupros[key][1])
            aupro_0p05.append(self.aupros[key][2])

        auc, seg_auc, aupro, aupro_0p01 = np.array(auc), np.array(seg_auc), np.array(aupro), np.array(aupro_0p01)
        aupro_0p1, aupro_0p05 = np.array(aupro_0p1), np.array(aupro_0p05)
        return np.round(auc, decimals=4), np.round(seg_auc, decimals=4), np.round(aupro, decimals=4), \
            np.round(aupro_0p01, decimals=4), np.round(aupro_0p1, decimals=4), np.round(aupro_0p05, decimals=4)

    def save_fig(self, save_path):
        for i in tqdm(range(len(self.file_name_list)), desc='Save figure'):
            path, name = os.path.split(self.file_name_list[i])
            path = os.path.join(save_path, path)
            if not os.path.exists(path):
                os.makedirs(path)
            imgs, names = [self.image_list[i], self.gt_mask_list[i]], [self.file_name_list[i], 'MASK']
            for key in self.modality:
                imgs.append(self.seg_scores[key][i])
                names.append(key)
            plot_fig(imgs, names, self.file_name_list[i], show=False, save_name=name + '.jpg', save_path=path)
