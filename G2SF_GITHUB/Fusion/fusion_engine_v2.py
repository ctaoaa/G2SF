import torch
import numpy as np
from Util import *
import os
from .util import get_dataloader
from Model import MultiModalGateNet
import lightning as L
from torch import optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from .loss import WDSADMarginLoss, l2_regularization, l1_regularization
import torch.nn.functional as F
from collections import defaultdict
from itertools import chain
import math
import shutil
import time


class FusionModelV2(object):
    def __init__(self, args, fm, verbose=False, is_train=True):
        init_seeds(0)
        torch.set_float32_matmul_precision('high')

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = args.fusion_dataset_path
        print('current path: ', self.path)
        print('complete path: ', self.args.complete_fusion_dataset_path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.paras = defaultdict(float)
        self.paras['svdd'] = 1.
        for key in ['l1', 'l2', 'margin', 'perm', 'magnitude', 'anchor', 'const', 'scaling']:
            self.paras[key] = getattr(args, key)
        for key in ['lr', 'max_epoch', 'alpha', 'perc']:
            setattr(self, key, getattr(args, key))

        self.coresets = {key: fm.coresets[key].to(self.device) for key in fm.coresets}
        self.print_interval = args.print_interval
        self.net = MultiModalGateNet(args, n_hidden=args.n_hidden).to(self.device)
        self.center = torch.tensor([0.0]).to(self.device)

        m = 5.0
        if is_train:
            self.train_loader, self.test_loader = get_dataloader(self.args.complete_fusion_dataset_path, args.fusion_batch_size, args.fusion_test_batch_size)
            self.score_mean = self.normalize_score(self.train_loader).to(self.device)
            m = self.adaptive_margin(self.train_loader)

            self.criterion = WDSADMarginLoss(self.center, 2 * m, percentile=self.perc, alpha=self.alpha).to(self.device)

            print("Mean {:.2f} {:.2f}, Margin {:.2f}".format(self.score_mean[0].item(), self.score_mean[1].item(), m))
            np.save(os.path.join(self.path, 'score_mean'), self.score_mean.cpu().numpy())
        else:
            self.score_mean = np.load(os.path.join(self.path, 'score_mean.npy'))
            self.score_mean = torch.from_numpy(self.score_mean).to(self.device)

            self.global_scale = np.load(os.path.join(self.path, 'global_scale.npy'))
            self.global_scale = torch.from_numpy(self.global_scale).to(self.device)

            self.net.load_state_dict(torch.load(os.path.join(self.path, 'MMScoringTrainerV2.pth')))

        if verbose:
            print(self.net)

    def train(self):
        model = MMScoringTrainer(self.paras, self.score_mean, self.coresets, self.net, self.criterion, self.lr,
                                 self.max_epoch, self.print_interval, self.path)
        trainer = L.Trainer(max_epochs=self.max_epoch, enable_progress_bar=False, enable_checkpointing=False, logger=False, check_val_every_n_epoch=1)
        trainer.fit(model, self.train_loader, self.test_loader)

    @staticmethod
    def initialize_weights_to_zero(model):
        for param in model.parameters():
            torch.nn.init.constant_(param, 0)

    @staticmethod
    def normalize_score(dataloader):
        scores = []
        with torch.no_grad():
            for _, _, s, l, _ in dataloader:
                scores.append(torch.mean(s[l == 1], dim=0).unsqueeze(0))
        score_mean = torch.cat(scores, dim=0)
        return torch.mean(score_mean, dim=0)

    def adaptive_margin(self, dataloader):
        max_score = []
        score_mean = self.score_mean.cpu()
        with torch.no_grad():
            for _, _, s, l, _ in dataloader:
                s = s / score_mean.view(1, 2)
                max_score.append(torch.max(s[l == -1]).item())
        return torch.tensor(max_score).max().item()

    def find_idx(self, feature, coreset):
        dist = torch.cdist(feature, coreset)
        min_dist, anchor_idx = torch.topk(dist, dim=1, k=int(self.paras['anchor'] + 1), largest=False)
        core_feats, scores = [], []
        for i in range(anchor_idx.shape[1]):
            core_feats.append(coreset[anchor_idx[:, i]])
            scores.append(F.pairwise_distance(feature, coreset[anchor_idx[:, i]]).view(-1, 1))

        return core_feats, scores

    def predict(self, feature, score):
        assert feature.shape[1] == 1920
        self.net.eval()
        feature, score = feature.to(self.device), score.to(self.device)

        x, y = feature[:, :768], feature[:, 768:]
        core_xs, sx = self.find_idx(x, self.coresets['IMG'])
        core_ys, sy = self.find_idx(y, self.coresets['PCD'])

        feat = torch.cat([x, y], dim=1)
        cores = [torch.cat([core_x, core_y], dim=1) for core_x, core_y in zip(core_xs, core_ys)]
        scores = [torch.cat([s_x, s_y], dim=1) for s_x, s_y in zip(sx, sy)]
        scores = [score / self.score_mean.unsqueeze(0) for score in scores]
        global_scale = get_scale(self.global_scale)

        """ We consider more local distances
        """
        preds = []
        for i, (core_feat, score) in enumerate(zip(cores, scores)):
            with torch.no_grad():
                norm = F.normalize(feat - core_feat, dim=1, eps=1e-3) * 5  # 保证数值不会很小
                pred = self.net(norm, core_feat)
                if i == 0:
                    weight_img, weight_pcd = pred[:, 0], pred[:, 1]
                pred = pred * global_scale
                pred = torch.sum(pred * score, dim=1).detach().cpu()
                preds.append(pred.view(-1, 1))

        preds = torch.cat(preds, dim=1)
        pred, _ = torch.min(preds, dim=1)

        return pred, weight_img.detach().cpu(), weight_pcd.detach().cpu(), preds

    def predict_with_bg(self, feature, score, bg_mask):

        self.net.eval()
        feature, score = feature.to(self.device), score.to(self.device)

        x, y = feature[:, :768], feature[:, 768:]
        core_xs, sx = self.find_idx(x, self.coresets['IMG'])
        core_ys, sy = self.find_idx(y, self.coresets['PCD'])
        feat = torch.cat([x, y], dim=1)
        cores = [torch.cat([core_x, core_y], dim=1) for core_x, core_y in zip(core_xs, core_ys)]
        scores = [torch.cat([s_x, s_y], dim=1) for s_x, s_y in zip(sx, sy)]

        scores = [score / self.score_mean.unsqueeze(0) for score in scores]
        global_scale = get_scale(self.global_scale)

        preds = []
        for i, (core_feat, score) in enumerate(zip(cores, scores)):
            with torch.no_grad():
                norm = F.normalize(feat - core_feat, dim=1, eps=1e-3) * 5  # 保证数值不会很小
                pred = self.net(norm, core_feat)
                pred[bg_mask] = 1

                if i == 0:
                    weight_img, weight_pcd = pred[:, 0].detach().cpu().clone(), pred[:, 1].detach().cpu().clone()

                pred = pred * global_scale
                pred = torch.sum(pred * score, dim=1).detach().cpu().clone()
                preds.append(pred.view(-1, 1))

        preds = torch.cat(preds, dim=1)
        if self.args.scoring_mode == 'min':
            pred, _ = torch.min(preds, dim=1)
        elif self.args.scoring_mode == 'max':
            pred, _ = torch.max(preds, dim=1)
        elif self.args.scoring_mode == 'average':
            pred = torch.mean(preds, dim=1)
        elif self.args.scoring_mode == 'zero':
            pred = preds[:, 0]

        return pred, weight_img.detach().cpu(), weight_pcd.detach().cpu(), preds


def get_scale(scale):
    scale = 0.5 * (torch.tanh(scale) + 1).view(1, 2)
    scale = scale / torch.sum(scale)
    return scale


def __normalize__(feat, core_feat):
    norm_x = F.normalize(feat[:, :768] - core_feat[:, :768], dim=1, eps=1e-5)
    norm_y = F.normalize(feat[:, 768:] - core_feat[:, 768:], dim=1, eps=1e-5)
    return torch.cat([norm_x, norm_y], dim=1)


class MMScoringTrainer(L.LightningModule):
    def __init__(self, paras, score_mean, coreset, net, criterion, lr, max_epoch, print_interval, save_path):
        super().__init__()
        self.net, self.criterion = net, criterion
        self.automatic_optimization = False
        self.coreset = coreset
        self.lr = lr
        self.max_epoch, self.epoch, self.print_interval = max_epoch, -1, print_interval
        self.train_loss, self.val_loss = defaultdict(list), defaultdict(list)
        self.score, self.gt, self.auc = [], [], []
        self.global_scale = torch.nn.Parameter(data=torch.Tensor([[0.0], [0.0]]), requires_grad=True)
        self.global_scale = self.global_scale.to(self.device)
        self.score_mean = score_mean.to('cuda').unsqueeze(0)

        self.log = []
        if not os.path.exists(os.path.join(save_path, 'log')):
            os.makedirs(os.path.join(save_path, 'log'))

        self.best_loss = 1e6
        self.save_path = save_path
        self.paras = paras
        for key in self.paras:
            print('{}: {}'.format(key, self.paras[key]), end=' ')
        print('')
        self.anchor, self.magnitude = paras['anchor'], paras['magnitude']

    def get_final_score(self, feat, core_feat):
        score = self.__calculate_score__(feat, core_feat)
        norm = F.normalize(feat - core_feat, dim=1, eps=1e-3) * 5  # 保证数值不会很小
        local_scale = self.net(norm, core_feat)
        global_scale = get_scale(self.global_scale)
        pred = local_scale * global_scale
        pred = torch.sum(pred * score, dim=1).unsqueeze(-1)
        return pred, score.mean(dim=1), local_scale

    def permutation(self, feat, core):
        x, y = feat[:, :768], feat[:, 768:]
        core_x, core_y = core[:, :768], core[:, 768:]
        size = feat.shape[0] // 2
        x_perm = self.__permute__(x[:size])
        y_perm = self.__permute__(y[:size])
        x = torch.cat([x_perm, x[size:]], dim=0)
        y = torch.cat([y_perm, y[size:]], dim=0)

        core_x_perm = self.__permute__(core_x[size:])
        core_y_perm = self.__permute__(core_y[size:])
        core_x = torch.cat([core_x[:size], core_x_perm], dim=0)
        core_y = torch.cat([core_y[:size], core_y_perm], dim=0)

        return torch.cat([x, y], dim=1), torch.cat([core_x, core_y], dim=1)

    @staticmethod
    def __permute__(x):
        perm = torch.randperm(x.shape[0])
        return x[perm]

    def calculate_loss(self, batch, batch_idx):
        loss = defaultdict()
        feat, core_feats, score, label, mask = batch

        """ 随机噪声增强
        """
        core_feat = core_feats[0]
        ub = (0.1 * self.score_mean[0, 0].item(), 0.1 * self.score_mean[0, 1].item())
        feat = self.add_random_noise(feat, ub=ub)
        core_feat = self.add_random_noise(core_feat, ub=ub)
        pred_s, dist_s, local_scale = self.get_final_score(feat, core_feat)
 
        loss['scaling'] = (math.exp(1) - local_scale[label == -1]).mean() + F.relu(local_scale[label == 1] - 1).mean()
        loss['svdd'], loss['margin'] = self.criterion(pred_s, targets=label, is_weighted=True)

        """ 两个局部尺度和各自方向的关系，应该也存在一个匹配的特性，这是多模态的特性
        """
        feat_perm, core_feat_perm = self.permutation(feat, core_feat)
        _, _, ls_perm = self.get_final_score(feat_perm, core_feat_perm)
        loss['perm'] = (math.exp(1) - ls_perm).mean()

        """ 局部距离的一致性
        """
        if self.anchor >= 1:
            loss['const'] = 0.0
            for i in range(1, self.anchor + 1):
                pred_nn, dist_nn, _ = self.get_final_score(feat, core_feats[i])
                loss['const'] += 1 / self.anchor * F.relu(pred_nn - self.magnitude * dist_nn / dist_s * pred_s).mean()

            for i in range(self.anchor + 1, 2 * self.anchor + 1):
                pred_anchor, _, _ = self.get_final_score(feat, core_feats[i])
                loss['const'] += 1 / self.anchor * F.relu(pred_s - pred_anchor).mean()

        loss['l2'] = l2_regularization(self.net)
        loss['l1'] = l1_regularization(self.net)
        return loss, pred_s

    def __calculate_score__(self, feat, core_feat):
        x, y = feat[:, :768], feat[:, 768:]
        x0, y0 = core_feat[:, :768], core_feat[:, 768:]
        sx = F.pairwise_distance(x, x0).view(-1, 1)
        sy = F.pairwise_distance(y, y0).view(-1, 1)
        score = torch.cat([sx, sy], dim=1)
        score /= self.score_mean
        return score

    def add_random_noise(self, feat, lb=(0.0, 0.0), ub=(1.0, 1.0)):
        x, y = feat[:, :768], feat[:, 768:]
        x, y = self._add_noise(x, ub=ub[0], lb=lb[0]), self._add_noise(y, ub=ub[1], lb=lb[1])
        return torch.cat([x, y], dim=1)

    def _add_noise(self, x, lb=0.0, ub=1.0):
        noise = torch.rand(size=x.size(), device=self.device)
 
        noise = F.normalize(noise, dim=1)
        scale = torch.rand((noise.shape[0], 1), device=self.device)
        scale = scale * (ub - lb) + lb
        return x + noise * scale

    def coreset_to_feature(self, idx, k):
        total_anchor = idx.shape[1] // 2
        core_feat = torch.cat([self.coreset['IMG'][idx[:, k]],
                               self.coreset['PCD'][idx[:, total_anchor + k]]], dim=1)
        return core_feat

    def preprocess(self, batch):
        feat, idx, score, label, mask = batch
        core_feats = []
        for i in range(2 * self.anchor + 1):
            core_feats.append(self.coreset_to_feature(idx, i))
        score = score / self.score_mean
        return feat, core_feats, score, label, mask

    def training_step(self, batch, batch_idx):
        batch = self.preprocess(batch)
        loss, _ = self.calculate_loss(batch, batch_idx)
        final = 0.
        for key in loss.keys():
            
            self.train_loss[key].append(loss[key].detach())
            final += self.paras[key] * loss[key]

        optimizer1, optimizer2 = self.optimizers()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        self.manual_backward(final)

        optimizer1.step()
        optimizer2.step()

    def on_train_epoch_end(self):
        self.epoch += 1
        losses = defaultdict()
        for key in self.train_loss:
            losses[key] = torch.stack(self.train_loss[key]).mean().detach().cpu()

        if self.epoch % self.print_interval == 0:
            print("\r Epoch {}: ".format(self.epoch), end=' ')
            for key in self.train_loss:
                if key not in ['l2', 'l1']:
                    print('{} loss {:.4f}'.format(key, losses[key].item()), end=' ')
            print('')

        self.train_loss.clear()

    def validation_step(self, batch, idx):
        batch = self.preprocess(batch)

        loss = defaultdict()
        feat, core_feats, score, label, mask = batch
        pred_s, dist_s, scale = self.get_final_score(feat, core_feats[0])

        loss['svdd'], loss['margin'] = self.criterion(pred_s, targets=label, is_weighted=False)

        self.gt.append(mask.detach())
        self.score.append(pred_s.detach())
        for key in loss.keys():
            self.val_loss[key].append(loss[key].detach())

    def on_validation_epoch_end(self):
        losses = defaultdict()
        for key in self.val_loss:
            losses[key] = torch.stack(self.val_loss[key]).mean().detach().cpu()

        """ Calculate auroc in test dataset
        """
        scores = torch.stack(self.score).cpu().flatten().numpy()
        gts = torch.stack(self.gt).cpu().flatten().numpy()
        if self.epoch >= 0:
            auroc = roc_auc_score(gts, scores)
            final = losses['svdd'] + losses['margin'] * self.paras['margin']

            self.log.append((self.epoch, auroc))

            if final <= self.best_loss:
                self.best_loss = final
                torch.save(self.net.state_dict(), os.path.join(self.save_path, 'MMScoringTrainerV2.pth'))
                np.save(os.path.join(self.save_path, 'global_scale'), self.global_scale.data.cpu().numpy())

            if self.epoch % self.print_interval == 0:
                print("\r Val Epoch {}: ".format(self.epoch), end=' ')
                for key in self.val_loss:
                    if key not in ['l2', 'l1']:
                        print('{} loss {:.4f}'.format(key, losses[key].item()), end=' ')
                print('Auroc {:.4f}'.format(auroc))

        self.val_loss.clear()
        self.score.clear()
        self.gt.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.net.parameters()), lr=self.lr, weight_decay=1e-4)
        optimizer1 = optim.Adam([self.global_scale], lr=5e-3, weight_decay=1e-4)
        return [optimizer, optimizer1]
