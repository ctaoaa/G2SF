import torch
import torch.nn.functional as F
import torch.nn as nn


class WDSADMarginLoss(torch.nn.Module):

    def __init__(self, c, margin, percentile=0.995, alpha=None, eta=1.0, eps=1e-5, reduction='mean'):
        super(WDSADMarginLoss, self).__init__()
        self.reduction = reduction
        self.c = c
        self.eta = eta
        self.eps = eps
        self.margin = margin
        if alpha is not None:
            self.is_weighted = True
            self.alpha = alpha
        else:
            self.is_weighted = False
        self.percentile = percentile

    """ We want to promote a larger margin between normal and anomaly distributions
    """

    def calculate_margin(self, dist, targets):
        bs = dist.shape[0]
        normal_dist = dist[targets == 1]
        scale = torch.quantile(normal_dist, self.percentile)

        normal_dist = normal_dist / scale
        anomaly_dist = dist[targets == -1] / scale

        a_bound = torch.min(anomaly_dist)
        n_bound = torch.quantile(normal_dist, self.percentile)

        margin_score0 = torch.sum(F.relu(normal_dist - a_bound))
        """ Calculate the intersected interval, distances have been normalized
        """
        margin_score1 = torch.sum(F.relu(n_bound - anomaly_dist))
        return (margin_score0 + margin_score1) / bs, n_bound, a_bound

    def anomaly_score(self, rep):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        score = torch.sqrt(dist + 1e-8)
        return score

    def forward(self, rep, targets=None, is_weighted=True):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        na_margin, n_bound, a_bound = self.calculate_margin(dist, targets)

        if targets is not None:
            loss_normal = dist[targets == 1]
            loss_anomaly = F.relu(1 / (dist[targets == -1] + self.eps) - 1 / self.margin)

            if is_weighted:
                # weight = (dist + 0.5) ** self.alpha
                # torch.clamp_(weight, max=3.0)
                # weight_a = (1 / (dist + self.eps) + 0.5) ** self.alpha
                # torch.clamp_(weight_a, max=3.0)

                weight = (dist) ** self.alpha
                torch.clamp_(weight, max=3.0, min=0.33)
                weight_a = (1 / (dist + self.eps)) ** self.alpha
                torch.clamp_(weight_a, max=3.0, min=0.33)

                loss_normal = dist[targets == 1] * weight[targets == 1]
                loss_anomaly = F.relu(1 / (dist[targets == -1] + self.eps) - 1 / self.margin) * weight_a[targets == -1]

            loss = torch.cat([loss_normal, loss_anomaly], dim=0)
        else:
            loss = dist

        return torch.mean(loss), na_margin

    def forward_negative(self, rep):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        loss = F.relu(1 / (dist + self.eps) - 1 / self.margin)
        return torch.mean(loss)


class NegativeMarginLoss(torch.nn.Module):

    def __init__(self, c, margin, percentile=0.995, eta=1.0, eps=1e-5, reduction='mean'):
        super(NegativeMarginLoss, self).__init__()
        self.reduction = reduction
        self.c = c
        self.eta = eta
        self.eps = eps
        self.margin = margin

        self.percentile = percentile

    """ We want to promote a larger margin between normal and anomaly distributions
    """

    def calculate_margin(self, dist, targets):
        bs = dist.shape[0]
        normal_dist = dist[targets == 1]
        scale = torch.quantile(normal_dist, self.percentile)

        normal_dist = normal_dist / scale
        anomaly_dist = dist[targets == -1] / scale

        a_bound = torch.min(anomaly_dist)
        n_bound = torch.quantile(normal_dist, self.percentile)

        margin_score0 = torch.sum(F.relu(normal_dist - a_bound))
        """ Calculate the intersected interval, distances have been normalized
        """
        margin_score1 = torch.sum(F.relu(n_bound - anomaly_dist))
        return (margin_score0 + margin_score1) / bs, n_bound, a_bound

    def anomaly_score(self, rep):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        score = torch.sqrt(dist + 1e-8)
        return score

    def forward(self, rep, targets=None, reduction=None):
        dist = torch.sum((rep - self.c) ** 2, dim=1)
        score = torch.sqrt(dist + 1e-8)
        na_margin, n_bound, a_bound = self.calculate_margin(dist, targets)

        loss_normal = dist[targets == 1]
        loss_anomaly = F.relu(-dist[targets == -1] + self.margin) * self.eta
        loss = torch.cat([loss_normal, loss_anomaly], dim=0)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss), score, na_margin
        elif reduction == 'sum':
            return torch.sum(loss), score, na_margin
        else:
            raise NotImplementedError


def l2_regularization(model):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Linear:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return sum(l2_loss)


def l1_regularization(model):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.Linear:  # 检查模块是否是nn.Linear
            l1_loss.append(torch.sum(torch.abs(module.weight)))
    return sum(l1_loss)
