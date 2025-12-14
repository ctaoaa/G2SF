import numpy as np
import cv2 as cv
from torch.utils.data import Sampler
import torch


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.randomGenerator(self.dataset.normal_idx)
        self.outlier_generator = self.randomGenerator(self.dataset.anomaly_idx)
        # n_normal: 2/3; n_outlier: 1/3
        if self.cfg.num_anomalies != 0:
            self.n_normal = 2 * self.cfg.batch_size // 3
            self.n_anomaly = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_anomaly = 0

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch

    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_anomaly):
                batch.append(next(self.outlier_generator))

            yield batch
