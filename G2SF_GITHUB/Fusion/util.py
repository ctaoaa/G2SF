import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
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


def feature_interpolation(feature, source_size, target_size):
    dim = feature.shape[1]
    interp_feat = feature.reshape(*source_size, dim)
    interp_feat = F.interpolate(interp_feat.permute(2, 0, 1).unsqueeze(0), size=target_size, mode='nearest')
    interp_feat = interp_feat.squeeze().permute(1, 2, 0).reshape(-1, dim)
    return interp_feat


def get_dataloader(path, batch_size, test_batch_size):
    feature_name = os.path.join(path, 'patch_lib.npz')
    # feature_name = os.path.join(path, 'patch_lib.npy')
    
    mask_name = os.path.join(path, 'per_label.npy')
    idx2coreset_name = os.path.join(path, 'idx2coreset.npy')
    score_name = os.path.join(path, 'score.npy')

    features = np.load(feature_name)['feature']
    # features = np.load(feature_name)

    masks = np.load(mask_name)
    scores = np.load(score_name)
    idx2coreset = np.load(idx2coreset_name)

    assert features.shape[0] == masks.shape[0]

    print("Number of anomaly features {}".format(np.count_nonzero(masks)))

    """ mask = 0 -> normal -> label = +1
        mask = 1 -> anomaly -> label = -1
    """
    labels = np.zeros_like(masks)
    labels[masks == 1] = -1
    labels[masks == 0] = 1

    print("Size of feature {}, score {}, and label {}".format(features.shape, scores.shape, labels.shape))

    dataset = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(idx2coreset).long(), torch.from_numpy(scores).float(),
                            torch.from_numpy(labels).long(), torch.from_numpy(masks).long())
    train_dataset, test_dataset = random_split(dataset, (0.5, 0.5))

    fix_batch_size = int(features.shape[0] * 0.7) // 20
    batch_size = min(fix_batch_size, batch_size)
    print(batch_size)
    test_batch_size = batch_size

    print("Train dataset: ", end='')
    train_sampler = get_sampler(train_dataset)
    print("Test dataset: ", end='')
    test_sampler = get_sampler(test_dataset)

    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True if train_sampler is None else False,
                              drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler, shuffle=True if test_sampler is None else False,
                             drop_last=True, **kwargs)

    return train_loader, test_loader


def get_sampler(subset):
    dataset = subset.dataset[subset.indices]
    print("Length {} ".format(dataset[0].shape[0]), end='')
    labels = dataset[-2].numpy()
    counter = Counter(labels)
    weight_map = {-1: 1. / counter[-1], 1: 1. / counter[1]}
    print("Weight map {}".format(weight_map))
    sampler = WeightedRandomSampler(weights=[weight_map[label] for label in labels],
                                    num_samples=len(dataset[0]), replacement=True)
    return sampler
