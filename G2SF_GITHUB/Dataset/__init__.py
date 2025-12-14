from .mvtec3d import BaseDataset, MVTecDataset
from torch.utils.data import DataLoader
from .util import BalancedBatchSampler, MultiEpochsDataLoader
from .mvtec3d_pseudo import MVTecPseudoDataset
from .eyecandies import EyecandiesDataset
from .eyecandies_pseudo import EyeCandiesPseudoDataset

__all__ = ['BaseDataset', 'MVTecDataset', 'mvtec3d_classes', 'eyecandies_classes', 'create_data_loader', 'MVTecPseudoDataset', 'EyeCandiesPseudoDataset', 'create_pseudo_anomaly_data_loader']


def mvtec3d_classes():
    return ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire", ]


def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',
    ]


def create_data_loader(args):
    kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True}
    if args.dataset == 'mvtec':
        train_dataset = MVTecDataset(args, is_train=True)
        test_dataset = MVTecDataset(args, is_train=False)
    elif args.dataset == 'eyecandies':
        train_dataset = EyecandiesDataset(args, is_train=True)
        test_dataset = EyecandiesDataset(args, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    return train_loader, test_loader


def create_pseudo_anomaly_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'persistent_workers': True}
    if args.dataset == 'mvtec':
        dataset = MVTecPseudoDataset(args)
    elif args.dataset == 'eyecandies':
        dataset = EyeCandiesPseudoDataset(args)
    else:
        raise NotImplementedError
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    return data_loader
