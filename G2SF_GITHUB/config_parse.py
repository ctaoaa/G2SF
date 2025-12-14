import argparse
import numpy as np

__all__ = ['parse_args']

import os


def str2bool(string):
    return True if string.lower() == 'true' else False


def decimal2str(dec):
    dec = np.around(dec, decimals=1)
    string = str(dec)
    string.replace('.', 'p')
    return string


def parse_args():
    parser = argparse.ArgumentParser(description='G2SF')

    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # model hyperparameters
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str)
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str)
    parser.add_argument('--group_size', default=128, type=int)
    parser.add_argument('--num_group', default=512, type=int)
    parser.add_argument('--random_state', default=None, type=int, help='random_state for random project')

    parser.add_argument('--num_normal', default=0, type=int)
    parser.add_argument('--num_anomalies', default=800, type=int)
    parser.add_argument('--fusion_batch_size', default=1024 * 8, type=int)
    parser.add_argument('--fusion_test_batch_size', default=1024 * 8, type=int)
    parser.add_argument('--print_interval', default=5, type=int)

    """ learning rate hyperparameter
    """
    parser.add_argument('--lr', default=1.5e-4, type=int)
    parser.add_argument('--max_epoch', default=80, type=int)

    """ hyper-parameters for fusion model
    """
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--n_hidden', default='512', type=str)
    parser.add_argument('--drop_out', default=0.5, type=float)
    parser.add_argument('--final_dim', default=1, type=int)

    """ hyper-parameters in loss function
    """
    parser.add_argument('--perc', default=1, type=float)
    parser.add_argument('--alpha', default=2.0, type=float)

    parser.add_argument('--anchor', default=5, type=int)
    parser.add_argument('--magnitude', default=1.2, type=float)

    parser.add_argument('--margin', default=10.0, type=float)
    parser.add_argument('--const', default=40.0, type=float)
    parser.add_argument('--perm', default=20.0, type=float)
    parser.add_argument('--scaling', default=3.0, type=float)

    parser.add_argument('--scoring_mode', default='min', type=str, choices=['zero', 'min', 'max', 'average', 'weight'])
    parser.add_argument('--scoring_weight', default=0.9, type=float, choices=['min', 'max', 'average', 'weight'])

    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=5e-4, type=float)

    parser.add_argument('--coreset_eps', default=0.9, type=float, help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float)
    parser.add_argument('--float16', default=True, type=bool)
    parser.add_argument('--coreset_mode', default='sparse', type=str, choices=['sparse', 'nn'])
    parser.add_argument('--proj_dim', default=256, type=int)

    """ unsupervised fusion
    """
    parser.add_argument('--ocsvm_nu', default=0.5, type=float)
    parser.add_argument('--ocsvm_max_iter', default=1000, type=int)

    """ Whether to use saved data or re-train them
    """
    parser.add_argument('--load_feature', default=True, type=str2bool)
    parser.add_argument('--load_fusion_dataset', default=True, type=str2bool)
    parser.add_argument('--load_fuser', default=False, type=str2bool)

    parser.add_argument('--dataset', default='mvtec', type=str, choices=['mvtec', 'eyecandies'])

    """ Experiment name
    """
    parser.add_argument('--exp_name', default='Complete', type=str)
    args = parser.parse_args()

    if args.dataset == 'mvtec':
        args.dataset_path = r"/home/chengyu/dataset/MVTEC 3D"
        args.full_dataset_path = r"/home/chengyu/MVTEC-3D-AD/MVTEC 3D"
        args.source_path = os.path.join(args.dataset_path, 'anomaly_source')

    if args.dataset == 'eyecandies':
        args.dataset_path = r"D:\Eyecandies_preprocessed"
        args.full_dataset_path = r"D:\Eyecandies_preprocessed"
        args.source_path = os.path.join(args.dataset_path, 'anomaly_source')

    args.normalize_mean, args.normalize_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return args

