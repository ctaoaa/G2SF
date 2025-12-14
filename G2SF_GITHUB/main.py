from __future__ import print_function
import os

import numpy as np
import pandas as pd

from Util import init_seeds
import warnings
from config_parse_v2 import parse_args
from Dataset import mvtec3d_classes, eyecandies_classes
from Engine.train import train
import pickle


def main_single(args):
    args.model = "{}_{}_{}_{}".format(args.xyz_backbone_name, args.rgb_backbone_name, args.img_size, args.class_name)
    print(args)
    cur_auc, cur_seg_auc, cur_aupro, cur_aupro_0p01, cur_aupro_0p1, cur_aupro_0p05 = train(args)
    return cur_auc, cur_seg_auc, cur_aupro, cur_aupro_0p01, cur_aupro_0p1, cur_aupro_0p05


from collections import defaultdict


def main(args):
    init_seeds(0)

    if args.dataset == 'mvtec':
        classes = mvtec3d_classes()

    elif args.dataset == 'eyecandies':
        classes = eyecandies_classes()

    else:
        raise NotImplementedError

    result_path = os.path.join('./Result', args.dataset, args.exp_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.exists(os.path.join(result_path, 'result.pkl')):
        with open(os.path.join(result_path, 'result.pkl'), 'rb') as file:
            auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05 = pickle.load(file)
    else:
        auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05 = defaultdict(), defaultdict(), defaultdict(), defaultdict(), defaultdict(), defaultdict()

    for class_name in classes:
        print("Class {}".format(class_name))
        args.class_name = class_name

        auc[class_name], seg_auc[class_name], aupro[class_name], aupro_0p01[class_name], aupro_0p1[class_name], aupro_0p05[class_name] = main_single(args)

        with open(os.path.join(result_path, 'result.pkl'), 'wb') as file:
            pickle.dump([auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05], file)

    auc = pd.DataFrame.from_dict(auc)
    seg_auc = pd.DataFrame.from_dict(seg_auc)
    aupro = pd.DataFrame.from_dict(aupro)
    aupro_0p01 = pd.DataFrame.from_dict(aupro_0p01)
    aupro_0p1 = pd.DataFrame.from_dict(aupro_0p1)
    aupro_0p05 = pd.DataFrame.from_dict(aupro_0p05)

    auc['mean'] = auc.mean(axis=1)
    seg_auc['mean'] = seg_auc.mean(axis=1)
    aupro['mean'] = aupro.mean(axis=1)
    aupro_0p01['mean'] = aupro_0p01.mean(axis=1)
    aupro_0p1['mean'] = aupro_0p1.mean(axis=1)
    aupro_0p05['mean'] = aupro_0p05.mean(axis=1)

    auc.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])
    seg_auc.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])
    aupro.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])
    aupro_0p01.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])
    aupro_0p1.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])
    aupro_0p05.index = pd.Series(['IMG', 'PCD', 'FUS', 'WEI_IMG', 'WEI_PCD'])

    auc.to_csv(os.path.join(result_path, 'auc.csv'))
    seg_auc.to_csv(os.path.join(result_path, 'seg_auc.csv'))
    aupro.to_csv(os.path.join(result_path, 'aupro.csv'))
    aupro_0p01.to_csv(os.path.join(result_path, 'aupro_0p01.csv'))
    aupro_0p1.to_csv(os.path.join(result_path, 'aupro_0p1.csv'))
    aupro_0p05.to_csv(os.path.join(result_path, 'aupro_0p05.csv'))

    argsDict = args.__dict__
    with open(os.path.join(result_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main(parse_args())
