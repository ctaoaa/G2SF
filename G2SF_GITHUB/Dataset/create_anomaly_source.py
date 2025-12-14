import os
import random
import json

MVTEC_CLASS_NAMES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']
EYECANDIES_CLASS_NAMES = ['CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear',
                          'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy', ]

import argparse



def create_source_for_mvtec(args):
    DatasetPath = args.mvtec_path
    AnomalySource = []
    Num = 8
    for class_name in MVTEC_CLASS_NAMES:
        file_dir = os.path.join(DatasetPath, class_name, 'train', 'good', 'xyz')
        fpath_list = sorted([os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.tiff')])
        random.shuffle(fpath_list)
        AnomalySource.extend(fpath_list[:Num])

    SavePath = os.path.join(DatasetPath, 'anomaly_source')
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    print(AnomalySource)
    # with open(os.path.join(SavePath, 'anomaly_source_path.json'), "w") as file:
    #     file.write(json.dumps(AnomalySource))


def create_source_for_eyecandies(args):
    DatasetPath = args.eyecandies_path
    AnomalySource = []
    Num = 24
    for class_name in EYECANDIES_CLASS_NAMES:
        file_dir = os.path.join(DatasetPath, class_name, 'train', 'good', 'xyz')
        fpath_list = sorted([os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.tiff')])
        random.shuffle(fpath_list)
        AnomalySource.extend(fpath_list[:Num])

    SavePath = os.path.join(DatasetPath, 'anomaly_source')
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    with open(os.path.join(SavePath, 'anomaly_source_path.json'), "w") as file:
        file.write(json.dumps(AnomalySource))


def parse_args():
    parser = argparse.ArgumentParser(description='G2SF')

    parser.add_argument("--mvtec_path", default="/home/chengyu/dataset/MVTEC 3D", type=str)
    parser.add_argument("--eyecandies_path", default="Datasets/Eyecandies_preprocessed", type=str)
    parser.add_argument("--dataset", default="mvtec", type=str, choices=['mvtec', 'eyecandies'])

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.dataset == 'mvtec':
        create_source_for_mvtec(args)
    elif args.dataset == 'eyecandies':
        create_source_for_eyecandies(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    random.seed(0)
    main()
