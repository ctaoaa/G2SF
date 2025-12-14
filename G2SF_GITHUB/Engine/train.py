import os
from Util import *
from Engine.feature import Features
from Dataset import create_data_loader, create_pseudo_anomaly_data_loader
import gc
from Engine.evaluation import Validator
from Fusion import FusionDataCollector, UNSFusionModel, FusionModelV2


def create_feature(args, train_loader):
    feature_model = Features(args)
    feature_model.construct_memory_bank(train_loader)
    feature_model.get_coreset()
    feature_model.save_model(args.save_path)

    del feature_model
    gc.collect()


def create_semi_supervised_fusion_data(args):
    feature_model = Features(args)
    feature_model.load_model(args.save_path.replace(args.exp_name, 'Complete'), is_patch_lib=False)

    evaluate_model = Validator(args)

    fusion_data_loader = create_pseudo_anomaly_data_loader(args)
    data_collector = FusionDataCollector(args)
    data_collector.collect(feature_model, evaluate_model, fusion_data_loader, remove_bg=True)
    data_collector.save_dataset(args.fusion_dataset_path)

    del feature_model, evaluate_model, data_collector
    gc.collect()


def train_fuser(args):
    print("Train supervised fusion model")
    feature_model = Features(args)
    feature_model.load_model(args.save_path.replace(args.exp_name, 'Complete'), is_patch_lib=False)
    fusion_model = FusionModelV2(args, feature_model, is_train=True)
    fusion_model.train()
    del fusion_model
    gc.collect()


def train(args):
    train_loader, test_loader = create_data_loader(args)
    print("Datasets have loaded ")

    args.save_path = os.path.join('./Result', args.dataset, args.exp_name, args.class_name)

    """ Extract features
    """

    args.fusion_dataset_path = os.path.join(args.save_path, 'fusion')
    args.complete_fusion_dataset_path = args.fusion_dataset_path.replace(args.exp_name, 'Complete')

    if not args.load_feature:
        create_feature(args, train_loader)

    if not args.load_fusion_dataset:
        print("Generate features for fusion")
        create_semi_supervised_fusion_data(args)

    if not args.load_fuser:
        train_fuser(args)

    print("Start evaluation! ")
    feature_model = Features(args)
    feature_model.load_model(args.save_path.replace(args.exp_name, 'Complete'), is_patch_lib=False)

    fuser = FusionModelV2(args, feature_model, is_train=False)

    evaluate_model = Validator(args)

    auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05 = evaluate_model.evaluate(feature_model, test_loader, fuser=fuser, remove_bg=False)
    evaluate_model.save_fig(args.save_path)
    return auc, seg_auc, aupro, aupro_0p01, aupro_0p1, aupro_0p05
