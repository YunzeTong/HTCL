import sys
sys.path.append("../")
import argparse
import time
import torch
import random
import numpy as np
from domainbed.datasets import datasets
from heterolize.dataset import EnvChangeDataset as EDataset
from heterolize.feature_heterolizer import FeatureHeterolizer
from heterolize.env_splitter import EnvSplitter
from heterolize.utils import make_final_dataset
import os

def initialize_dataset(args) -> EDataset:
    start_time = time.time()
    dataset = vars(datasets)[args.dataset](args.data_dir)
    H_dataset = EDataset(num_envs=args.num_clusters, num_labels=dataset.num_classes)
    env_sub = 0
    for env_i, env in enumerate(dataset):
        if env_i == args.test_env:
            env_sub += 1
            continue
        H_dataset.old_envs.append(env)
        H_dataset.old_envs_num.append(len(env))
        for pic_i in range(len(env)):
            H_dataset.add_item(env[pic_i][1], env_i - env_sub) # pic's label, original env
    H_dataset.shuffle(args.seed)
    H_dataset.update_domain_labels()
    print("Finish creating initialized dataset, {:.3f} s".format(time.time() - start_time))
    # logger.info("Finish creating no env dataset, {:.3f} s".format(time.time() - start_time))
    # target_index = dataset.environments.index(args.test_env_name)  # samely args.test_envs
    args.test_env_name = dataset.environments[args.test_env]
    target_env = dataset.datasets[args.test_env]
    return H_dataset, target_env, time.time() - start_time


def make_heterogenous_dataset(args, logger=None, hparams=None):
    
    dataset, target_env, dual_time = initialize_dataset(args)

    # logging
    assert logger != None, "no logger for recording heterogeneous pattern generation"
    logger.info("Finish creating no domain-distinguished dataset, {:.3f} s".format(dual_time))
    logger.info(f"target env:{args.test_env_name}, index in original dataset: {args.test_env}")

    # prepare and pretrain feature_heterolizer
    feature_heterolizer = FeatureHeterolizer(args, dataset, args.num_clusters, logger)
    pretrain_start_time = time.time()
    
    dir = f'/data/home/tongyunze/DG_model/heterogeneity/{args.dataset}/te_{args.test_env_name}'
    featurizer_path = dir + f'/{args.seed}s_res18_{args.feature_dim}d_{args.pretrain_epoch}e_featurizer.pt'
    classifier_path = dir + f'/{args.seed}s_res18_{args.feature_dim}d_{args.pretrain_epoch}e_classifier.pt'

    assert os.path.exists(featurizer_path) and os.path.exists(classifier_path), "can't find pretrained backbone model, please use pretrain in args list first"
    
    feature_heterolizer.featurizer.load_state_dict(torch.load(featurizer_path))
    feature_heterolizer.classifier.load_state_dict(torch.load(classifier_path))

    logger.info("Total Loading Time: {:.3f} s".format(time.time() - pretrain_start_time))

    # prepare pattern generation module
    env_splitter = EnvSplitter(args, dataset.labels, dataset.num_labels, args.num_clusters, \
                            feature_heterolizer.featurizer.n_outputs, logger=logger)
    
    best_domain_labels = None
    best_distance = 0

    train_start_time = time.time()
    for epoch_i in range(args.main_epoch):
        logger.info(f"[Main Bone] Start Heterogeneous Dividing Pattern Generation for Epoch {epoch_i}")
        # heterolize feature with stable splitted env
        features, distance = feature_heterolizer.explore_heterogeneity(dataset.domain_label, args.lambda_1)
        
        if best_domain_labels == None:
            best_domain_labels = dataset.domain_label.clone()
            best_distance = distance
        else:
            if distance > best_distance:
                logger.info(f"[Main bone] Update domain labels, old dist:{best_distance}, new dist:{distance}")
                best_domain_labels = dataset.domain_label.clone()
                best_distance = distance
        if epoch_i != args.main_epoch - 1:
            # split environment with given feature
            if args.use_KMeans_method:
                new_domain_labels = env_splitter.KMeans_split(features)
            else:
                new_domain_labels = env_splitter.generate_group(features)
            dataset.update_domain_labels(new_domain_labels)

            logger.info(f"[Main Bone] Finish Epoch {epoch_i}; Domain splitted into {[torch.nonzero(new_domain_labels==i).ravel().shape[0] for i in range(args.num_clusters)]}")
        else:
            logger.info(f"[Main Bone] Finish Epoch {epoch_i}; Domain splitted into {[torch.nonzero(best_domain_labels==i).ravel().shape[0] for i in range(args.num_clusters)]}")

    logger.info("Train totally use {:.2f} hours".format((time.time() - train_start_time) / 3600))

    for i in range(args.num_clusters):
        temp = torch.nonzero(best_domain_labels == i).shape[0]
        if temp == 0: # If a domain is reduced into 0, just omit it
            logger.info("[Warning]one env has extincted")
            continue
        if int(temp * args.holdout_fraction) < 1: # if the number of contained samples in a domain is too small, set it to unexisted index and don't use these samples
            logger.info("[Warning]one env has little samples, drop")
            best_domain_labels[temp] = args.num_clusters + 1
    
    if args.use_KMeans_method:
        return make_final_dataset(args, dataset, dataset.domain_label, target_env, hparams)
    else:
        return make_final_dataset(args, dataset, best_domain_labels, target_env, hparams)


def pretrain(args, logger):
    """
    pretrain the backbone for heterogeneous dividing pattern generation.

    use `args.seed` to control the parameters of featurizer instead of `trial_seed` because `trial_seed` is used to seed split_dataset and random_hparams according to the description of DomainBed
    """
    assert logger != None, "logger not ready"

    dataset, _, dual_time = initialize_dataset(args) # use args.seed to split

    logger.info("Finish creating no group split dataset, {:.3f} s".format(dual_time))
    logger.info(f"target env:{args.test_env_name}, index in original dataset: {args.test_env}")

    # prepare and pretrain feature_heterolizer
    feature_heterolizer = FeatureHeterolizer(args, dataset, args.num_clusters, logger)
    pretrain_start_time = time.time()
    
    dir = f'/data/home/tongyunze/DG_model/heterogeneity/{args.dataset}/te_{args.test_env_name}'
    feature_heterolizer.pretrain(args, dataset, dir)
    logger.info("Total Pretrain Time: {:.3f} s".format(time.time() - pretrain_start_time))
    
