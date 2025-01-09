# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from datetime import datetime

import nni

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset

from trainers import SparseEnNetTrainer
from models import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:---------------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_id", default='None')
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--tune", type=bool, default=False, help="for nni experiment")

    # data augmentation args
    parser.add_argument(
        "--training_data_ratio",
        default=1.0,
        type=float,
        help="percentage of training samples used for training - robustness analysis",
    )
    parser.add_argument(
        "--augment_type",
        default="random",
        type=str,
        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, pooling, random"
    )
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--omega", type=float, default=0.5, help="compress ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )
    parser.add_argument(
        "--n_views", default=2, type=int, help="Number of augmented data for each sequence - not studied."
    )
    parser.add_argument(
        "--num_clusters",
        default="256",
        type=str,
    )
    parser.add_argument(
        "--seq_representation_type",
        default="mean",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")

    # model args
    parser.add_argument("--model_name", default="SparseEnNet", type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--nip_weight", type=float, default=1.0, help="weight of next item prediction learning task")
    parser.add_argument("--nsl_weight", type=float, default=0.1, help="weight of negative sampling learning task")
    parser.add_argument("--discriminator_weight", type=float, default=0.1, help="weight of adversarial learning task")
    parser.add_argument("--sel_weight", type=float, default=0.1, help="weight of self-training enhanced learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    args = parser.parse_args()
    args_dict = vars(args)

    if args.tune:
        tune_args = nni.get_next_parameter()
        args_dict.update(tune_args)
        args = argparse.Namespace(**args_dict)
        args.model_idx = nni.get_trial_id()
        args.output_dir = os.path.join(args.output_dir, args.model_name, args.data_name, args.model_idx)
        args.attention_probs_dropout_prob = args.hidden_dropout_prob
    else:
        args.output_dir = os.path.join(args.output_dir, args.model_name, args.data_name)

    if args.do_eval:
        args.model_idx = args.eval_id
    else:
        args.model_idx = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    set_seed(args.seed)
    check_path(args.output_dir)

    if not args.tune:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 3
    args.mask_id = max_item + 1
    args.compress_id = max_item + 2

    # save model args
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    show_args_info(args)

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # training data for node classification
    cluster_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
    )
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    train_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = SASRecModel(args=args)

    trainer = SparseEnNetTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        load = torch.load(args.checkpoint_path)
        trainer.model.load_state_dict(load)
        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        print(f"---------------Training-------------------")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=20, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _, scores_dict = trainer.valid(epoch, full_sort=True)
            if args.tune:
                scores_dict['default'] = scores_dict['HIT@20']
                nni.report_intermediate_result(scores_dict)
            early_stopping(np.array(scores), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Testing-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info, scores_dict = trainer.test(0, full_sort=True)
        if args.tune:
            scores_dict['default'] = scores_dict['HIT@20']
            nni.report_final_result(scores_dict)

    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")


main()
