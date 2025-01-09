# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam

from models import KMeans
from modules import NCELoss, PCLoss
from utils import recall_at_k, ndcg_k, get_metric, nCr, GRL


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.num_clusters = [int(i) for i in self.args.num_clusters.split(",")]
        self.clusters = []
        for num_intent_cluster in self.num_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        self.stability_discriminator = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.augmentation_discriminator = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.grl_layer = GRL(1.0)
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
            self.stability_discriminator.cuda()
            self.augmentation_discriminator.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam([{'params': self.model.parameters()}, {'params': self.stability_discriminator.parameters()},
                           {'params': self.augmentation_discriminator.parameters()}],
                          lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)
        self.class_criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        post_fix_dic = {
            "HIT@5": HIT_1,
            "NDCG@5": NDCG_1,
            "HIT@10": HIT_5,
            "NDCG@10": NDCG_5,
            "HIT@20": HIT_10,
            "NDCG@20": NDCG_10,
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), post_fix_dic

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        post_fix_dic = {
            "HIT@5": recall[0],
            "NDCG@5": ndcg[0],
            "HIT@10": recall[1],
            "NDCG@10": ndcg[1],
            "HIT@20": recall[3],
            "NDCG@20": ndcg[3],
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix), post_fix_dic

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SparseEnNetTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(SparseEnNetTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

    def _instance_cl_one_pair_contrastive_learning(self, inputs, crops, aug_types, intent_ids=None):
        output = []
        for i in range(2):
            if crops[i].shape[0] != 1:
                crop_batch = crops[i].to(self.device)
                cl_batch = inputs[i].to(self.device)
                cl_sequence_output = self.model(cl_batch, crop_batch)
            else:
                cl_batch = inputs[i].to(self.device)
                cl_sequence_output = self.model(cl_batch)
            output.append(cl_sequence_output)
        cl_sequence_output = torch.cat(output, dim=0)
        batch_size = cl_sequence_output.shape[0]
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(batch_size, -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = batch_size // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)

        # con_output_slice = torch.cat(cl_output_slice, dim=1)
        con_output_slice = torch.mul(*cl_output_slice)
        con_output_slice = self.grl_layer(con_output_slice)
        same_dis_out = self.stability_discriminator(con_output_slice)

        same = (aug_types[0] == aug_types[1]).long()
        same_label = torch.ones(same_dis_out.shape[0], dtype=torch.long).cuda() * same.cuda()

        same_class_loss = self.class_criterion(same_dis_out, same_label)

        output_slice = self.grl_layer(cl_output_slice[1])
        modal_dis_out = self.augmentation_discriminator(output_slice)

        modal = aug_types[1]
        modal_label = torch.ones(same_dis_out.shape[0], dtype=torch.long).cuda() * modal.cuda()

        modal_class_loss = self.class_criterion(modal_dis_out, modal_label)

        class_loss = same_class_loss + modal_class_loss

        return cl_loss, class_loss

    def _sel_one_pair_contrastive_learning(self, inputs, intents, negintents, intent_ids):
        """
        self-training enhance learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)
        # cl_loss = self.pcl_criterion(seq, seq, intents=intents, intent_ids=None)/2
        cl_loss = -self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=negintents, intent_ids=None)
        if self.args.de_noise:
            cl_loss += self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss += self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            # ------clustering ----- #
            print("Clustering For Category Feature Representation:")
            self.model.eval()
            kmeans_training_data = []
            rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
            for i, (rec_batch, _, _) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch
                sequence_output = self.model(input_ids)

                # average sum
                if self.args.seq_representation_type == "mean":
                    sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                sequence_output = sequence_output.detach().cpu().numpy()
                kmeans_training_data.append(sequence_output)
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)

            # train multiple clusters
            print("Training Clusters:")
            for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                cluster.train(kmeans_training_data)
                self.clusters[i] = cluster
            # clean memory
            del kmeans_training_data
            import gc

            gc.collect()

            class_ls = 0

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                """
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_crop_batch in cl_batches:
                    crop_batch = [cl_crop_batch[1], cl_crop_batch[4]]
                    cl_batch = [cl_crop_batch[0], cl_crop_batch[3]]
                    aug_type = [cl_crop_batch[2], cl_crop_batch[5]]
                    cl_loss1, class_loss = self._instance_cl_one_pair_contrastive_learning(
                        cl_batch, crop_batch, aug_type, intent_ids=seq_class_label_batches
                    )
                    cl_losses.append(self.args.nsl_weight * cl_loss1)
                    cl_losses.append(self.args.discriminator_weight * class_loss)
                    class_ls += class_loss.item()
                    if self.args.seq_representation_type == "mean":
                        sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                    sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                    sequence_output_0= sequence_output
                    sequence_output = sequence_output.detach().cpu().numpy()
                    # query on multiple clusters
                    for cluster in self.clusters:
                        seq2intents = []
                        intent_ids = []
                        negintents = []
                        intent_id, seq2intent, negintent = cluster.query(sequence_output)
                        seq2intents.append(seq2intent)
                        intent_ids.append(intent_id)
                        negintents.append(negintent)

                    cl_loss3 = self._sel_one_pair_contrastive_learning(
                        cl_batch, intents=seq2intents, negintents=negintents, intent_ids=intent_ids
                    )
                    cl_losses.append(self.args.sel_weight * cl_loss3)

                joint_loss = self.args.nip_weight * rec_loss


                for cl_loss in cl_losses:
                    joint_loss += cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
