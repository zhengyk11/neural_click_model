# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import torch
import copy
import math
import random
from torch.autograd import Variable
from tqdm import tqdm
from network import Network
from tensorboardX import SummaryWriter
from torch import nn
# from utils import calc_metrics
use_cuda = torch.cuda.is_available()

MINF = 1e-30

class Model(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, args, query_size, doc_size, vtype_size):
        self.args = args

        # logging
        self.logger = logging.getLogger("neural_click_model")

        # basic config
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.max_d_num = args.max_d_num
        self.writer = None
        if args.train:
            self.writer = SummaryWriter(self.args.summary_dir)

        self.model = Network(self.args, query_size, doc_size, vtype_size)

        if args.data_parallel:
            self.model = nn.DataParallel(self.model)
        if use_cuda:
            self.model = self.model.cuda()

        self.optimizer = self.create_train_op()
        self.criterion = nn.MSELoss()

    def compute_loss(self, pred_scores, target_scores):
        """
        The loss function
        """
        total_loss = 0.
        loss_list = []
        cnt = 0
        for batch_idx, scores in enumerate(target_scores):
            cnt += 1
            loss = 0.
            for position_idx, score in enumerate(scores[2:]):
                if score == 0:
                    loss -= torch.log(1. - pred_scores[batch_idx][position_idx].view(1) + 1e-30)
                else:
                    loss -= torch.log(pred_scores[batch_idx][position_idx].view(1) + 1e-30)
            loss_list.append(loss.data[0])
            total_loss += loss
        total_loss /= cnt
        # print loss.data[0]
        return total_loss, loss_list

    def create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'rprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, data, min_loss_value, metric_save, patience, step_pbar):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        evaluate = True
        exit_tag = False
        num_steps = self.args.num_steps
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.model_dir, self.args.algo

        for bitx, batch in enumerate(train_batches):
            self.global_step += 1
            step_pbar.update(1)
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:,:-1])
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.model.train()
            self.optimizer.zero_grad()
            pred_logits = self.model(QIDS, UIDS, VIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/loss', loss.data[0], self.global_step)
            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if evaluate and self.global_step % self.eval_freq == 0:
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, shuffle=False)
                    eval_loss = self.evaluate(eval_batches, data, result_dir=self.args.result_dir, t=-1,
                                                  result_prefix='train_dev.predicted.{}.{}'.format(self.args.algo,
                                                                                                   self.global_step))
                    self.writer.add_scalar("dev/loss", eval_loss, self.global_step)
                    # for metric in ['ndcg@1', 'ndcg@3', 'ndcg@10', 'ndcg@20']:
                    #     self.writer.add_scalar("dev/{}".format(metric), metrics['{}'.format(metric)], self.global_step)
                    # if metrics['ndcg@10'] > max_metric_value:
                    #     self.save_model(save_dir, save_prefix+'_best')
                    #     max_metric_value = metrics['ndcg@10']
                    if eval_loss < min_loss_value:
                        self.save_model(save_dir, save_prefix + '_best')
                        min_loss_value = eval_loss

                    if eval_loss < metric_save:
                        metric_save = eval_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= self.patience:
                        self.adjust_learning_rate(self.args.lr_decay)
                        self.learning_rate *= self.args.lr_decay
                        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                        metric_save = eval_loss
                        patience = 0
                        self.patience += 1
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            if self.global_step >= num_steps:
                exit_tag = True

        return min_loss_value, exit_tag, metric_save, patience

    def train(self, data):
        min_loss_value, epoch, patience, metric_save = 1e10, 0, 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
        while not exit_tag:
            epoch += 1
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            min_loss_value, exit_tag, metric_save, patience = self._train_epoch(train_batches, data,
                                                                                min_loss_value, metric_save,
                                                                                patience, step_pbar)

    def evaluate(self, eval_batches, dataset, result_dir=None, result_prefix=None, t=-1):
        eval_ouput = []
        # total_loss_list = []
        total_loss, total_num = 0., 0
        for b_itx, batch in enumerate(eval_batches):
            if b_itx == t:
                break
            if b_itx % 100 == 0:
                self.logger.info('Evaluation step {}.'.format(b_itx))
            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:,:-1])
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda()

            self.model.eval()
            pred_logits = self.model(QIDS, UIDS, VIDS, CLICKS)
            loss, loss_list = self.compute_loss(pred_logits, batch['clicks'])
            # total_loss_list += loss_list
            # pred_logits_list = pred_logits.data.cpu().numpy().tolist()
            for pred_metric, data, pred_logit in zip(loss_list, batch['raw_data'], pred_logits.data.cpu().numpy().tolist()):
                eval_ouput.append([data['session_id'], data['query'],
                                   data['urls'][1:], data['vtypes'][1:], data['clicks'][2:], pred_logit, pred_metric])
            total_loss += loss.data[0] * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                for sample in eval_ouput:
                    fout.write('\t'.join(map(str, sample)) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_span_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        # metrics = self.cal_metrics(eval_ouput)
        # print metrics
        return ave_span_loss # , np.mean(total_loss_list)

    # def cal_dcg(self, y_true, y_pred, rel_threshold=0., k=10):
    #     if k <= 0.:
    #         return 0.
    #     s = 0.
    #     y_true_ = copy.deepcopy(y_true)
    #     y_pred_ = copy.deepcopy(y_pred)
    #     c = zip(y_true_, y_pred_)
    #     random.shuffle(c)
    #     c = sorted(c, key=lambda x: x[1], reverse=True)
    #     dcg = 0.
    #     for i, (g, p) in enumerate(c):
    #         if i >= k:
    #             break
    #         if g > rel_threshold:
    #             # dcg += (math.pow(2., g) - 1.) / math.log(2. + i) # nDCG
    #             dcg += g / math.log(2. + i) # * math.log(2.) # MSnDCG
    #     return dcg
    #
    # def cal_metrics(self, eval_ouput):
    #     total_metric = {}
    #     for k in [1, 3, 10, 20]:
    #         ndcg_list = []
    #         random_ndcg_list = []
    #         for _ in range(10):
    #             rel_list = {}
    #             pred_score_list = {}
    #             for sample in eval_ouput:
    #                 qid, uid, rel, pred_score = sample
    #                 if qid not in rel_list:
    #                     rel_list[qid] = []
    #                 if qid not in pred_score_list:
    #                     pred_score_list[qid] = []
    #                 rel_list[qid].append(rel)
    #                 pred_score_list[qid].append(pred_score)
    #             for qid in rel_list:
    #                 dcg = self.cal_dcg(rel_list[qid], pred_score_list[qid], k=k)
    #                 random_dcg = self.cal_dcg(rel_list[qid], [0.] * len(pred_score_list[qid]), k=k)
    #                 idcg = self.cal_dcg(rel_list[qid], rel_list[qid], k=k)
    #                 ndcg, random_ndcg = 0., 0.
    #                 if idcg > 0.:
    #                     ndcg = dcg / idcg
    #                     random_ndcg = random_dcg / idcg
    #                 ndcg_list.append(ndcg)
    #                 random_ndcg_list.append(random_ndcg)
    #         total_metric['ndcg@{}'.format(k)] = np.mean(ndcg_list)
    #         total_metric['random_ndcg@{}'.format(k)] = np.mean(random_ndcg_list)
    #     print total_metric
    #     return total_metric

    def save_model(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_prefix+'_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(), os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info('Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                      model_prefix,
                                                                                                      self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
        if not os.path.isfile(optimizer_path):
            optimizer_path = os.path.join(model_dir, model_prefix + '_best_{}.optimizer'.format(global_step))
        if os.path.isfile(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.logger.info('Optimizer restored from {}, with prefix {} and global step {}.'.format(model_dir,
                                                                                                     model_prefix,
                                                                                                     global_step))
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if not os.path.isfile(model_path):
            model_path = os.path.join(model_dir, model_prefix + '_best_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.logger.info('Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
