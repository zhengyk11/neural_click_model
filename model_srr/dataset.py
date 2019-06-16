# -*- coding:utf8 -*-

import glob
import os
import json
import logging
import math

# import Levenshtein
import numpy as np



class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, args, train_dirs=[], dev_dirs=[], test_dirs=[], isRank=False):
        self.logger = logging.getLogger("neural_click_model")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.args = args
        self.num_train_files = args.num_train_files
        self.num_dev_files = args.num_dev_files
        self.num_test_files = args.num_test_files
        self.qid_query = {}
        self.uid_url = {}

        self.qid_query, self.query_qid = {0: ''}, {'': 0}
        self.uid_url, self.url_uid = {0: ''}, {'': 0}
        self.vid_vtype, self.vtype_vid = {0: ''}, {'': 0}

        # self.query_freq = {}
        # for line in open('../data/mcm_0212_50/query_id.txt'):
        #     attr = line.strip().split('\t')
        #     query = attr[0].strip().lower()
        #     freq = int(attr[3])
        #     self.query_freq[query] = freq

        # self.train_session_id, self.dev_session_id = {}, {}
        # for line in open('../data/mcm_0212_50/metrics.txt'):
        #     attr = line.strip().split('\t')
        #     session_id = attr[0]
        #     self.train_session_id[session_id] = 0
        # for line in open('../data/mcm_0212_50/click_prediction.txt'):
        #     attr = line.strip().split('\t')
        #     session_id = attr[0]
        #     freq = int(attr[1])
        #     if freq < 10:
        #         continue
        #     self.dev_session_id[session_id] = 0

        self.qid_uid_set = {}
        self.train_set, self.dev_set, self.test_set = [], [], []
        if isRank:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset_rank(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset_rank(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))
            # if test_dirs:
            #     for test_dir in test_dirs:
            #         self.test_set += self.load_dataset_rank(test_dir, num=self.num_test_files, mode='test')
            #     self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))
        else:
            if train_dirs:
                for train_dir in train_dirs:
                    self.train_set += self.load_dataset(train_dir, num=self.num_train_files, mode='train')
                self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
            if dev_dirs:
                for dev_dir in dev_dirs:
                    self.dev_set += self.load_dataset(dev_dir, num=self.num_dev_files, mode='dev')
                self.logger.info('Dev set size: {} sessions.'.format(len(self.dev_set)))
            if test_dirs:
                for test_dir in test_dirs:
                    self.test_set += self.load_dataset(test_dir, num=self.num_test_files, mode='test')
                self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))
        self.logger.info('Query size: {}.'.format(len(self.qid_query)))
        self.logger.info('URL size: {}.'.format(len(self.uid_url)))
        self.logger.info('Vertical type size: {}.'.format(len(self.vid_vtype)))
        cnt = 0
        for qid in self.qid_uid_set:
            for uid in self.qid_uid_set[qid]:
                cnt += 1
        self.logger.info('Query-URL pair size: {}.'.format(cnt))

    def load_dataset(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        files = sorted(glob.glob(data_path + '/click_data*'))
        if num > 0:
            files = files[0:num]
        for fn in files:
            # print fn
            lines = open(fn).readlines()
            for line in lines:
                attr = line.strip().split('\t')
                session_id = attr[4]
                # if mode == 'train' and session_id not in self.train_session_id:
                #     continue
                # if mode == 'dev' and session_id not in self.dev_session_id:
                #     continue
                query = attr[0].strip().lower()

                urls = map(int, attr[2].strip().split())
                if len(urls) < self.max_d_num:
                    continue
                urls = urls[:self.max_d_num]
                # vtypes = map(int, attr[1].strip().split())[:self.max_d_num]
                vtypes = [0] * self.max_d_num
                clicks = map(int, attr[3].strip().split())[:self.max_d_num]
                clicks = [0, 0] + clicks
                if query not in self.query_qid and mode == 'train':
                    self.query_qid[query] = len(self.query_qid)
                    self.qid_query[self.query_qid[query]] = query
                if query in self.query_qid:
                    qid = self.query_qid[query]
                else:
                    continue
                qids = [qid for _ in range(self.max_d_num+1)]
                uids = [0]
                for url in urls:
                    if url not in self.url_uid and mode == 'train':
                        self.url_uid[url] = len(self.url_uid)
                        self.uid_url[self.url_uid[url]] = url
                    if url in self.url_uid:
                        uids.append(self.url_uid[url])
                    else:
                        uids.append(0)
                    if mode == 'train':
                        if qid not in self.qid_uid_set:
                            self.qid_uid_set[qid] = {}
                        if uids[-1] not in self.qid_uid_set[qid]:
                            self.qid_uid_set[qid][uids[-1]] = 0
                vids = [0]
                for vtype in vtypes:
                    if vtype not in self.vtype_vid and mode == 'train':
                        self.vtype_vid[vtype] = len(self.vtype_vid)
                        self.vid_vtype[self.vtype_vid[vtype]] = vtype
                    if vtype in self.vtype_vid:
                        vids.append(self.vtype_vid[vtype])
                    else:
                        vids.append(0)
                data_set.append({'session_id': session_id,
                                 'qids': qids, 'query': query,
                                 'uids': uids, 'urls': [''] + urls,
                                 'vids': vids, 'vtypes': [''] + vtypes,
                                 'rank': [i for i in range(11)],
                                 'clicks': clicks})
        return data_set

    def load_dataset_rank(self, data_path, num, mode):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        data_set = []
        files = sorted(glob.glob(data_path + '/click_data*'))
        if num > 0:
            files = files[0:num]
        for fn in files:
            # print fn
            lines = open(fn).readlines()
            for line in lines:
                attr = line.strip().split('\t')
                session_id = attr[4]
                # if session_id not in self.train_session_id:
                #     continue
                query = attr[0].strip().lower()
                if query not in self.query_qid and mode == 'train':
                    self.query_qid[query] = len(self.query_qid)
                    self.qid_query[self.query_qid[query]] = query
                if query in self.query_qid:
                    qid = self.query_qid[query]
                else:
                    continue
                urls = map(int, attr[2].strip().split())
                if len(urls) < self.max_d_num:
                    continue
                urls = urls[:self.max_d_num]
                vtypes = [0] * self.max_d_num
                # vtypes = map(int, attr[1].strip().split())[:self.max_d_num]
                # clicks = map(int, attr[3].strip().split())[:self.max_d_num]
                # clicks = json.loads(attr[6])[:self.max_d_num]
                for idx, (curr_url, curr_vtype) in enumerate(zip(urls, vtypes)):
                    clicks = [0, 0, 0]
                    qids = [qid, qid]
                    uids = [0]
                    if curr_url not in self.url_uid and mode == 'train':
                        self.url_uid[curr_url] = len(self.url_uid)
                        self.uid_url[self.url_uid[curr_url]] = curr_url
                    if curr_url not in self.url_uid:
                        continue
                    else:
                        uid = self.url_uid[curr_url]
                    uids.append(uid)
                    vids = [0]
                    if curr_vtype not in self.vtype_vid and mode == 'train':
                        self.vtype_vid[curr_vtype] = len(self.vtype_vid)
                        self.vid_vtype[self.vtype_vid[curr_vtype]] = curr_vtype
                    if curr_vtype not in self.vtype_vid:
                        vid = 0
                    else:
                        vid = self.vtype_vid[curr_vtype]
                    vids.append(vid)

                    if mode != 'train':
                        if qid not in self.qid_uid_set:
                            self.qid_uid_set[qid] = {}
                        if uids[-1] not in self.qid_uid_set[qid]:
                            self.qid_uid_set[qid][uids[-1]] = 0
                        else:
                            continue
                        data_set.append({'session_id': session_id + '-' + str(idx),
                                         'qids':qids, 'query': query,
                                         'uids': uids, 'urls': ['', curr_url],
                                         'vids': vids, 'vtypes': ['', curr_vtype],
                                         'rank': [i for i in range(2)],
                                         'clicks': clicks})
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'qids': [],
                      'uids': [],
                      'vids': [],
                      'ranks': [],
                      'clicks': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['vids'].append(sample['vids'])
            batch_data['ranks'].append(sample['rank'])
            batch_data['clicks'].append(sample['clicks'])
        return batch_data

    # def id_padding(self, batch_data, pad_id):
    #     pad_q_len = batch_data['max_query_length']
    #     pad_p_len = batch_data['max_passage_length']
    #     # word padding
    #     batch_data['query_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
    #                                             for ids in batch_data['query_token_ids']]
    #     batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
    #                                        for ids in batch_data['passage_token_ids']]
    #     return batch_data

    # def word_iter(self, set_name=None):
    #     """
    #     Iterates over all the words in the dataset
    #     Args:
    #         set_name: if it is set, then the specific set will be used
    #     Returns:
    #         a generator
    #     """
    #     if set_name is None:
    #         data_set = self.train_set + self.dev_set + self.test_set
    #     elif set_name == 'train':
    #         data_set = self.train_set
    #     elif set_name == 'dev':
    #         data_set = self.dev_set
    #     elif set_name == 'test':
    #         data_set = self.test_set
    #     else:
    #         raise NotImplementedError('No data set named as {}'.format(set_name))
    #     if data_set is not None:
    #         for sample in data_set:
    #             for token in self.qid_question_tokens[sample['question_id']]:
    #                 yield token
    #             for token in self.pid_passage_tokens[sample['passage_id']]:
    #                 yield token

    def convert_to_ids(self, text, vocab):
        """
        Convert the question and paragraph in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        return vocab.convert_to_ids(text, 'vocab')

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        indices += indices[:(self.gpu_num - data_size % self.gpu_num)%self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
