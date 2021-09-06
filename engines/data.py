# -*- coding: utf-8 -*-
# @Time : 2020/9/13 3:18 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import os
import json
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer


class DataManager:
    """
    数据管理器
    """
    def __init__(self, logger):
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_token_number = len(self.tokenizer.get_vocab())
        self.max_sequence_length = 100
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        self.train_file = 'data/dev_data_mini.json'
        self.dev_file = 'data/dev_data_mini.json'
        self.predict2id_file = 'data/vocab/predict2id'
        self.token2id_file = 'data/vocab/token2id'
        self.token2id, self.id2token, self.predict2id, self.id2predict = self.load_vocab()
        self.predict_label_nums = len(self.predict2id)

    @staticmethod
    def load_data(file):
        with open(file, encoding='utf-8') as data_file:
            data = json.load(data_file)
        return data

    def load_vocab(self):
        """
        加载词表
        :return:
        """
        if not os.path.isfile(self.token2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_vocab()
        self.logger.info('loading vocab...')
        token2id, id2token = {}, {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        predict2id, id2predict = {}, {}
        with open(self.predict2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                predict2id[label] = label_id
                id2predict[label_id] = label
        return token2id, id2token, predict2id, id2predict

    def build_vocab(self):
        """
        生成词表
        :return:
        """
        data = self.load_data(self.train_file)
        dev_data = self.load_data(self.dev_file)
        data.extend(dev_data)
        token_list = []
        predict_list = []
        for item in tqdm(data):
            token_list.extend(list(item.get('text', '')))
            for _, p, _ in item['spo_list']:
                predict_list.append(p)
        token_list = list(set(token_list))
        token_list = [token for token in token_list if not re.search(r'\s', token)]
        token2id = dict(zip(token_list, range(1, len(token_list) + 1)))
        id2token = dict(zip(range(1, len(token_list) + 1), token_list))
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(token_list) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(token_list) + 1
        # 保存词表及标签表
        with open(self.token2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')

        predict_list = list(set(predict_list))
        predict2id = dict(zip(predict_list, range(1, len(predict_list) + 1)))
        id2predict = dict(zip(range(1, len(predict_list) + 1), predict_list))
        id2predict[0] = self.PADDING
        predict2id[self.PADDING] = 0
        with open(self.predict2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2predict:
                outfile.write(id2predict[idx] + '\t' + str(idx) + '\n')
        return token2id, id2token, predict2id, id2predict

    def sequence_padding(self, inputs, value=0, seq_dims=1):
        """
        Numpy函数，将序列padding到同一长度
        """
        length = [self.max_sequence_length]
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        return np.array(outputs)

    def prepare(self, tokens, labels, is_padding=True):
        pass

    @staticmethod
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def prepare_bert_embedding(self, data):
        batch_token_ids, batch_attention_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for item in tqdm(data):
            token_ids = self.tokenizer.encode(item['text'])
            segment_ids = [1] * len(token_ids)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for subject, predict, object_ in item['spo_list']:
                subject_ids = self.tokenizer.encode(subject)[1:-1]
                p = self.predict2id[predict]
                object_ids = self.tokenizer.encode(object_)[1:-1]
                subject_idx = self.search(subject_ids, token_ids)
                object_idx = self.search(object_ids, token_ids)
                if subject_idx != -1 and object_idx != -1:
                    s = (subject_idx, subject_idx + len(subject_ids) - 1)
                    o = (object_idx, object_idx + len(object_ids) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(self.predict2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                batch_token_ids.append(token_ids)
                batch_attention_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
        data_token_ids = self.sequence_padding(batch_token_ids)
        data_attention_ids = self.sequence_padding(batch_attention_ids)
        data_subject_labels = self.sequence_padding(batch_subject_labels)
        data_subject_ids = np.array(batch_subject_ids)
        data_object_labels = self.sequence_padding(batch_object_labels)
        return data_token_ids, data_attention_ids, data_subject_labels, data_subject_ids, data_object_labels

    def get_training_set(self):
        self.logger.info('loading training datasets...')
        train_data = self.load_data(self.train_file)
        token_ids_train, attention_ids_train, subject_labels_train, \
            subject_ids_train, object_labels_train = self.prepare_bert_embedding(train_data)
        self.logger.info('loading validation datasets...')
        dev_data = self.load_data(self.dev_file)
        token_ids_val, attention_ids_val, subject_labels_val, \
            subject_ids_val, object_labels_val = self.prepare_bert_embedding(dev_data)
        train_dataset = tf.data.Dataset.from_tensor_slices((token_ids_train, attention_ids_train,
                                                            subject_labels_train, subject_ids_train,
                                                            object_labels_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((token_ids_val, attention_ids_val, subject_labels_val,
                                                          subject_ids_val, object_labels_val))
        return train_dataset, val_dataset

    def get_valid_set(self):
        pass
