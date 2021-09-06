# -*- coding: utf-8 -*-
# @Time : 2021/8/21 23:56 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import tensorflow as tf
import time
from engines.model import Model
from tqdm import tqdm


def train(data_manager, logger):
    train_dataset, val_dataset = data_manager.get_training_set()
    batch_size = 5
    epoch = 10
    predict_label_nums = data_manager.predict_label_nums
    relation_model = Model(predict_label_nums)
    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(batch_size).enumerate()):
            token_ids_train, attention_ids_train, subject_labels_train, subject_ids_train, object_labels_train = batch
            model = relation_model(token_ids_train, attention_ids_train, subject_ids_train)

