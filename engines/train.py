# -*- coding: utf-8 -*-
# @Time : 2021/8/21 23:56 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import tensorflow as tf
import time
from engines.model import Model
from transformers import AdamW
from tqdm import tqdm


def train(data_manager, logger):
    train_dataset, val_dataset = data_manager.get_training_set()
    batch_size = 5
    epoch = 10
    learning_rate = 5e-5
    predict_label_nums = data_manager.predict_label_nums
    relation_model = Model(predict_label_nums)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(batch_size).enumerate()):
            token_ids_train, attention_ids_train, subject_labels_train, subject_ids_train, object_labels_train = batch
            attention_ids_train = tf.cast(attention_ids_train, dtype=tf.double)
            with tf.GradientTape() as tape:
                subject_predicts, predict_output = relation_model(token_ids_train, attention_ids_train, subject_ids_train)
                subject_loss_vec = tf.keras.losses.binary_crossentropy(subject_predicts, subject_labels_train)
                object_loss_vec = tf.keras.losses.binary_crossentropy(predict_output, object_labels_train)
                subject_loss = tf.reduce_mean(subject_loss_vec)
                object_loss = tf.reduce_mean(object_loss_vec, 2)
                object_loss = tf.reduce_mean(object_loss)
                total_loss = subject_loss + object_loss
                # 定义好参加梯度的参数
                gradients = tape.gradient(total_loss, relation_model.trainable_variables)
                # 反向传播，自动微分计算
                optimizer.apply_gradients(zip(gradients, relation_model.trainable_variables))









