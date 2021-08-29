# -*- coding: utf-8 -*-
# @Time : 2021/8/29 23:08 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py 
# @Software: PyCharm
from abc import ABC
from transformers import TFBertModel
import tensorflow as tf


class Model(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()
        self.bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.subject_dense = tf.keras.layers.Dense(units=2, activation='sigmoid')

    @tf.function
    def call(self):
        pass




