# -*- coding: utf-8 -*-
# @Time : 2021/8/29 23:08 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py 
# @Software: PyCharm
from abc import ABC
from transformers import TFBertModel
from tensorflow.keras.layers import LayerNormalization
import tensorflow as tf


class ConditionalLayerNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        # self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.beta = self.add_weight(shape=hidden_size, initializer='zeros', name='beta')
        self.gamma = self.add_weight(shape=hidden_size, initializer='ones', name='gamma')
        self.variance_epsilon = eps

        self.beta_dense = tf.keras.layers.Dense(hidden_size)
        self.gamma_dense = tf.keras.layers.Dense(hidden_size)

    # @tf.function
    def call(self, x, cond):
        cond = tf.expand_dims(cond, 1)
        beta = self.beta_dense(cond) + self.beta
        gamma = self.gamma_dense(cond) + self.gamma

        mean = tf.reduce_mean(x, -1, keepdims=True)
        variance = tf.reduce_mean(tf.pow((x - mean), 2), -1, keepdims=True)
        std = tf.sqrt(variance + self.variance_epsilon)
        outputs = (x - mean) / std * gamma + beta
        return outputs


class Model(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()
        self.bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.subject_dense = tf.keras.layers.Dense(units=2, activation='sigmoid')
        self.cond_layer_norm = ConditionalLayerNorm(768)

    @staticmethod
    def extract_subject(sequence_output, subject_ids):
        """
        根据subject_ids从output中取出subject的首尾token的向量表征融合
        """
        start = tf.gather(sequence_output, subject_ids[:, :1], batch_dims=1)
        end = tf.gather(sequence_output, subject_ids[:, 1:], batch_dims=1)
        subject = tf.concat([start, end], axis=1)
        return subject

    # @tf.function
    def call(self, sentences, attention_mask, subject_ids):
        bert_hidden_states = self.bert_model(sentences, attention_mask=attention_mask)
        sequence_output = bert_hidden_states[0]
        layer_norm_result = self.layer_norm(sequence_output)
        subject_predicts = self.subject_dense(layer_norm_result)

        subject_encode = self.extract_subject(sequence_output, subject_ids)
        cond_layer_norm_result = self.cond_layer_norm(sequence_output, subject_encode)
        print('asd')
















