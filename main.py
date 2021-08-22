# -*- coding: utf-8 -*-
# @Time : 2021/8/21 20:55 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py 
# @Software: PyCharm
from engines.utils.logger import get_logger
from engines.data import DataManager


logger = get_logger('./logs')
data_manager = DataManager(logger)
train_dataset, val_dataset = data_manager.get_training_set()
print('asd')
