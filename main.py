# -*- coding: utf-8 -*-
# @Time : 2021/8/21 20:55 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py 
# @Software: PyCharm
from engines.utils.logger import get_logger
from engines.data import DataManager
from engines.train import train


logger = get_logger('./logs')
data_manager = DataManager(logger)
train(data_manager, logger)

