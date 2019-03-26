# -*- coding:utf-8 -*-
# @Author: wangli
# @Date: 2019-03-26

import os

class config():
    def __init__(self):
        # data
        self.max_len = 15
        self.train_path = './train_part.txt'
        self.eval_path = None
        self.test_path = None
        
        # save dict path
        self.save_dict_path = './source/saved_dict'
        self.counter_path = os.path.join(self.save_dict_path, 'couter_dict.pkl')
        self.vocab_path = os.path.join(self.save_dict_path, 'vocab_dict.pkl')
        
        # net
        self.embedding_size = 512
        self.gru_hidden_size = 128
        self.gru_layers = 1
        self.bidirection = True
        