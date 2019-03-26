# -*- coding:utf-8
# @Author: wangli
# @Date: 2019-03-26

import json
import numpy as np
import pickle
import os
import torch
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from config import config


class Vocab():
    """
    词表类，包括了生成统计词数量，生成词索引表方法
    """

    def __init__(self, cfg):
        self.counter = Counter()
        self.vocab = dict()
        self.vocab_len = 0
        
    def gen_counter_dict(self):
        """
        统计词频，并保存为pickle文件
        """
        
        if os.path.exists(cfg.counter_path):
            self.counter = pickle.load(open(cfg.counter_path, 'rb'))
            print('counter dict loaded.')
        else:
            # mkdir -p 如果上级目录不存在，会首先建立上级目录
            if not os.path.exists(cfg.save_dict_path):
                os.system('mkdir -p ' +  cfg.save_dict_path)
            f = open(cfg.train_path, 'r', encoding='utf-8')
            for line in tqdm(f):
                conversation = ' ' .join(json.loads(line)['conversation'])
                self.counter.update(conversation.split())
            f.close()

            pickle.dump(self.counter, open(cfg.counter_path, 'wb'))
            print('counter dict saved.')
        
    def gen_vocab(self):
        """
        生成词索引表，并保存为pickle文件
        """
        
        if os.path.exists(cfg.vocab_path):
            self.vocab = pickle.load(open(cfg.vocab_path, 'rb'))
            self.vocab_len = len(self.vocab)
            print('vocab dict loaded.')
        else:
            # mkdir -p 如果上级目录不存在，会首先建立上级目录
            if not os.path.exists(cfg.save_dict_path):
                os.system('mkdir -p ' +  cfg.save_dict_path)
            PAD = 0 # padding token
            SOS = 1 # start of sentence
            EOS = 2 # end of sentence
            UNK = 3 # out of vocabulary
            # 按照词频排序
            sort_tmp = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
            # 词表
            vocab_list = list(map(lambda x: x[0], sort_tmp))
            self.vocab_len = len(vocab_list) + 4
            self.vocab.update({'PAD':0, 'SOS': 1, 'EOS': 2, 'UNK': 3})
            self.vocab.update(dict(zip(vocab_list, range(4,self.vocab_len))))

            # 存成二进制的pickle文件
            pickle.dump(self.vocab, open(cfg.vocab_path, 'wb'))
            print('vocab dict saved.')

class Preprocess():
    def __init__(self, cfg, vocab_dict):
        self.vocab_dict = vocab_dict
    
    def gen_pair_sen(self):
        """从一轮对话中生成对话长度减1的pair对。即上一句为输入，下一句为目标输出"""

        pairs = []
        f = open(cfg.train_path, 'r', encoding='utf-8')
        for line in tqdm(f):
            conversation = json.loads(line)['conversation']
            for i in range(len(conversation)-1):
                pairs.append([conversation[i].split(), conversation[i+1].split()])
        return pairs
    
    def cut_pad(self, line):
        """句子长度大于max_len的截掉，小于max_len的补'PAD'"""

        line[0] = line[0] + (cfg.max_len - len(line[0]))*['PAD'] if len(line[0]) < cfg.max_len \
                                                                 else line[0][:cfg.max_len]
        line[1] = line[1] + (cfg.max_len - len(line[1]))*['PAD'] if len(line[1]) < cfg.max_len \
                                                                 else line[1][:cfg.max_len]
        return line
    
    def tokenize(self, pairs):
        return list(map(self.cut_pad, pairs))


class dataset(Dataset):
    def __init__(self, pairs, vocab_dict):
        self.pairs = pairs
        self.vocab_dict = vocab_dict
        # super.__init__()
    
    def __getitem__(self, index):
        inputs, targets = pairs[index]
        inputs = [self.vocab_dict[char] if self.vocab_dict.get(char) or self.vocab_dict.get(char)==0
                          else self.vocab_dict['UNK'] for char in inputs]
        targets = [self.vocab_dict[char] if self.vocab_dict.get(char) or self.vocab_dict.get(char)==0
                          else self.vocab_dict['UNK'] for char in targets]
        i_length = len(inputs) - inputs.count(0)
        t_length = len(targets) - targets.count(0)
        mask = [1]*t_length + [0]*targets.count(0)
        outs = {
            'inputs': inputs,
            'length': i_length,
            'targets': targets,
            'mask': mask
        }
        return {key: torch.tensor(value) for key, value in outs.items()}
        
    def __len__(self):
        return len(self.pairs)