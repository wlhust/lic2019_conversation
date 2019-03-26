# -*- coding:utf-8 -*-
# @Author: wangli
# @Date: 2019-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from config import config
from preprocess import Vocab, Preprocess, dataset
from model import Encoder, Attention, Decoder
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    print('==> Loading config......')
    cfg = config()
    print('==> Preprocessing data......')
    voc = Vocab(cfg)
    voc.gen_counter_dict()
    voc.gen_vocab()
    cfg.vocab_len = voc.vocab_len
    print('The length of vocab is: {}'.format(cfg.vocab_len))

    prep = Preprocess(cfg, voc.vocab)
    pairs = prep.gen_pair_sen()
    print('pairs sentences generated.')
    pairs = prep.tokenize(pairs)
    print('sentences tokenized.')

    traindataset = dataset(pairs, voc.vocab)
    traindataloader = DataLoader(traindataset, batch_size=5, shuffle=False)
    one_iter = iter(traindataloader).next()
    
    encoder = Encoder(cfg)
    encoder_outputs, _ = encoder(one_iter['inputs'], one_iter['length'])
    
