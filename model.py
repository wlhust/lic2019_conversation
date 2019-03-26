# -*- coding:utf-8 -*-
# @Author: wangli
# @Date: 2019-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
# cfg = config()

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(cfg.vocab_len, cfg.embedding_size)
        self.gru = nn.GRU(input_size=cfg.embedding_size, hidden_size=cfg.gru_hidden_size,
                         num_layers=cfg.gru_layers, bidirectional=cfg.bidirection)
        
    def forward(self, inputs, in_lengths):
        inputs = self.embedding(inputs)
        outs, hidden = self.gru(inputs.transpose(0,1))
#         packed = torch.nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=in_lengths, batch_first=True)
#         outs, hidden = self.gru(packed)
#         unpack = torch.nn.utils.rnn.pad_packed_sequence(outs)
#         return unpack, hidden
        return outs, hidden

class Attention(nn.Module):
    def __init__(self, method):
        self.method = method
        super(Attention, self).__init__()
        if self.method not in ['dot']:
            raise ValueError(self.method, 'is not a appropriate method. ')
    
    def dot(self, rnn_out, encoder_outputs):
        return torch.sum(rnn_out * encoder_outputs, dim=2)
    
    def forward(self, rnn_out, encoder_outputs):
        # encoder_outputs shape: [seq_len, batch_size, num_directions*hidden_size]
        # rnn_out shape: [1, batch_size, num_directions*hidden_size]
        if self.method == 'dot':
            # attention shape: [seq_len, batch_size]
            attention = self.dot(rnn_out, encoder_outputs)
            attention = attention.t()
            attention = F.softmax(attention, dim=1)
            
        return attention

class Decoder(nn.Module):
    def __init__(self, cfg, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(cfg.vocab_len, cfg.embedding_size)
        self.gru = nn.GRU(inputs=cfg.embedding_size, hidden_size=cfg.gru_hidden_size,
                         num_layers=cfg.gru_layers, bidirectional=cfg.bidirection)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attn = Attention('dot')
    
    def forward(self, decoder_inputs, last_hidden, encoder_outputs):
        # decoder_inputs shape: [batch_size, 1]
        # encoder_outputs shape: [seq_len, batch_size, num_directions*hidden_size]
        # embedded shape: [batch_size, 1, embedding_size]
        embedded = self.embedding(decoder_inputs)
        embedded = self.embedding_dropout(embedded)
        # rnn_inputs shape: [1, batch_size, embedding_size]
        # rnn_outs shape: [1, batch_size, num_directions*hidden_size]
        rnn_outs, hidden = self.gru(embedded.transpose(0,1), last_hidden)
        attention = self.attn(rnn_outs, encoder_outputs)