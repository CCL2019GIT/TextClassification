# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            TextRNN in pytorch!——>models!
            no Minibatch support!
===============================================================
"""
import torch
import math
import torch.nn.functional as F

class TextRNN(torch.nn.Module):
    '''
    TextRNN for text classification!
    Bi-GRU
    '''
    def __init__(self,class_nums,vocab_size,embed_dim,hidden_dim,droup_out_prob,bi_direction,droup_out_use):
        super(TextRNN,self).__init__()
        #Base parameters and structures!
        #
        self.class_nums=class_nums
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.droup_out_prob=droup_out_prob
        self.vocab_size=vocab_size
        self.bi_direction=bi_direction
        self.droup_out_use=droup_out_use
        #
        self.embeddings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        if(self.droup_out_use):
            self.droup_out = torch.nn.Dropout(self.droup_out_prob)

        if(self.bi_direction):
            self.bi_gru = torch.nn.GRU(self.embed_dim, self.hidden_dim, bidirectional=True,dropout=self.droup_out_prob) #双向GRU！
            self.bi_out = torch.nn.Linear(self.hidden_dim * 2, self.class_nums)
        else:
            self.s_gru=torch.nn.GRU(self.embed_dim,self.hidden_dim,dropout=self.droup_out_prob) #单向GRU！
            self.s_out = torch.nn.Linear(self.hidden_dim, self.class_nums)
        self.softmax=torch.nn.LogSoftmax()

    def forward(self,seq_input,hidden=None):
        '''
        No minibatch!
        batch_input:[H]:H=1或者任意长度是一样的！
        '''
        input_embed=self.embeddings(seq_input)#[H,embed_dim]
        #
        if(self.droup_out_use):
            input_embed = self.droup_out(input_embed)
        #
        input_embed=input_embed.unsqueeze(1)#[H,1,embed_dim]
        if(self.bi_direction):
            # Bi-Gru:
            outPut, hidden = self.bi_gru(input_embed, hidden)
            hidden_concat = torch.cat((hidden[0], hidden[1]), 1)  # [1,1,2*hidden_dim]——>[1,2*hidden_dim]
            outPut = self.bi_out(hidden_concat)  # [1,class_nums]
        else:
            #S-Gru:
            outPut, hidden = self.s_gru(input_embed, hidden)
            outPut = self.s_out(hidden.squeeze(0))  # [1,class_nums]

        out=self.softmax(outPut)# [1,class_nums]
        #
        return out,hidden

class TextRNN_AM_short(torch.nn.Module):
    '''
    TextRNN with word_level attention!
    Generate sent_vector!
    No batch!
    '''
    def __init__(self,class_nums,vocab_size,embed_dim,hidden_dim,droup_out_prob,bi_direction,droup_out_use):
        super(TextRNN_AM_short,self).__init__()
        #Base parameters and structures!
        #
        self.class_nums=class_nums
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.droup_out_prob=droup_out_prob
        self.vocab_size=vocab_size
        self.bi_direction=bi_direction
        self.droup_out_use=droup_out_use
        #
        self.embeddings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        if(self.droup_out_use):
            self.droup_out = torch.nn.Dropout(self.droup_out_prob)
        #
        if(self.bi_direction):
            #双向GRU！
            self.gru = torch.nn.GRU(self.embed_dim, self.hidden_dim, bidirectional=True,dropout=self.droup_out_prob)
            self.attn = torch.nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
            self.out = torch.nn.Linear(2 * self.hidden_dim, self.class_nums)
        else:
            #单向GRU！
            self.gru=torch.nn.GRU(self.embed_dim,self.hidden_dim,dropout=self.droup_out_prob)
            self.attn = torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
            self.out = torch.nn.Linear(self.hidden_dim, self.class_nums)
        #u:word_context_vector
        self.u=torch.nn.Parameter(torch.rand(2*self.hidden_dim))
        bound=1./(math.sqrt(self.u.size(0)))
        self.u.data.uniform_(-bound,bound)
        #
        self.softmax=torch.nn.LogSoftmax()

    def forward(self,seq_input,hidden=None):
        '''
        No minibatch!
        batch_input:[H]:H=1或者任意长度是一样的！
        '''
        input_embed=self.embeddings(seq_input)#[T,embed_dim]
        #
        if(self.droup_out_use):
            input_embed = self.droup_out(input_embed)
        #
        input_embed=input_embed.unsqueeze(1)#[T,1,embed_dim]
        #
        if(self.bi_direction):
            # Bi-Gru:
            #outPut:[T,B=1,2*H]
            #hidden:[2,B=1,H]
            outPut, hidden = self.gru(input_embed, hidden)
            attn_energy = self.attn(outPut.squeeze(1))#[T,2*H]
        else:
            #S-Gru:
            #outPut:[T,B=1,H]
            #hidden:[1,B=1,H]
            outPut, hidden = self.gru(input_embed, hidden)
            attn_energy = self.attn(outPut.squeeze(1))#[T,2*H]
        #
        #1*1*2H_1*2H*T=1*1*T
        score=torch.bmm(self.u.unsqueeze(0).unsqueeze(0),attn_energy.transpose(0,1).unsqueeze(0))#1*1*T
        weight=F.softmax(score)#[1,1,T]
        #
        #sentence_vector:
        # 双向：[1,1,T]*[1,T,2H]->[1,1,2H]
        # 单向：[1,1,T]*[1,T,H]->[1,1,H]
        sent_vector=torch.bmm(weight,outPut.transpose(0,1))
        #
        outPut=self.out(sent_vector.squeeze(0))#[]
        # 双向：[1,1,2H]->[1,2H]——>[1,class_nums]
        # 单向：[1,1,H]->[2,H]——>[1,class_nums]
        out=self.softmax(outPut)# [1,class_nums]
        #
        return out,hidden