# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            TextCnn in pytorch!——>models!
            Minibatch no support!
===============================================================
"""
import torch
import torch.autograd as F

class TextCnn(torch.nn.Module):
    '''
    TextCnn!
    Minibatch support!
    '''
    def __init__(self,Config):
        '''
        初始化！
        '''
        #
        super(TextCnn,self).__init__()
        #Base parameters and structures!
        #
        self.class_nums=Config.class_nums
        self.vocab_size=Config.vocab_size
        self.embed_dim=Config.embed_dim
        self.hidden_dim=Config.hidden_dim
        self.droup_out_prob=Config.droup_out_prob
        self.inChannels=Config.inChannels
        self.num_filters=Config.num_filters
        self.filter_size=Config.filter_size
        self.is_droupOut=Config.is_droupOut
        #
        self.embeddings=torch.nn.Embedding(self.vocab_size,self.embed_dim)
        #
        #定义卷积层与池化层！
        # 2DConv!:三种卷积[3,4,5]
        #
        self.conv_1=torch.nn.Conv2d(in_channels=1,out_channels=self.num_filters,
                                    kernel_size=(self.filter_size[0],self.embed_dim),
                                    stride=1,padding=0)

        self.conv_2=torch.nn.Conv2d(in_channels=1,out_channels=self.num_filters,
                                    kernel_size=(self.filter_size[1],self.embed_dim),
                                    stride=1,padding=0)

        self.conv_3=torch.nn.Conv2d(in_channels=1,out_channels=self.num_filters,
                                    kernel_size=(self.filter_size[2],self.embed_dim),
                                    stride=1,padding=0)
        #pooling_max:
        self.pool_1=torch.nn.MaxPool2d(kernel_size=(self.sequence_length-self.filter_size[0]+1,1),
                                       stride=1,padding=0)
        self.pool_2=torch.nn.MaxPool2d(kernel_size=(self.sequence_length-self.filter_size[1]+1,1),
                                       stride=1,padding=0)
        self.pool_3=torch.nn.MaxPool2d(kernel_size=(self.sequence_length-self.filter_size[2]+1,1),
                                       stride=1,padding=0)
        #

        self.droup=torch.nn.Dropout(self.droup_out_prob)
        #每一种卷积：
        self.out=torch.nn.Linear(self.num_filters*len(self.filter_size),self.class_nums)

    def forward(self, seq_input,hidden=None):
        '''
        TextCNN no minibatch!
        seq_input:[T]
        '''
        # embeding
        T=seq_input.size(0)
        input_embed=self.embeddings(seq_input).view(1,1,T,-1)#[T,embed_dim]->[B=1,channel=1,T,embed_dim]
        # convolution：输入与输出！
        ## Input:(N, C_{in}, H_{in}, W_{in}):[B,in_channels,in_height,in_width]
        ## Output:(N, C_{out}, H_{out}, W_{out}):[B_out,out_channels,out_height,out_width]
        conv_out_1 = self.conv_1(input_embed)
        conv_out_2 = self.conv_2(input_embed)
        conv_out_3 = self.conv_3(input_embed)
        # pooling:输入/输出！注意：pooling不改变channel(厚度)！——>feature selecting!
        ## Input:(N, C, H_{in}, W_{in}):[B,channel,in_height,out_width]
        ## Output:(N, C, H_{out}, W_{out}):[B,channel,out_height,out_width]=[1,num_filters,1,1]
        pool_out_1 = self.pool_1(conv_out_1)
        pool_out_2 = self.pool_2(conv_out_2)
        pool_out_3 = self.pool_3(conv_out_3)
        # merging!pool outPut!

