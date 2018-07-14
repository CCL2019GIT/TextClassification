# -*- coding: UTF-8 -*-
"""
===============================================================
author：xieqiqi
email：xieqiqi@jd.com
date：2018
introduction:
===============================================================
"""
class Config:
    '''
    Config for this project!
    '''
    def __init__(self,class_nums,vocab_size,embed_dim,hidden_dim,droup_out_prob,
                 droup_out_use,inChannels,num_filters,filter_size,epoch_nums,is_droupOut):
        '''
        Base information!
        '''
        #
        self.class_nums = class_nums
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.droup_out_prob = droup_out_prob
        self.inChannels = inChannels
        self.num_filters = num_filters
        self.filter_size = filter_size
        #
        self.epoch_nums = epoch_nums
        self.is_droupOut=is_droupOut
