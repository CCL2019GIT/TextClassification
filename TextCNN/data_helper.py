# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            data_helper for TextCNN in pytorch!
            No minibatch!
===============================================================
"""
import torch
import pickle,random
import numpy as np
from torch.autograd import Variable

PAD_token=0

def load_data(source_data_fileName):
    '''
    loading data!
    注意data格式固定：
                    ...     ...

                    label   text

                    ...     ...
    return:[[label1,sent1],[label2,sent2]...]
    '''
    data=[]
    max_seq_length=0
    with open(source_data_fileName,'r',encoding='UTF-8')as r:
        lines=r.readlines()
        print("data size:{}".format({len(lines)}))
        for index,line in enumerate(lines):
            lsp=line.strip().lower().split('\t')
            if(len(lsp)!=2):
                print("data format error at:{}".format({index}))
                continue
            else:
                #
                sent_label=[]
                sent_label.append(lsp[0])#label
                sent_label.append(lsp[1])#sent
                #
                if(max_seq_length<len(lsp[1])):
                    max_seq_length=len(lsp[1])
                #
                data.append(sent_label)
    #
    return data,max_seq_length

def word2Id_id2Word(data,w2Id_save_fileName,i2Wd_save_fileName):
    '''
    word2ID dict 保存！
    index2Wd dict 保存！应该没啥用！
    '''
    #
    w2Id={}
    i2Wd={}
    #
    w2Id['PAD']=0
    i2Wd[0]='PAD'
    #
    for l_s in data:
        for wd in l_s[1]:
            if(wd in w2Id):
                continue
            else:
                w2Id[wd]=len(w2Id)
                i2Wd[w2Id[wd]]=wd
    #save
    print("Saving word2ID dict...!")
    with open(w2Id_save_fileName, 'wb') as fw:
        pickle.dump(w2Id, fw)
    print("Saving index2Wd dict...!")
    with open(i2Wd_save_fileName, 'wb') as fw:
        pickle.dump(i2Wd, fw)
    #
    vocab_size=len(w2Id)
    #
    print("vocab size:{}".format(vocab_size))
    #
    return vocab_size

def train_val_split(data_filter,ratio):
    '''
    train and val data set split!
    T:V=8:2
    '''
    #shuffle
    random.shuffle(data_filter)
    #
    data_size=len(data_filter)
    split_point=round(data_size*ratio)
    #
    train_data_filter=data_filter[:split_point]
    val_data_filter = data_filter[split_point:]
    #
    print("All data size:{},split val:train->{}".format(len(data_filter),(len(val_data_filter))/(len(train_data_filter))))
    #
    return train_data_filter,val_data_filter

def sent2Id(sent_label,w2id):
    '''
    One sentence to id!
    '''
    label = [int(sent_label[0])]
    sent2id=[]
    for char in sent_label[1]:
        sent2id.append(w2id[char])
    #
    return sent2id,label

def pad_sent(seq,max_length):
    '''
    每一块Batch数据进行Pading！
    '''
    seq+=[PAD_token for i in range(max_length-len(seq))]
    return seq

def batch_yeild(sent_labels,batch_size,max_seq_length,wd2id):
    '''
    每一个batch记得要PAD一下！
    维度要求：B*seqLen
    '''
    random.shuffle(sent_labels)
    seq_batch = []
    label_batch=[]
    for sent_label in sent_labels:
        #
        seq_batch.append([wd2id[wd] for wd in sent_label[1]])
        label_batch.append([int(sent_label[0])])
        #
        if len(seq_batch) == batch_size:
            #Padding
            source_batch_pad=[pad_sent(s,max_seq_length) for s in seq_batch]
            #Variable
            if torch.cuda.is_available():
                seq_batch_pad=Variable(torch.LongTensor(source_batch_pad)).cuda()
                label_batch=Variable(torch.LongTensor(label_batch)).cuda()
            else:
                seq_batch_pad=Variable(torch.LongTensor(source_batch_pad))
                label_batch=Variable(torch.LongTensor(label_batch))
            #
            yield seq_batch_pad,label_batch
            seq_batch, label_batch = [], []
    if len(seq_batch) != 0:
        # Padding
        source_batch_pad = [pad_sent(s, max_seq_length) for s in seq_batch]
        # Variable
        if torch.cuda.is_available():
            seq_batch_pad = Variable(torch.LongTensor(source_batch_pad)).cuda()
            label_batch = Variable(torch.LongTensor(label_batch)).cuda()
        else:
            seq_batch_pad = Variable(torch.LongTensor(source_batch_pad))
            label_batch = Variable(torch.LongTensor(label_batch))
        #
        yield seq_batch_pad, label_batch

def generate_one_sample(sent_label,w2id):
    '''
    Input sequence!
    Output label!
    '''
    #
    sent2id, label=sent2Id(sent_label=sent_label,w2id=w2id)
    #
    seq_input=Variable(torch.from_numpy(np.array(sent2id)))
    seq_output=Variable(torch.from_numpy(np.array(label)))
    #
    if(torch.cuda.is_available()):
        seq_input=seq_input.cuda()
        seq_output=seq_output.cuda()
    #
    return seq_input, seq_output