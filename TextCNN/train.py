# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
===============================================================
"""
import torch
from sklearn.metrics import classification_report
import numpy as np

def train_one_epoch_batch(epoch_num,model,optimizer,batch_iter):
    '''
    完成一个epoch数据训练！
    每一个batch进行一次BP！每一个Batch，print训练信息！
    '''
    criterion = torch.nn.NLLLoss()
    for step, batch_data in enumerate(batch_iter):
        #
        batch_loss=0.0
        #seq_batch_pad,label_batch
        seq_batch_pad = batch_data[0]  # [B,max_seq_len]
        label_batch = batch_data[1] # [B,1]
        #
        bsz = seq_batch_pad.size(0)
        #
        optimizer.zero_grad()
        #
        predict=model(seq_batch_pad) # [B,class_nums]
        for pos in range(bsz):
            batch_loss += criterion(predict[pos].view(1,predict[pos].data.size(0)),label_batch[pos])
        # backwards
        batch_loss.backward()
        # 参数个更新
        optimizer.step()
        # 每个batch：print损失loss信息！
        print("At epoch:{},batch:{}——>loss_avg:{}".format({epoch_num}, {step}, {batch_loss.data[0] / bsz}))

def eval_after_one_epoch_batch(model,batch_iter):
    '''
    Evaluting after one epoch on validation set!
    '''
    criterion = torch.nn.NLLLoss()
    total_loss=0.0
    count=0
    #
    result=[] #[[y_true,y_pred],[y_true,y_pred],...]
    #
    for step, batch_data in enumerate(batch_iter):
        #
        #seq_batch_pad,label_batch
        seq_batch_pad = batch_data[0]  # [B,max_seq_len]
        label_batch = batch_data[1] # [B,1]
        #
        bsz = seq_batch_pad.size(0)
        count+=bsz
        #
        predict = model(seq_batch_pad) # [B,class_num]
        for pos in range(bsz):
            total_loss += criterion(predict[pos].view(1,predict[pos].data.size(0)),label_batch[pos])
            tmp=[]
            tmp.append(label_batch[pos].data[0])  # y_true
            pred_y = torch.max(predict[pos].view(1,predict[pos].data.size(0)), 1)[1].data[0]
            tmp.append(pred_y)
            result.append(tmp)
        #
    # 每个batch：print损失loss信息！
    print("Validation after training one epoch——>loss_avg:{},count:{}".format({total_loss.data[0] / count},{count}))
    #
    return result

def confusion_matrix(result):
    '''
    测试集分类效果展示：Confusion_Matrix!
    '''
    result=np.array(result)
    print(classification_report(y_true=result[:,0],y_pred=result[:,-1]))