# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            Train and eval for TextRNN!
===============================================================
"""
import torch
from data_helper import generate_one_sample
from sklearn.metrics import classification_report
import numpy as np
import random

def train_one_epoch(epoch_num,model,optimizer,data_train,batch_size,w2id):
    '''
    完成一个epoch数据训练！
    每一个batch进行一次BP！每一个Batch，print训练信息！
    注意：该函数执行之前data_filter需要乱序！
    '''
    #shuffle
    random.shuffle(data_train)
    data_size=len(data_train)
    criterion = torch.nn.NLLLoss()
    for batchIndex in range(int(data_size / batch_size)):
        optimizer.zero_grad()
        batch_loss= 0.0
        counts = 0
        for step in range(batchIndex * batch_size, min((batchIndex + 1) * batch_size, data_size)):
            input, real_out = generate_one_sample(sent_label=data_train[step],w2id=w2id)
            if torch.cuda.is_available():
                input=input.cuda()
                real_out=real_out.cuda()
            output, hidden = model(input)
            #
            batch_loss += criterion(output,real_out)
            #
            counts += 1
        #
        print("At epoch:{},batch:{}——>loss_avg:{}".format({epoch_num}, {batchIndex},{batch_loss.data[0] / counts}))
        #
        batch_loss.backward()
        #
        optimizer.step()

def eval_after_one_epoch(model,data_val,w2id):
    '''
    每一个epoch结束，进行一次eval！
    分类效果评估！confusion matrix！
    '''
    result=[]#[[y_true,y_pred],[y_true,y_pred],...]
    loss=0.0
    data_size=len(data_val)
    criterion=torch.nn.NLLLoss()
    for step in range(len(data_val)):
        tmp=[]
        tmp.append(data_val[step][0])#y_true
        input, real_out = generate_one_sample(sent_label=data_val[step],w2id=w2id)
        if torch.cuda.is_available():
            input = input.cuda()
            real_out = real_out.cuda()
        output, hidden = model(input)
        #
        pred_y = torch.max(output,1)[1].data[0]
        #
        tmp.append(pred_y)
        result.append(tmp)
        #
        loss += criterion(output, real_out)
    #
    return loss / data_size,result

def confusion_matrix(result):
    '''
    测试集分类效果展示：Confusion_Matrix!
    '''
    result=np.array(result)
    print(classification_report(y_true=result[:,0],y_pred=result[:,-1]))
