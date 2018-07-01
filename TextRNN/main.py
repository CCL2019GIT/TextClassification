# -*- coding: UTF-8 -*-
"""
===============================================================
author：XieDake
email：DakeXqq@126.com
date：2018
introduction:
            Main Program for TextRNN!
===============================================================
"""
import torch
import argparse,os,pickle
from models import TextRNN,TextRNN_AM_short
from data_helper import load_data,word2Id_id2Word,train_val_split
from train import train_one_epoch,eval_after_one_epoch,confusion_matrix

def parse_arguments():
    parse = argparse.ArgumentParser(description='Hyperparams of this project!')
    #
    parse.add_argument('--hidden_dim', type=int, default=128,help='Hidden dim of encoder!')
    parse.add_argument('--embed_dim', type=int, default=200,help='Embed dim of encoder!')
    parse.add_argument('--droup_out_prob', type=float, default=0.2, help='droup out keep prob!')
    parse.add_argument('--class_nums', type=int, default=3, help='classes num of classification task!')
    parse.add_argument('--bi_direction', type=bool, default=False, help='Whether using Bi_direction!')
    parse.add_argument('--droup_out_use', type=bool, default=False, help='Whether using droup out!')
    #
    parse.add_argument('--epochs', type=int, default=10,help='number of epochs for train')
    parse.add_argument('--batch_size', type=int, default=100,help='number of epochs for train')
    parse.add_argument('--lr', type=float, default=0.005,help='initial learning rate')
    parse.add_argument('--use_AM', type=bool, default=True, help='Whether using Attention Mechanisim')
    #
    parse.add_argument('--Base_path', type=str, default='data/', help='Base path!')
    parse.add_argument('--Save_path', type=str, default='data/save_test_00/',help='Save path!')
    #
    parse.add_argument('--mode',type=str,default='train',help='Type of mode!')
    #
    return parse.parse_args()
#
args=parse_arguments()
print("===============================================================")
print("Path setting...")
source_file_name=os.path.join(args.Base_path,"data_5q")
w2Id_save_fileName=os.path.join(args.Save_path,'w2Id')
i2Wd_save_fileName=os.path.join(args.Save_path,'id2Wd')

model_save_fileName=os.path.join(args.Save_path,'TextRNN.pt')

save_path=args.Save_path
if(not os.path.exists(save_path)):
    os.mkdir(save_path)

print("===============================================================")
print("Loading data and Data processing!")
data_filter=load_data(source_data_fileName=source_file_name)

vocab_size=word2Id_id2Word(data=data_filter,
                           w2Id_save_fileName=w2Id_save_fileName,
                           i2Wd_save_fileName=i2Wd_save_fileName)

with open(w2Id_save_fileName, "rb") as fr:
    w2Id = pickle.load(fr)

print("=============================定义Model网络===============================")
print("Models initializing....")
if(args.use_AM==True):
    model = TextRNN_AM_short(class_nums=args.class_nums,
                             vocab_size=vocab_size,
                             embed_dim=args.embed_dim,
                             hidden_dim=args.hidden_dim,
                             droup_out_prob=args.droup_out_prob,
                             bi_direction=args.bi_direction,
                             droup_out_use=args.droup_out_use)
else:
    model = TextRNN(class_nums=args.class_nums,
                    vocab_size=vocab_size,
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    droup_out_prob=args.droup_out_prob,
                    bi_direction=args.bi_direction,
                    droup_out_use=args.droup_out_use)
#
if(torch.cuda.is_available()):
    model=model.cuda()
print("Structure of Model....")
print(model)
#
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.0001)
#
print("=============================Model_Training!===============================")
if(args.mode=="train"):
    # train_val_split
    train_data_filter, val_data_filter = train_val_split(data_filter=data_filter, ratio=0.80)
    for epoch in range(args.epochs):
        #train one epoch!
        train_one_epoch(epoch_num=epoch,model=model,
                        optimizer=optimizer,
                        data_train=train_data_filter,
                        batch_size=args.batch_size,w2id=w2Id)
        #
        avg_val_loss,result=eval_after_one_epoch(model=model,data_val=val_data_filter,w2id=w2Id)
        print("After training on epoch:{},Model performance on val_set——>avg_val_loss:{}".format({epoch},{avg_val_loss.data[0]}))
        print("==================== Model of Classifier performance showing:====================")
        confusion_matrix(result)
        print("=================================================================================")
    #save model!s
    print("Saving model...!")
    torch.save(model,model_save_fileName)
    print("Training stop!")