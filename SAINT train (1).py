#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import psutil
import joblib
import random
import logging
from tqdm import tqdm

import numpy as np#matlab
import gc
import pandas as pd#execl
import time
import pickle

from sklearn.metrics import roc_auc_score#机器学习库，评价标准auc
from sklearn.preprocessing import QuantileTransformer#评价transformer模型

import torch#深度学习的框架，facebook
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os


# In[ ]:


#超参数，用户自定义
#普通参数，训练参数
#以下全是超参数
MAX_SEQ = 100#序列长度，自然语言处理transformer模型不善于处理长序列，将序列切分，最大长度是100
D_MODEL = 128 #向量的长度，或者使用256
N_LAYER = 2#层数，encoder的层数
BATCH_SIZE = 256#256为1批数据
DROPOUT = 0.1#神经网络改进参数，随机丢掉一些神经元不使用，每次丢掉10%神经云，防止某一神经元过于粗大
NUM_WORKERS = 0#cpu核心，win使用0


# In[ ]:


train_df = pickle.load(open("D:/kaggle/input/riiid-test-answer-prediction/cv_data/cv1_train.pickle","rb")) # 这个数据在百度网盘
question = pd.read_csv("D:/kaggle/input/riiid-test-answer-prediction/questions.csv")
#数据导入，pickle转换了二进制时间比较快；把数据切割了一部分，训练集的一部分，另外一部分是自己的验证集。cv表示交叉验证


# In[ ]:



def feature_time_lag(df, time_dict):
    '''
    生成time_lag特征
    '''
    tt = np.zeros(len(df), dtype=np.int64)
    for ind, row in enumerate(df[['user_id','timestamp','task_container_id']].values):
        if row[0] in time_dict.keys():#row[0]:user_id检测是否在dic里
            if row[2]-time_dict[row[0]][1] == 0:#如果time_lag=0
                tt[ind] = time_dict[row[0]][2]#更新index
            else:
                t_last = time_dict[row[0]][0]#计算上一次的time_lag
                task_ind_last = time_dict[row[0]][1]#上一次task_id的index值
                tt[ind] = row[1]-t_last #做时间差存给df
                time_dict[row[0]] = (row[1], row[2], tt[ind])#更新给index
        else:
            # time_dict : timestamp, task_container_id, lag_time
            time_dict[row[0]] = (row[1], row[2], -1)
            tt[ind] =  0
    df["time_lag"] = tt
    return df

time_dict = dict()#time字典，存time_lag特征 key:user_id value:(timestamp,contianer_id,ind(index))最后一次做题的时间，用于测试时算time lag
train_df = feature_time_lag(train_df, time_dict) # 生成time_lag特征
#df相当于一个二维的excel表格
pickle.dump(time_dict,open("D:/kaggle/input/riiid-test-answer-prediction/time_dict.pkl","wb")) # 保存到本地，inference时要用的
del time_dict


# In[ ]:

#选择列
train_df = train_df[["timestamp","user_id","content_id","content_type_id","answered_correctly","prior_question_elapsed_time","prior_question_had_explanation","time_lag"]]
train_df = train_df[train_df.content_type_id == 0] # 选择行 去掉讲座部分，只保留题目部分

train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.fillna(0) # 用0填充空值（新用户无上次数据）
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(False).astype(int) # 用false填充空值（看解释）


# In[ ]:


#merge question.csv两张表连接 前为主表
train_df = train_df.merge(question[["question_id","part"]], how = "left", left_on = 'content_id', right_on = 'question_id') 


# In[ ]:


# 切分数据集97.5:2.5
train = train_df.iloc[:int(97.5/100 * len(train_df))]
val = train_df.iloc[int(97.5/100 * len(train_df)):]
print(train.shape,val.shape)


# In[ ]:


skills = train["content_id"].unique()#把content_id里的唯一值挑出来
n_skill = len(skills)#有多少道题
print("number skills", len(skills))


# In[ ]:


n_part = len(train["part"].unique())
print(n_part)


# In[ ]:


del train_df 
gc.collect()


# In[ ]:

#把每个题都变成元祖的的形式
train_group = train[['user_id', 'content_id', 'answered_correctly', 'part', 'prior_question_elapsed_time', 'time_lag', 'prior_question_had_explanation']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values,
            r['time_lag'].values,
            r['prior_question_had_explanation'].values))


# In[ ]:


val_group = val[['user_id', 'content_id', 'answered_correctly', 'part', 'prior_question_elapsed_time', 'time_lag', 'prior_question_had_explanation']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values,
            r['part'].values,
            r['prior_question_elapsed_time'].values,
            r['time_lag'].values,
            r['prior_question_had_explanation'].values))


# In[ ]:


all_group = pd.concat([train_group,val_group])
pickle.dump(all_group,open("D:/kaggle/input/riiid-test-answer-prediction/group.pkl","wb")) # inference时要用的


# In[ ]:


class SAINTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):#初始化 group，题目数量，最大序列数
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q, qa, part, pri_elap, lag, pri_exp = group[user_id]
            if len(q) < 2: #样本过少不参考
                continue
            
            # Main Contribution
            if len(q) > self.max_seq:#如果组大于最大序列数（100），将组切分，最大长度是100
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial], part[:initial], pri_elap[:initial], lag[:initial], pri_exp[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end], part[start:end], pri_elap[start:end], lag[start:end], pri_exp[start:end])
            #115 q qa (q,qa都大于100）切分为
            #115-1 100
            #115-2 q-100
            else:#<100不切分
                user_id = str(user_id)
                self.user_ids.append(user_id) #user_id加入数组
                self.samples[user_id] = (q, qa, part, pri_elap, lag, pri_exp)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, part_, pri_elap_, lag_, pri_exp_ = self.samples[user_id]
        seq_len = len(q_)

        ## for zero padding 序列表示使之从1开始
        q_ = q_+1
        pri_exp_ = pri_exp_ + 1
        res_ = qa_ + 1
        
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        res = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        pri_elap = np.zeros(self.max_seq, dtype=float)
        lag = np.zeros(self.max_seq, dtype=float)
        pri_exp = np.zeros(self.max_seq, dtype=int)

        if seq_len == self.max_seq:#满100的部分

            q[:] = q_
            qa[:] = qa_
            res[:] = res_
            part[:] = part_
            pri_elap[:] = pri_elap_
            lag[:] = lag_
            pri_exp[:] = pri_exp_
            
        else:#不够100的部分，从后面开始填充，前面全是0
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            res[-seq_len:] = res_
            part[-seq_len:] = part_
            pri_elap[-seq_len:] = pri_elap_
            lag[-seq_len:] = lag_
            pri_exp[-seq_len:] = pri_exp_
        
        exercise = q[1:]#预测时第一个值不需要
        part = part[1:]
        response = res[:-1]#最后一个值不要
        label = qa[1:]
        elap = pri_elap[1:]

        ## It's different from paper. The lag time including present lag time have more information. 
        lag = lag[1:]
        pri_exp = pri_exp[1:]


        return exercise, part, response, elap, lag, pri_exp, label


# In[ ]:


train_dataset = SAINTDataset(train_group, n_skill)#一行一行
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
#数据管道批量传给模型 shuffle=True打乱顺序
val_dataset = SAINTDataset(val_group, n_skill)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# In[ ]:


item = val_dataset.__getitem__(3)


# In[ ]:


item # item格式


# In[ ]:


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):#上三角函数
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAINTModel(nn.Module):
    def __init__(self, n_skill, n_part, max_seq=MAX_SEQ, embed_dim= 128, time_cat_flag = True):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part
        self.time_cat_flag = time_cat_flag#时间是否分类continous和category

        self.e_embedding = nn.Embedding(self.n_skill+1, embed_dim) ## exercise
        self.c_embedding = nn.Embedding(self.n_cat+1, embed_dim) ## category
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim) ## position
        self.res_embedding = nn.Embedding(2+1, embed_dim) ## response


        if self.time_cat_flag == True:
            self.elapsed_time_embedding = nn.Embedding(300+1, embed_dim) ## elapsed time (the maximum elasped time is 300)
            self.lag_embedding1 = nn.Embedding(300+1, embed_dim) ## lag time1 for 300 seconds
            self.lag_embedding2 = nn.Embedding(1440+1, embed_dim) ## lag time2 for 1440 minutes
            self.lag_embedding3 = nn.Embedding(365+1, embed_dim) ## lag time3 for 365 days

        else:
            self.elapsed_time_embedding = nn.Linear(1, embed_dim, bias=False) ## elapsed time
            self.lag_embedding = nn.Linear(1, embed_dim, bias=False) ## lag time


        self.exp_embedding = nn.Embedding(2+1, embed_dim) ## user had explain
        #引入transformer模型
        self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = DROPOUT)
#nhead是多头的层数
        self.dropout = nn.Dropout(DROPOUT)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        
    def tasks_3d_mask(self, tasks, diagonal=1):#做3d mask
        mask_3d = [self.tasks_mask(t, seq_length, diagonal=diagonal) for t in tasks]
        mask_3d = torch.stack(mask_3d, dim=0)
        # Need BS*num_heads shape
        repeat_3d = [mask_3d for t in range(self.nhead)]
        repeat_3d = torch.cat(repeat_3d)
        return repeat_3d
    
    def forward(self, question, part, response, elapsed_time, lag_time, exp):

        device = question.device  

        ## embedding layer
        question = self.e_embedding(question)
        part = self.c_embedding(part)
        pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        res = self.res_embedding(response)
        exp = self.exp_embedding(exp)

        if self.time_cat_flag == True:

            ## feature engineering
            ## elasped time
            elapsed_time = torch.true_divide(elapsed_time, 1000)#毫秒变秒 以秒记
            elapsed_time = torch.round(elapsed_time)#变成小数
            elapsed_time = torch.where(elapsed_time.float() <= 300, elapsed_time, torch.tensor(300.0).to(device)).long()#传给cpu必须用浮点形式不能用整数
            elapsed_time = self.elapsed_time_embedding(elapsed_time)

            ## lag_time1
            lag_time = torch.true_divide(lag_time, 1000)
            lag_time = torch.round(lag_time)
            lag_time1 = torch.where(lag_time.float() <= 300, lag_time, torch.tensor(300.0).to(device)).long()

            ## lag_time2
            lag_time = torch.true_divide(lag_time, 60)#以分钟记
            lag_time = torch.round(lag_time)
            lag_time2 = torch.where(lag_time.float() <= 1440, lag_time, torch.tensor(1440.0).to(device)).long()

            ## lag_time3
            lag_time = torch.true_divide(lag_time, 1440)#以天记
            lag_time = torch.round(lag_time)
            lag_time3 = torch.where(lag_time.float() <= 365, lag_time, torch.tensor(365.0).to(device)).long()

            ## lag time
            lag_time1 = self.lag_embedding1(lag_time1) 
            lag_time2 = self.lag_embedding2(lag_time2) 
            lag_time3 = self.lag_embedding3(lag_time3)
            
            enc = question + part + pos_id + exp
            dec = pos_id + res + elapsed_time + lag_time1 + lag_time2 + lag_time3
  

        else:

            elapsed_time = elapsed_time.view(-1,1)
            elapsed_time = self.elapsed_time_embedding(elapsed_time)
            elapsed_time = elapsed_time.view(-1, MAX_SEQ-1, self.embed_dim)

            lag_time = lag_time.view(-1,1)
            lag_time = self.lag_embedding(lag_time)
            lag_time = lag_time.view(-1, MAX_SEQ-1, self.embed_dim)

            enc = question + part + pos_id + exp
            dec = pos_id + res + elapsed_time + lag_time
        

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]#格式转换
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = SAINTModel(n_skill, n_part, embed_dim= D_MODEL, time_cat_flag = True)

## AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()

model.to(device)
criterion.to(device)


# In[ ]:


print(model)


# In[ ]:

#训练
def train_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device="cpu", time_cat_flag = True):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []#记录循环次数
    outs = []

    start_time = time.time()

    ## training
    for item in train_dataloader:
        exercise = item[0].to(device).long() #q
        part = item[1].to(device).long() #qa
        response = item[2].to(device).long() #

        if time_cat_flag == True:
            elapsed_time = item[3].to(device).long()
            lag_time = item[4].to(device).long()
        else :
            elapsed_time = item[3].to(device).float()
            lag_time = item[4].to(device).float()

        exp = item[5].to(device).long()
        label = item[6].to(device).float()
        target_mask = (exercise != 0)

        optimizer.zero_grad()
        output = model(exercise, part, response, elapsed_time, lag_time, exp)
        
        loss = criterion(output, label)
        loss.backward() #反向更新参数改善loss
        optimizer.step()
        train_loss.append(loss.item())
        
        # mask the output
        output_mask = torch.masked_select(output, target_mask)
        label_mask = torch.masked_select(label, target_mask)

        labels.extend(label_mask.view(-1).data.cpu().numpy())
        outs.extend(output_mask.view(-1).data.cpu().numpy())

    train_auc = roc_auc_score(labels, outs)#评价体系 官方指定
    train_loss = np.mean(train_loss)

    labels = []
    outs = []
    val_loss = []

    # validation
    model.eval()
    for item in val_dataloader:
        exercise = item[0].to(device).long()
        part = item[1].to(device).long()
        response = item[2].to(device).long()

        if time_cat_flag == True: 实验true更好
            elapsed_time = item[3].to(device).long()
            lag_time = item[4].to(device).long()
        else :
            elapsed_time = item[3].to(device).float()
            lag_time = item[4].to(device).float()

        exp = item[5].to(device).long()
        label = item[6].to(device).float()
        target_mask = (exercise != 0)
        
        output = model(exercise, part, response, elapsed_time, lag_time, exp)
        
        ## mask the output
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        
        loss = criterion(output, label)
        val_loss.append(loss.item())

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    val_auc = roc_auc_score(labels, outs)
    val_loss = np.mean(val_loss)

    run_time = time.time() - start_time #运行时间

    return train_loss, train_auc, val_loss, val_auc, run_time


# In[ ]:


logging.basicConfig(level=logging.DEBUG, filename="logfile20.txt", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")


# In[ ]:


# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# epochs = 1


# In[ ]:


epochs = 10#所有的数据都跑一次
for epoch in range(epochs):
    train_loss, train_auc, val_loss, val_auc, run_time = train_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device, time_cat_flag = True)
    print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, elapsed_time))
    logging.info("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, elapsed_time))


# In[ ]:


torch.save(model.state_dict(), "D:/kaggle/input/riiid-test-answer-prediction/saint_plus_model.pt")
采用分类法（cat）
论文里给的t2-t3 实际没给t2 用的t1-t3
transfomer q key value