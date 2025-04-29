import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.nn.utils import weight_norm
from scipy.stats import pearsonr
from scipy import spatial
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    # set device to cpu or cuda
    device = torch.device('cpu')
    print("Device set to : cpu")

file_path =  r'C:\Users\Administrator\Desktop\koopman-data\data\train.xlsx'   # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# print(raw_data)
data=np.array(raw_data)
data_size=len(data)
dataX=data[0:60000,:]
num_train=60000
#设置滑动时间窗口，将数组按时间窗转换成tensor

# te_dataY=t.from_numpy(dataYv)
#数据归一化过程，除最大值
for i in range(num_train):
        dataX[i, 0] = dataX[i, 0]/ 198.8755
        dataX[i, 1] = dataX[i, 1] / 183.4
        dataX[i, 2] = dataX[i, 2] / 183.5725
        dataX[i, 3] = dataX[i, 3] / 184.4340

window=80

train_dataX=np.zeros((num_train-window-1,window,10))  #num_train-30
old_dataX=np.zeros((int((num_train-1)/window),window,10))
# print(train_dataX.shape)
# print(old_dataX.shape)
j=1
for i in range(int(num_train-window-1)):  #num_train-30
    train_dataX[i,:,0:4]=dataX[j:j+window,0:4]
    for k in range(window):
        train_dataX[i, k, 4:10] = dataX[j-1, 4:10]  #将初始状态扩充到训练样本中
    j=j+1

j=1
for i in range(int((num_train-1)/window)):  # num_train-30
    old_dataX[i, :, 0:4] = dataX[j:j+window, 0:4]
    # print(old_dataX[i, :, 0:4].shape)
    # print(dataX[j:j + window, 0:4].shape)
    for k in range(window):
        old_dataX[i, k, 4:10] = dataX[j-1, 4:10]  # 将初始状态扩充到训练样本中
    j = j + window
    # print('ooo')
#print(dataX[4000,:])
file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\50-hour-test.xlsx'   # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# print(raw_data)
data=np.array(raw_data)
data_size=len(data)
t_dataX=data[0:26000,:]
num_train=26000
#数据归一化过程，除最大值
for i in range(num_train):
        t_dataX[i, 0] = t_dataX[i, 0]/198.8755
        t_dataX[i, 1] = t_dataX[i, 1] / 183.4
        t_dataX[i, 2] = t_dataX[i, 2] / 183.5725
        t_dataX[i, 3] = t_dataX[i, 3] / 184.4340

test_dataX=np.zeros((int((num_train-1)/window),window,10))  #num_train-30
# print(train_dataX[0,:,:].shape)
j=1
for i in range(int((num_train-1)/window)):  #num_train-30
    test_dataX[i,:,0:4]=t_dataX[j:j+window,0:4]    #按时间窗划分数据集
    for k in range(window):
        train_dataX[i, k, 4:10] = t_dataX[j-1, 4:10]  # 将初始状态扩充到训练样本中
    j=j+window
#-----------------------------------------------------------------------------label______________________________________
file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\train.xlsx'   # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# print(raw_data)
data=np.array(raw_data)
data_size=len(data)
dataY=data[0:60000,:]  #控制输入
num_train=60000
#设置滑动时间窗口，将数组按时间窗转换成tensor
train_dataY=np.zeros((num_train-window-1,window,6))  #num_train-30
old_dataY=np.zeros((int((num_train-1)/window),window,6))
# print(train_dataX[0,:,:].shape)
j=1
for i in range(int(num_train-window-1)):  #num_train-30
    train_dataY[i,:,:]=dataY[j:j+window,4:10]    #按时间窗划分数据集
    j=j+1
j=1
for i in range(int((num_train-1)/window)):  #num_train-30
    old_dataY[i,:,:]=dataY[j:j+window,4:10]    #按时间窗划分数据集
    j=j+window
#print(dataY[4000,:])
file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\50-hour-test.xlsx'  # r对路径进行转义，windows需要
raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# print(raw_data)
data=np.array(raw_data)
data_size=len(data)
t_dataY=data[0:26000,:]  #控制输入
num_train=26000
#设置滑动时间窗口，将数组按时间窗转换成tensor

test_dataY=np.zeros((int((num_train-1)/window),window,6))  #num_train-30
# print(int(num_train/window))
j=1
for i in range(int((num_train-1)/window)):  #num_train-30
    test_dataY[i,:,:]=t_dataY[j:j+window,4:10]    #按时间窗划分数据集
    j=j+window


#------L2--------------------------------------------------------------------


# file_path = r'D:\wky-python\quator-dataset\train.xlsx'   # r对路径进行转义，windows需要
# raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# # print(raw_data)
# data=np.array(raw_data)
# data_size=len(data)
# dataX=data[0:30000,:]  #控制输入
# num_train=30000
# #设置滑动时间窗口，将数组按时间窗转换成tensor
#
# # te_dataY=t.from_numpy(dataYv)
# #数据归一化过程，除最大值
# for i in range(30000):
#         dataX[i, 0] = dataX[i, 0]/198.8755
#         dataX[i, 1] = dataX[i, 1] / 183.4
#         dataX[i, 2] = dataX[i, 2] / 183.5725
#         dataX[i, 3] = dataX[i, 3] / 184.4340
# window=40
# train_dataX=np.zeros((num_train-window,window,4))  #num_train-30
# # print(train_dataX[0,:,:].shape)
# j=0
# for i in range(num_train-window):  #num_train-30
#     train_dataX[i,:,:]=dataX[j:j+window,0:4]    #按时间窗划分数据集
#     j=j+1
#     # print('ooo')
# #print(dataX[4000,:])
# file_path = r'D:\wky-python\quator-dataset\50-hour-test.xlsx'   # r对路径进行转义，windows需要
# raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# # print(raw_data)
# data=np.array(raw_data)
# data_size=len(data)
# t_dataX=data[10000:20000,:]
# num_train=10000
# #数据归一化过程，除最大值
# for i in range(10000):
#         t_dataX[i, 0] = t_dataX[i, 0]/198.8755
#         t_dataX[i, 1] = t_dataX[i, 1] / 183.4
#         t_dataX[i, 2] = t_dataX[i, 2] / 183.5725
#         t_dataX[i, 3] = t_dataX[i, 3] / 184.4340
#
# test_dataX=np.zeros((num_train-window,window,4))  #num_train-30
# # print(train_dataX[0,:,:].shape)
# j=0
# for i in range(int(num_train/window)):  #num_train-30
#     test_dataX[i,:,:]=t_dataX[j:j+window,0:4]    #按时间窗划分数据集
#     j=j+window
# #-----------------------------------------------------------------------------label______________________________________
# file_path = r'D:\wky-python\quator-dataset\train.xlsx'   # r对路径进行转义，windows需要
# raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# # print(raw_data)
# data=np.array(raw_data)
# data_size=len(data)
# dataY=data[0:30000,:]  #控制输入
# num_train=30000
# #设置滑动时间窗口，将数组按时间窗转换成tensor
# train_dataY=np.zeros((num_train-window,window,6))  #num_train-30
# # print(train_dataX[0,:,:].shape)
# j=0
# for i in range(num_train-window):  #num_train-30
#     train_dataY[i,:,:]=dataY[j:j+window,4:10]    #按时间窗划分数据集
#     j=j+1
# #print(dataY[4000,:])
# file_path = r'D:\wky-python\quator-dataset\50-hour-test.xlsx'  # r对路径进行转义，windows需要
# raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
# # print(raw_data)
# data=np.array(raw_data)
# data_size=len(data)
# t_dataY=data[10000:20000,:]  #控制输入
# num_train=10000
# #设置滑动时间窗口，将数组按时间窗转换成tensor
#
# test_dataY=np.zeros((num_train-window,window,6))  #num_train-30
# # print(train_dataX[0,:,:].shape)
# j=0
# for i in range(int(num_train/window)):  #num_train-30
#     test_dataY[i,:,:]=t_dataY[j:j+window,4:10]    #按时间窗划分数据集
#     j=j+window
#--------------------------------------------------------------------------------------------------------dataloader
tr_dataX=torch.from_numpy(train_dataX)
old_dataX=torch.from_numpy(old_dataX)
te_dataX=torch.from_numpy(test_dataX)#将处理好的数据集转换成张量
tr_dataY=torch.from_numpy(train_dataY)
old_dataY=torch.from_numpy(old_dataY)
te_dataY=torch.from_numpy(test_dataY)#
tr_dataX=tr_dataX.float()
old_dataX=old_dataX.float()
te_dataX=te_dataX.float()
tr_dataY=tr_dataY.float()
old_dataY=old_dataY.float()
te_dataY=te_dataY.float()

class MiningDataset():
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.data_size = int(self.data.shape[0])

    def __getitem__(self, i):
        """
        :param i:
        :return:  (time_step. feature_size)
        """
        con_data=self.data[i,:,:]
        label = self.label[i,:,:]
        # label1 = self.data[i, -1, :]  # 每个时间窗的最后一行数据
        return  con_data, label

    def __len__(self):
        return self.data_size

tr_data = MiningDataset(tr_dataX, tr_dataY)
te_data = MiningDataset(te_dataX, te_dataY)
old_data = MiningDataset(old_dataX, old_dataY)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=tr_data,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=te_data,
                                           batch_size=1,
                                          shuffle=False, drop_last=False, pin_memory=True)

previous_loader=torch.utils.data.DataLoader(dataset=old_data,
                                           batch_size=1,
                                          shuffle=False, drop_last=False, pin_memory=True)

src_len = 200 # enc_input max sequence length
tgt_len = 200  # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 64 # Embedding Size
d_ff = 256 # FeedForward dimension
d_k = d_v = 128 # dimension of K(=Q), V
n_layers =  2              # number of Encoder of Decoder Layer
n_heads = 6 # num4er of heads in Multi-Head Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(200.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
     #   scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

      #  attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
      #                                            1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

class ini_embeding(nn.Module):
    def __init__(self):
        super(ini_embeding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16, bias=False),
            nn.Tanh(),
            nn.Linear(16, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32,d_model, bias=False),
        )
    def forward(self,inputs):
        output=self.fc(inputs)
        return output

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)  #初始化状态不用位置编码
        self.ini_emb=ini_embeding()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.ini_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DNN(nn.Module):       #负责将提取的特征回归为系统输出
    def __init__(self):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 6, bias=False),
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output # [batch_size, seq_len, d_model]

class DT_transformer(nn.Module):
    def __init__(self):
        super(DT_transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.dnn = DNN().cuda()

    def forward(self, ini_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        ini_outputs, _ = self.encoder(ini_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        #output=c_outputs
        output = self.dnn( ini_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return output
if __name__=='__main__':
    model = DT_transformer()
    model=model.cuda()
    model_best=DT_transformer()
    model_best=model_best.cuda()
    lr = 6e-4
    num_epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 传入网络参数和学习率
    loss_function = torch.nn.MSELoss(reduction="mean")  # 最小均方误差
    Train_Loss=9999
    Test_Loss = 9999
    history = dict(predict=[], feature=[])
    fisher={}
    for epoch in range(num_epochs):
        Loss = []
        Loss1 = []
        for i, (ini_datas, labels) in enumerate(train_loader):
            ini_datas=ini_datas.cuda()
            labels = labels.cuda()
            # labels1 = labels1.cuda()
            model.train()
            prediction = model(ini_datas)  # 把数据x喂给net，输出预测值
            # loss1 = loss_function(prediction[:,:,0:3], labels[:,:,0:3])
            loss2 = loss_function(prediction, labels)
            # p_loss=6e-5*physics_loss(con_datas,prediction,0.01) # 计算两者的误差，要注意两个参数的顺序，rul误差
            # # p_loss=3e-2*p_cost(prediction,labels)
            # loss+=p_loss
            loss=loss2
            Loss.append(loss.detach().cpu().numpy())
            # loss1 = loss_function(x, labels1)  #状态预测误差
            # Loss1.append(loss1.detach().cpu().numpy())
            loss_sum = (loss)
            #print(loss_sum)
            optimizer.zero_grad()  # 清空上一步的更新参数值
            # loss1.backward()  # 误差反相传播，计算新的更新参数值
            loss_sum.backward()
            optimizer.step()  # 将计算得到的更新值赋给net.parameters()

        if loss_sum < Train_Loss:
            Train_Loss=loss_sum
            print(epoch, 'train_loss: ', loss_sum)
            # print(loss_p)
            model_best= deepcopy(model)  #记录下最好的模型网络

    # test_loss = []
    # Y_pre=[]
    # # p_sum_loss=[]
    # model_flag=model_best
    # for i, (con_datas, labels) in enumerate(test_loader):
    #         con_datas = con_datas.cuda()
    #         labels = labels.cuda()
    #         target = model_best(con_datas)
    #         Y_pre.append(target)
    #         loss_for_train = loss_function(target, labels)
    #         loss_true = loss_function(target, labels)
    #         test_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差

    # test_loss = []
    # Y_pre=[]
    # # p_sum_loss=[]
    # epoch=20
    # # model1=DT_transformer()
    # model_inc=DT_transformer()
    # model_old=model_best
    # for i, (datas, labels) in enumerate(test_loader):
    #         datas=datas.cuda()
    #         labels = labels.cuda()
    #         if i == 0:
    #             model_inc=model_best
    #         target = model_inc(datas)
    #         Y_pre.append(target)
    #         loss_for_train = loss_function(target, labels)
    #         loss_true = loss_function(target, labels)
    #         test_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差
    #         online_loss = 9999
    #         optimizer2 = torch.optim.Adam(model_inc.dnn.parameters(), lr=6e-4)  # 传入网络参数和学习率
    #         for j in range(epoch):   #开始增量学习阶段
    #            loss_reg = 0
    #            for (name, param), (_, param_old) in zip(model_inc.dnn.named_parameters(),
    #                                                      model_best.dnn.named_parameters()):
    #                 loss_reg += torch.sum(fisher[name] * (param_old - param).pow(2)) / 2  # EWC的损失函数的正则化部分
    #                 print(param_old-param)
    #            target = model_inc(datas)
    #            loss_for_train = loss_reg+loss_function(target, labels)
    #            optimizer2.zero_grad()  # 清空上一步的更新参数值
    #            loss_for_train.backward()  # 误差反相传播，计算新的更新参数值
    #            optimizer2.step()  # 将计算得到的更新值赋给net.parameters()
    #
    #            if loss_for_train < online_loss:
    #                online_loss = loss_for_train
    #                model_flag=model_inc  #记录最好的网络
    #         model_inc=model_flag #用于下一个批数据的预测

    test_loss = []
    Y_pre=[]
    # p_sum_loss=[]
    epoch=20
    model_inc=DT_transformer()
    model_flag=deepcopy(model_best)
    triggering_flag=0
    count=0

    #计算模型各参数在历史数据集上的重要程度
    for i,(ini_datas,labels) in enumerate(previous_loader):
        ini_datas = ini_datas.cuda()
        labels = labels.cuda()
        pre_old=model_best(ini_datas)
        loss_train=loss_function(pre_old,labels)
        loss_train.backward(retain_graph=True)  # 误差反传，计算Fisher矩阵
        for n, p in model_best.dnn.named_parameters():  # 遍历网络中的每一个参数
            fisher[n] = 0 * p.data
        for n, p in model_best.dnn.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

        # for n, _ in model_flag.named_parameters():
        #     fisher[n] = fisher[n] /   # 样本总数 一个batch中
        #     fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)  # 计算获得的Fisher信息矩阵，不计算这块的梯度\

    for i, (ini_datas,  labels) in enumerate(test_loader):
            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            if i == 0:
                model_inc=model_best
            target = model_flag(ini_datas)
            Y_pre.append(target)
            loss_for_train = loss_function(target, labels)
            # v1=target
            # v2=labels
            # v1=v1.detach().cpu()
            # v2=v2.detach().cpu()
            # v1 = v1.detach().numpy()
            # v1 = v1.flatten()
            # v2 = v2.detach().numpy()
            # v2 = v2.flatten()
            # cos_similar=1-spatial.distance.cosine(v1, v2)
            # print('cos', cos_similar)
            loss_true = loss_function(target, labels)
            test_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差
            online_loss = 999
            # loss_for_train.backward(retain_graph=True)
            if  loss_for_train > 0.3:
                triggering_flag=1
                count+=1
            if triggering_flag == 1: #触发一次演化
                model_inc = model_flag
                start1 = time.perf_counter()
                for j in range(epoch):  # 开始增量学习阶段
                    loss_reg = 0
                    for (name, param), (_, param_old) in zip(model_flag.dnn.named_parameters(),
                                                             model_best.dnn.named_parameters()):
                        loss_reg += torch.sum(fisher[name] * (param_old - param).pow(2)) / 2  # EWC的损失函数的正则化部分
                    optimizer2 = torch.optim.Adam(model_inc.dnn.parameters(), lr=6e-4)  # 传入网络参数和学习率
                    optimizer2.zero_grad()  # 清空上一步的更新参数值
                    # print(loss_reg)
                    target = model_inc(ini_datas)
                    loss_for_train = loss_function(target, labels)
                    loss_for_train = 10 * loss_reg + loss_for_train
                    loss_for_train.backward()  # 误差反相传播，计算新的更新参数值
                    optimizer2.step()  # 将计算得到的更新值赋给net.parameters()
                    # target = model_inc(ini_datas)
                    # loss_for_train = loss_function(target, labels)
                    if loss_for_train < online_loss:
                        online_loss = loss_for_train
                        model_flag = deepcopy(model_inc)  # 记录最好的网络
                end1 = time.perf_counter()
                triggering_flag = 0
                # model_inc=deepcopy(model_flag)  #用于下一个批数据的预测

    # Y_pre=[]
    # p_sum_loss=[]
    # model_flag=model_best

    y_pre=torch.cat(Y_pre,dim=0)
    y_pre=y_pre.detach().cpu()
    # y1=y_pre[-1,20,:]
    #y_real=te_dataY[-1,20,:]
    # x=np.arange(0,y1.shape[0])
    # x=torch.from_numpy(x)
    # x=x.reshape(y_real.shape)
    #------------------------------------------------------------------------------------------------------画图，可视化
    pre_dataX=np.zeros((200,window,6))
    real_dataX=np.zeros((200,window,6))
    pre_dataX=torch.from_numpy(pre_dataX)
    real_dataX=torch.from_numpy(real_dataX)

    j=0
    for i in range(200):
        pre_dataX[i,:,:]=y_pre[i,:,:]
    j=0
    for i in range(200):
        real_dataX[i, :, :] = te_dataY[i, :, :]
        # j = j + window


    a=pre_dataX[:,:,0]
    a=a.detach().numpy()
    a=a.flatten()
    b=real_dataX[:,:,0]
    b=b.detach().numpy()
    b=b.flatten()

    a1=pre_dataX[:,:,1]
    a1=a1.detach().numpy()
    a1=a1.flatten()
    b1=real_dataX[:,:,1]
    b1=b1.detach().numpy()
    b1=b1.flatten()

    a2=pre_dataX[:,:,2]
    a2=a2.detach().numpy()
    a2=a2.flatten()
    b2=real_dataX[:,:,2]
    b2=b2.detach().numpy()
    b2=b2.flatten()

    a3=pre_dataX[:,:,3]
    a3=a3.detach().numpy()
    a3=a3.flatten()
    b3=real_dataX[:,:,3]
    b3=b3.detach().numpy()
    b3=b3.flatten()

    a4=pre_dataX[:,:,4]
    a4=a4.detach().numpy()
    a4=a4.flatten()
    b4=real_dataX[:,:,4]
    b4=b4.detach().numpy()
    b4=b4.flatten()

    a5=pre_dataX[:,:,5]
    a5=a5.detach().numpy()
    a5=a5.flatten()
    b5=real_dataX[:,:,5]
    b5=b5.detach().numpy()
    b5=b5.flatten()

    pc = pearsonr(a,b)
    print(pc)
    # print('time',end1-start1)
    print(np.sqrt(np.mean(test_loss))) #RMSE

    old_loss = []
    for i, (con_datas, labels) in enumerate(train_loader):
            con_datas = con_datas.cuda()
            labels = labels.cuda()
            target = model_flag(con_datas)
            # Y_pre.append(target)
            loss_for_train = loss_function(target, labels)
            loss_true = loss_function(target, labels)
            old_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差

    print('oldloss',np.sqrt(np.mean(old_loss))) #RMSE
    # print('oldloss',np.mean(old_loss)) #RMSE

    print(count)
    param = count_param(model_inc)
    print('The model size is',param)
    #
    plt.plot(a,color='r',label='prediction output of xi')
    plt.plot(b,color='b',label='real output of xi')
    plt.legend(prop = {'size':12})
    plt.xlabel('sample')
    plt.ylabel('mm')
    plt.show()


    plt.plot(a1,color='r',label='prediction output of yi')
    plt.plot(b1,color='b',label='real output of yi')
    plt.legend(prop = {'size':12})
    plt.xlabel('sample')
    plt.ylabel('mm')
    plt.show()

    plt.plot(a2,color='r',label='prediction output of zi')
    plt.plot(b2,color='b',label='real output of zi')
    plt.legend(prop = {'size':12})
    plt.xlabel('sample')
    plt.ylabel('mm')
    plt.show()

    plt.plot(a3,color='r',label='prediction output of Roll')
    plt.plot(b3,color='b',label='real output of Roll')
    plt.xlabel('sample')
    plt.ylabel('deg')
    plt.legend(prop = {'size':12})
    plt.show()

    plt.plot(a4,color='r',label='prediction output of Pitch')
    plt.plot(b4,color='b',label='real output of Pitch')
    plt.legend(prop = {'size':12})
    plt.xlabel('sample')
    plt.ylabel('deg')
    plt.show()

    plt.plot(a5,color='r',label='prediction output of Yaw')
    plt.plot(b5,color='b',label='real output of Yaw')
    plt.legend(prop = {'size':12})
    plt.xlabel('sample')
    plt.ylabel('deg')
    plt.show()
    # print(test_loss)