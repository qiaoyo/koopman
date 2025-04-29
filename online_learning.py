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
from utils import *
from DT_Former import DT_transformer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from test_model import *
if __name__=="__main__":
    window = 80
    batch_size = 512
    device = set_device()
    train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset=prepare_full_data(window=window)

    train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
                           batch_size=batch_size,
                           shuffle=True, 
                           drop_last=True,  
                           pin_memory=True)  
    train_split_loader = torch.utils.data.DataLoader(dataset=train_split_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True)
    test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
                               batch_size=batch_size,
                               shuffle=False,  
                               drop_last=False,
                               pin_memory=True)
    test_split_loader = torch.utils.data.DataLoader(dataset=test_split_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

    from sample import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_size=10, output_dim=6)
    model = model.cuda()


    save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\test_decode_ffn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model_best=deepcopy(model)
    
    loss_function = torch.nn.MSELoss(reduction="mean")
    fisher={}
    for i,(ini_datas,labels) in enumerate(train_split_loader):
        ini_datas = ini_datas.cuda()
        labels = labels.cuda()
        pre_old=model_best(ini_datas)
        loss_train=loss_function(pre_old,labels)
        loss_train.backward(retain_graph=True)  # 误差反传，计算Fisher矩阵
        for n, p in model_best.ffn.named_parameters():  # 遍历网络中的每一个参数
            fisher[n] = 0 * p.data
        for n, p in model_best.ffn.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    model_flag=deepcopy(model_best)
    norm_params_path = r'C:\Users\Administrator\Desktop\koopman-data\data\normalization_params.npy'  
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    
    # 初始化误差数组
    first_three_errors = []
    last_three_errors = []
    test_loss = []
    triggering_flag = 0
    count=0
    epoch=30
    model_inc=MultiScaleTimeSeriesModel(input_size=10, output_dim=6)
    for i, (ini_datas,  labels) in enumerate(test_slide_loader):
            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            if i == 0:
                model_inc=model_best
            target = model_flag(ini_datas)

            loss_for_train = loss_function(target, labels)
            loss_true = loss_function(target, labels)
            test_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差
            online_loss = 999
            if  loss_for_train > 0.01:
                triggering_flag=1
                count+=1
            if triggering_flag == 1: #触发一次演化
                model_inc = model_flag
                model_inc.train()
                start1 = time.perf_counter()
                for j in range(epoch):  # 开始增量学习阶段
                    loss_reg = 0
                    for (name, param), (_, param_old) in zip(model_flag.ffn.named_parameters(),
                                                             model_best.ffn.named_parameters()):
                        loss_reg += torch.sum(fisher[name] * (param_old - param).pow(2)) / 2  # EWC的损失函数的正则化部分
                    optimizer2 = torch.optim.Adam(model_inc.ffn.parameters(), lr=6e-4)  # 传入网络参数和学习率
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
               
                first_three_error, last_three_error = test_model(model_flag, test_slide_loader,device,norm_params,save_dir)
                
                # 将新的误差值添加到数组中
                first_three_errors.append(first_three_error)
                last_three_errors.append(last_three_error)
                
                # print(f"前三个自由度的平均误差（原始尺度）: {first_three_error:.4f}")
                # print(f"后三个自由度的平均误差（原始尺度）: {last_three_error:.4f}")
                # model_inc=deepcopy(model_flag)  #用于下一个批数据的预测
    torch.save(model_flag.state_dict(), os.path.join(save_dir, 'best_model_online.pth'))
    print("完成")
    first_three_errors = np.array(first_three_errors)
    last_three_errors = np.array(last_three_errors)
    print(count,len(first_three_errors),first_three_errors.shape)
    # 绘制误差变化图
    plt.figure(figsize=(12, 5))
    
    # 前三个自由度的误差变化
    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(range(len(first_three_errors)), first_three_errors[:, i], 
                label=f'自由度 {i+1}', 
                linestyle='-', 
                marker='o', 
                markersize=2)
    plt.xlabel('更新次数')
    plt.ylabel('误差值')
    plt.title('前三个自由度误差变化')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(range(len(last_three_errors)), last_three_errors[:, i], 
                label=f'自由度 {i+4}', 
                linestyle='-', 
                marker='o', 
                markersize=2)
    plt.xlabel('更新次数')
    plt.ylabel('误差值')
    plt.title('后三个自由度误差变化')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_evolution.png'), dpi=300)
    plt.close()

    