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
from tqdm import tqdm
from utils import *
from DT_Former import DT_transformer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from test_model import *
def visualize_error(first_three_errors,last_three_errors,save_dir):
    first_three_errors = np.array(first_three_errors)
    last_three_errors = np.array(last_three_errors)
    print(count,len(first_three_errors),first_three_errors.shape)
    # 绘制误差变化图
    plt.figure(figsize=(12, 5))

    # 前三个自由度的误差变化
    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(range(len(first_three_errors)), first_three_errors[:, i], 
                label=f'freedom {i+1}', 
                linestyle='-', 
                marker='o', 
                markersize=2)
    plt.xlabel('trigger')
    plt.ylabel('error')
    plt.title('first three degrees of freedom error')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(range(len(last_three_errors)), last_three_errors[:, i], 
                label=f'freedom {i+4}', 
                linestyle='-', 
                marker='o', 
                markersize=2)
    plt.xlabel('trigger')
    plt.ylabel('error')
    plt.title('last three degrees of freedom error')
    plt.grid(True)
    plt.title('last three degrees of freedom error')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.close()

def visualize_loss(loss_list, save_dir):
    """
    可视化loss的变化曲线
    Args:
        loss_list: 损失值列表
        save_dir: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_list)), loss_list, 'b-', label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Evolution During Online Learning')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.close()

if __name__=="__main__":
    window = 80
    batch_size =2048
    online_folder_idx=0
    device = set_device()
    
    train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
    test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
    online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')

    
    # test_slide_dataset=prepare_merged_data(folders=test_folders.tolist(),window=window,return_split=False,return_norm_data=True)
    #  test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
    #                            batch_size=batch_size,
    #                            shuffle=False,  
    #                            drop_last=False,
    #                            pin_memory=True)


    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    model = model.cuda()


    save_dir = r'/home/pika/koopman-data/data/LSTM_decoder_0503_3'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'best_model_0503.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model_best=deepcopy(model)
    
    loss_function = torch.nn.MSELoss(reduction="mean")

    fisher_path = os.path.join(save_dir, 'fisher_dict.npy')
    if not os.path.exists(fisher_path):
        train_slide_dataset=prepare_merged_data(folders=train_folders.tolist(),window=window,return_split=False,return_norm_data=True)
        train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
                           batch_size=batch_size,
                           shuffle=True, 
                           drop_last=True,  
                           pin_memory=True)  
        fisher={}
        for i,(ini_datas,labels) in enumerate(train_slide_loader):
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
        np.save(fisher_path, fisher)
        print(f"\nFisher字典已保存至: {fisher_path}")
    else:
        fisher = np.load(fisher_path, allow_pickle=True).item()
    # 保存Fisher字典
    train_slide_dataset=prepare_merged_data(folders=[31],window=window,return_split=False,return_norm_data=True)
    train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
                       batch_size=batch_size,
                       shuffle=False, 
                       drop_last=True,  
                       pin_memory=True)  
    fisher={}
    for iter in range(10):
        for i,(ini_datas,labels) in enumerate(train_slide_loader):
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
        np.save(fisher_path, fisher)
        print(f"\nFisher字典已保存至: {fisher_path}")
    
        
    
    base_path = '/home/pika/koopman-data/data/flights'    
    model_inc=MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)

    original_iter_loss=0
    online_folder=os.path.join(save_dir,'online_folder')
    if not os.path.exists(online_folder):
        os.makedirs(online_folder)
    test_folder=os.path.join(save_dir,'test_folder')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for online_folder_idx in test_folders:
        online_slide_dataset=prepare_merged_data(folders=[online_folder_idx],window=window,return_split=False,return_norm_data=True)
        online_slide_loader = torch.utils.data.DataLoader(dataset=online_slide_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,  
                                   drop_last=False,
                                   pin_memory=True)
        norm_data = np.load(os.path.join(base_path, str(online_folder_idx), 'folder_stats.npy'), allow_pickle=True).item()

        first_three_error,last_three_error,best_loss=test_model(model_best, online_slide_loader,device,save_dir=None,norm_data=norm_data)

        epochs=10
        model_flag=deepcopy(model_best)
        loss_list=[]
        first_three_errors=[]
        last_three_errors=[]

        count=0
        triggering_flag=0
        model_inc=model_flag
        optimizer2 = torch.optim.Adam(model_inc.parameters(), lr=1e-4, weight_decay=1e-4)

        for iter in range(epochs):
            for i, (ini_datas,  labels) in enumerate(online_slide_loader):
                ini_datas = ini_datas.cuda()
                labels = labels.cuda()
                labels_old=labels.clone()
                target = model_flag(ini_datas)

                target[:,:,0:3] = denormalize_data(target[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
                target[:,:,3:6] = denormalize_data(target[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])
                labels[:,:,0:3] = denormalize_data(labels[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
                labels[:,:,3:6] = denormalize_data(labels[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])

                loss_for_train = loss_function(target, labels)
                loss_list.append(loss_for_train.item())

                if  loss_for_train > 0.1:
                    triggering_flag=1
                    count+=1
                    print(f"触发在线学习，当前触发次数：{count}")  # 添加触发信号输出

                if triggering_flag == 1:
                    model_inc.train()
                    loss_reg = 0
                    for (name, param), (_, param_old) in zip(model_inc.ffn.named_parameters(),
                                                             model_best.ffn.named_parameters()):
                        loss_reg += torch.sum(fisher[name] * (param_old - param).pow(2)) / 2

                    optimizer2.zero_grad()
                    target = model_inc(ini_datas)
                    loss_error = loss_function(target, labels_old)
                    # loss_for_train = loss_error + 1000 * loss_reg  # 降低正则化项的权重
                    loss_for_train = loss_error
                    print(f"loss_reg:{loss_reg.item()},loss_error:{loss_error.item()},loss_for_train:{loss_for_train.item()}")
                    loss_for_train.backward()
                    optimizer2.step()
                    optimizer2.zero_grad()

                    triggering_flag = 0
                    first_three_error, last_three_error, avg_loss = test_model(model_inc, online_slide_loader,device,save_dir=None,norm_data=norm_data)
                    loss_list.append(avg_loss)
                    first_three_errors.append(first_three_error)
                    last_three_errors.append(last_three_error)
                    print(f"avg loss:{avg_loss}")
                    print(f"前三个自由度的平均误差（原始尺度）: {np.mean(first_three_error):.4f}")
                    print(f"后三个自由度的平均误差（原始尺度）: {np.mean(last_three_error):.4f}")
                    if avg_loss < best_loss:
                        best_loss=avg_loss
                        model_flag=deepcopy(model_inc)
                        torch.save(model_flag.state_dict(), os.path.join(test_folder, 'best_model_online_best_for_'+str(online_folder_idx)+'.pth'))
                        png_save_path=os.path.join(test_folder,'test_'+str(online_folder_idx)+'_after_online_learning.png')
                        test_model(model_inc,online_slide_loader,device,save_dir=png_save_path,norm_data=norm_data)

        if count>0:
            visualize_error(first_three_errors,last_three_errors,os.path.join(test_folder, 'error_evolution_for_'+str(online_folder_idx)+'.png'))
            visualize_loss(loss_list, os.path.join(test_folder, 'loss_evolution_for_'+str(online_folder_idx)+'.png'))
        
    # for online_folder_idx in online_folders:
    #     online_folder_idx=31
    #     online_slide_dataset=prepare_merged_data(folders=[online_folder_idx],window=window,return_split=False,return_norm_data=True)
    #     online_slide_loader = torch.utils.data.DataLoader(dataset=online_slide_dataset,
    #                            batch_size=batch_size,
    #                            shuffle=False,  
    #                            drop_last=False,
    #                            pin_memory=True)
    #     folder=online_folder_idx
    #     norm_data = np.load(os.path.join(base_path, str(folder), 'folder_stats.npy'), allow_pickle=True).item()
        
    #     first_three_errors = []
    #     last_three_errors = []
    #     count=0
    #     triggering_flag = 0
            
    #     epochs=30
    #     model_inc=deepcopy(model_best)
    #     loss_list=[]
    #     optimizer2 = torch.optim.Adam(model_inc.ffn.parameters(), lr=1e-4, weight_decay=1e-4)
    #     for iter in range(epochs):
    #         model_flag=model_inc
    #         for i, (ini_datas,  labels) in enumerate(online_slide_loader):
    #             ini_datas = ini_datas.cuda()
    #             labels = labels.cuda()
                
    #             target = model_flag(ini_datas)

    #             target[:,:,0:3] = denormalize_data(target[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
    #             target[:,:,3:6] = denormalize_data(target[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])
    #             labels[:,:,0:3] = denormalize_data(labels[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
    #             labels[:,:,3:6] = denormalize_data(labels[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])


    #             loss_for_train = loss_function(target, labels)
                
    #             loss_true = loss_function(target, labels)
    #             test_loss = []
    #             test_loss.append(loss_true.detach().cpu().numpy())   #记录测试集总误差
    #             online_loss = 999
    #             if  loss_for_train > 0.1:
    #                 triggering_flag=1
    #                 original_iter_loss=loss_for_train.item()
    #                 count+=1
    #                 print(f"触发在线学习，当前触发次数：{count}")  # 添加触发信号输出
                
    #             epoch=1
    #             if triggering_flag == 1: #触发一次演化
                    
    #                 model_inc.train()
    #                 start1 = time.perf_counter()
    #                 pbar = tqdm(range(epoch), desc=f'在线学习 批次 {count}')  # 添加进度条
    #                 for j in pbar:  # 开始增量学习阶段
    #                     loss_reg = 0
    #                     for (name, param), (_, param_old) in zip(model_flag.ffn.named_parameters(),
    #                                                              model_best.ffn.named_parameters()):
    #                         loss_reg += torch.sum(fisher[name] * (param_old - param).pow(2)) / 2
                        
    #                     optimizer2.zero_grad()
    #                     target = model_inc(ini_datas)
    #                     target[:,:,0:3] = denormalize_data(target[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
    #                     target[:,:,3:6] = denormalize_data(target[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])
    #                     loss_error = loss_function(target, labels)
    #                     loss_for_train =  10*loss_reg + loss_error  # 计算总的loss
    #                     loss_list.append(loss_for_train.item())
    #                     # loss_for_train = loss_error
    #                     # print("loss_error:",loss_error.item(),"loss_original:",original_iter_loss,"loss_train:",loss_for_train.item(),"loss_reg:",loss_reg.item(),"loss_error:",loss_error.item())
    #                     loss_for_train.backward()
    #                     optimizer2.step()
    #                     optimizer2.zero_grad()

    #                     # 更新进度条信息
    #                     pbar.set_postfix({
    #                         'loss_ERROR': f'{loss_error.item():.4f}',
    #                         'loss_REG': f'{loss_reg.item():.4f}',
    #                         'loss_original': f'{original_iter_loss:.4f}',
    #                         'loss_train': f'{loss_for_train.item():.4f}'
    #                     })

    #                     if loss_for_train < online_loss:
    #                         online_loss = loss_for_train
    #                         model_flag = deepcopy(model_inc)
    #                 end1 = time.perf_counter()
    #                 triggering_flag = 0

    #                 first_three_error, last_three_error, avg_loss = test_model(model_flag, online_slide_loader,device,save_dir=None,norm_data=norm_data)

    #                 # 将新的误差值添加到数组中
    #                 first_three_errors.append(first_three_error)
    #                 last_three_errors.append(last_three_error)
    #                 print(f"avg loss:{avg_loss}")
    #                 print(f"前三个自由度的平均误差（原始尺度）: {np.mean(first_three_error):.4f}")
    #                 print(f"后三个自由度的平均误差（原始尺度）: {np.mean(last_three_error):.4f}")
                    
    #             # break
    #     torch.save(model_flag.state_dict(), os.path.join(save_dir, 'best_model_online.pth'))
    #     print("完成")
        
    #     visualize_error(first_three_errors,last_three_errors,os.path.join(save_dir, 'error_evolution_'+str(online_folder_idx)+'.png'))
    #     visualize_loss(loss_list, os.path.join(save_dir, 'loss_evolution_'+str(online_folder_idx)+'.png'))
        
    #     break
    