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
import random
from MiningDataset import MiningDataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
    # set device to cpu or cuda
        device = torch.device('cpu')
        print("Device set to : cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def time_series_augmentation(data, noise_level=0.01, shift_range=2):
    """
    对时序数据进行增强
    params:
        data: 原始数据 numpy array
        noise_level: 高斯噪声水平
        shift_range: 时间平移范围
    """
    augmented_data = np.copy(data)
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level, augmented_data.shape)
    augmented_data += noise
    
    # 随机时间平移
    shift = np.random.randint(-shift_range, shift_range + 1)
    if shift > 0:
        augmented_data = np.roll(augmented_data, shift, axis=0)
        augmented_data[:shift] = augmented_data[shift]
    elif shift < 0:
        augmented_data = np.roll(augmented_data, shift, axis=0)
        augmented_data[shift:] = augmented_data[shift-1]
    
    return augmented_data

def analyze_flight_data():
    """
    遍历flights文件夹下所有子文件夹中的特定npy文件，统计每个维度的最大最小值并保存
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    # 初始化字典存储结果
    stats = {}
    first_run = True
    
    # 遍历0-53文件夹
    for folder_idx in range(54):
        folder_path = os.path.join(base_path, str(folder_idx))
        if os.path.exists(folder_path):
            # 遍历目标文件
            for file_name in target_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    key = file_name.split('.')[0]
                    
                    # 初始化该文件的统计信息
                    if first_run:
                        stats[key] = {
                            'max': np.full(data.shape[1], float('-inf')),
                            'min': np.full(data.shape[1], float('inf'))
                        }
                    
                    # 更新每个维度的最大最小值
                    for dim in range(data.shape[1]):
                        current_max = np.max(data[:, dim])
                        current_min = np.min(data[:, dim])
                        stats[key]['max'][dim] = max(stats[key]['max'][dim], current_max)
                        stats[key]['min'][dim] = min(stats[key]['min'][dim], current_min)
            
            if first_run:
                first_run = False
    
    # 打印结果
    print("\n数据统计结果:")
    for key in stats:
        print(f"\n{key}:")
        for dim in range(len(stats[key]['max'])):
            print(f"维度 {dim}:")
            print(f"  最大值: {stats[key]['max'][dim]:.4f}")
            print(f"  最小值: {stats[key]['min'][dim]:.4f}")
    
    # 保存统计结果
    save_path = os.path.join(base_path, 'flight_stats.npy')
    np.save(save_path, stats)
    print(f"\n统计结果已保存至: {save_path}")
    
    return stats

def load_flight_stats():
    """
    读取已保存的飞行数据统计结果
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    stats_path = os.path.join(base_path, 'flight_stats.npy')
    
    if not os.path.exists(stats_path):
        print("未找到统计文件，请先运行analyze_flight_data()函数")
        return None
    
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # 打印加载的结果
    print("\n加载的数据统计结果:")
    for key in stats:
        print(f"\n{key}:")
        print(f"最大值: {stats[key]['max']}")
        print(f"最小值: {stats[key]['min']}")
    
    return stats

def normalize_first_four(data, max_values=None, min_values=None):
    """
    对形状为(N, 10)的数据的前4列进行归一化
    params:
        data: numpy array, 形状为(N, 10)
        max_values: numpy array, 形状为(4,), 前4列的最大值，如果为None则使用数据本身的最大值
        min_values: numpy array, 形状为(4,), 前4列的最小值，如果为None则使用数据本身的最小值
    return:
        normalized_data: 归一化后的数据
    """
    normalized_data = data.copy()
    
    # 只对前4列进行归一化
    for i in range(4):
        # 如果没有提供max_values和min_values，则使用数据本身的最大最小值
        col_max = max_values[i] if max_values is not None else np.max(data[:, i])
        col_min = min_values[i] if min_values is not None else np.min(data[:, i])
        
        # 避免除以零
        if col_max == col_min:
            normalized_data[:, i] = 0
        else:
            normalized_data[:, i] = (data[:, i] - col_min) / (col_max - col_min)
    
    return normalized_data

def prepare_full_data(window=80):
    '''
    train_slide_data, test_slide_data, train_split_data, test_split_data
    train_slide_label, test_slide_label, train_split_label, test_split_label
    '''
    file_path =  r'C:\Users\Administrator\Desktop\koopman-data\data\train.xlsx'   # r对路径进行转义，windows需要
    raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    # print(raw_data)
    raw_data=np.array(raw_data)
    raw_data=raw_data[0:60000,:]
    data_size = len(raw_data)
    num_train=data_size

    train_slide_data=np.zeros((num_train-window-1,window,10))  #num_train-30
    train_split_data=np.zeros((int((num_train-1)/window),window,10))
    train_slide_label=np.zeros((num_train-window-1,window,6))  #num_train-30
    train_split_label=np.zeros((int((num_train-1)/window),window,6))
    
        # ... existing code ...
    
    # 计算训练数据前4列的最大最小值
    loaded_stats = load_flight_stats()
    motor_cmd_max= loaded_stats['Motors_CMD']['max']
    motor_cmd_min= loaded_stats['Motors_CMD']['min']
    pos_max = loaded_stats['Pos']['max']
    pos_min = loaded_stats['Pos']['min']
    euler_max = loaded_stats['Euler']['max']
    euler_min = loaded_stats['Euler']['min']
    
    # 使用min-max归一化
    raw_data[:,0:4] = normalize_first_four(raw_data[:,0:4], motor_cmd_max, motor_cmd_min)
    # raw_data[:,4:7] = normalize_first_four(raw_data[:,4:7], pos_max, pos_min)
    # raw_data[:,7:10] = normalize_first_four(raw_data[:,7:10], euler_max, euler_min)    
    j=1
    for i in range(int(num_train-window-1)):  
        train_slide_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+1
    j=1
    for i in range(int((num_train-1)/window)):  
        train_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window
   
    j=1
    for i in range(int(num_train-window-1)):  
        train_slide_data[i,:,0:4]=raw_data[j:j+window,0:4]
        for k in range(window):
            train_slide_data[i, k, 4:10] = raw_data[j-1, 4:10]  #将初始状态扩充到训练样本中
        j=j+1

    j=1
    for i in range(int((num_train-1)/window)):  
        train_split_data[i, :, 0:4] = raw_data[j:j+window, 0:4]
        for k in range(window):
            train_split_data[i, k, 4:10] = raw_data[j-1, 4:10]  # 将初始状态扩充到训练样本中
        j = j + window
    train_slide_data=time_series_augmentation(train_slide_data)

    file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\50-hour-test.xlsx'   # r对路径进行转义，windows需要
    raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    raw_data=np.array(raw_data)
    raw_data=raw_data[0:26000,:]
    data_size=len(raw_data)
    num_test=data_size

    test_split_label=np.zeros((int((num_test-1)/window),window,6))  #num_test-30
    test_slide_label=np.zeros((num_test-window-1,window,6))
    test_split_data=np.zeros((int((num_test-1)/window),window,10))  #num_test-30
    test_slide_data=np.zeros((num_test-window-1,window,10))

    raw_data[:,0:4] = normalize_first_four(raw_data[:,0:4], motor_cmd_max, motor_cmd_min)
    # raw_data[:,4:7] = normalize_first_four(raw_data[:,4:7], pos_max, pos_min)
    # raw_data[:,7:10] = normalize_first_four(raw_data[:,7:10], euler_max, euler_min)
    
    j=1
    for i in range(int(num_test-window-1)):
        test_slide_label[i,:,:]=raw_data[j:j+window,4:10]
        j=j+1
    j=1
    for i in range(int((num_test-1)/window)):  #num_test-30
        test_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window
    j=1
    for i in range(int(num_test-window-1)):
        test_slide_data[i,:,0:4]=raw_data[j:j+window,0:4]
        for k in range(window):
            test_slide_data[i,k,4:10]=raw_data[j-1,4:10]
        j=j+1
    j=1
    for i in range(int((num_test-1)/window)):  #num_test-30
        test_split_data[i,:,0:4]=raw_data[j:j+window,0:4]    #按时间窗划分数据集
        for k in range(window):
            test_split_data[i, k, 4:10] = raw_data[j-1, 4:10]  # 将初始状态扩充到训练样本中
        j=j+window

    train_slide_data=torch.from_numpy(train_slide_data)
    train_slide_data=train_slide_data.float()

    test_slide_data = torch.from_numpy(test_slide_data)
    test_slide_data = test_slide_data.float()

    train_split_data = torch.from_numpy(train_split_data)
    train_split_data = train_split_data.float()

    test_split_data = torch.from_numpy(test_split_data)
    test_split_data = test_split_data.float()

    train_slide_label = torch.from_numpy(train_slide_label)
    train_slide_label = train_slide_label.float()

    test_slide_label = torch.from_numpy(test_slide_label)
    test_slide_label = test_slide_label.float()

    train_split_label = torch.from_numpy(train_split_label)
    train_split_label = train_split_label.float()

    test_split_label = torch.from_numpy(test_split_label)
    test_split_label = test_split_label.float()

    train_slide_dataset=MiningDataset(train_slide_data,train_slide_label)
    train_split_dataset=MiningDataset(train_split_data,train_split_label)
    test_slide_dataset=MiningDataset(test_slide_data,test_slide_label)
    test_split_dataset=MiningDataset(test_split_data,test_split_label)
    train_size = int(0.8 * len(test_slide_dataset))
    val_size = len(test_slide_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(test_slide_dataset, [train_size, val_size])

    return train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset

if __name__=="__main__":
# 统计并保存数据
    stats = analyze_flight_data()

# 读取已保存的统计结果
    loaded_stats = load_flight_stats()