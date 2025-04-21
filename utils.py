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
    
    j=1
    for i in range(int(num_train-window-1)):  #num_train-30
        train_slide_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+1
    j=1
    for i in range(int((num_train-1)/window)):  #num_train-30
        train_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window

    for i in range(num_train):
        raw_data[i, 0] = raw_data[i, 0]/ 198.8755
        raw_data[i, 1] = raw_data[i, 1] / 183.4
        raw_data[i, 2] = raw_data[i, 2] / 183.5725
        raw_data[i, 3] = raw_data[i, 3] / 184.4340

    j=1
    for i in range(int(num_train-window-1)):  #num_train-30
        train_slide_data[i,:,0:4]=raw_data[j:j+window,0:4]
        for k in range(window):
            train_slide_data[i, k, 4:10] = raw_data[j-1, 4:10]  #将初始状态扩充到训练样本中
        j=j+1

    j=1
    for i in range(int((num_train-1)/window)):  # num_train-30
        train_split_data[i, :, 0:4] = raw_data[j:j+window, 0:4]
        for k in range(window):
            train_split_data[i, k, 4:10] = raw_data[j-1, 4:10]  # 将初始状态扩充到训练样本中
        j = j + window

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

    j=1
    for i in range(int(num_test-window-1)):
        test_slide_label[i,:,:]=raw_data[j:j+window,4:10]
        j=j+1
    j=1
    for i in range(int((num_test-1)/window)):  #num_test-30
        test_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window

    #数据归一化过程，除最大值
    for i in range(num_test):
        raw_data[i, 0] = raw_data[i, 0]/198.8755
        raw_data[i, 1] = raw_data[i, 1] / 183.4
        raw_data[i, 2] = raw_data[i, 2] / 183.5725
        raw_data[i, 3] = raw_data[i, 3] / 184.4340
    
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

    return train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset

def prepare_data(window=80):
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
    
    j=1
    for i in range(int(num_train-window-1)):  #num_train-30
        train_slide_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+1
    j=1
    for i in range(int((num_train-1)/window)):  #num_train-30
        train_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window

    for i in range(num_train):
        raw_data[i, 0] = raw_data[i, 0]/ 198.8755
        raw_data[i, 1] = raw_data[i, 1] / 183.4
        raw_data[i, 2] = raw_data[i, 2] / 183.5725
        raw_data[i, 3] = raw_data[i, 3] / 184.4340

    j=1
    for i in range(int(num_train-window-1)):  #num_train-30
        train_slide_data[i,:,0:4]=raw_data[j:j+window,0:4]
        for k in range(window):
            train_slide_data[i, k, 4:10] = raw_data[j-1, 4:10]  #将初始状态扩充到训练样本中
        j=j+1

    j=1
    for i in range(int((num_train-1)/window)):  # num_train-30
        train_split_data[i, :, 0:4] = raw_data[j:j+window, 0:4]
        for k in range(window):
            train_split_data[i, k, 4:10] = raw_data[j-1, 4:10]  # 将初始状态扩充到训练样本中
        j = j + window

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

    j=1
    for i in range(int(num_test-window-1)):
        test_slide_label[i,:,:]=raw_data[j:j+window,4:10]
        j=j+1
    j=1
    for i in range(int((num_test-1)/window)):  #num_test-30
        test_split_label[i,:,:]=raw_data[j:j+window,4:10]    #按时间窗划分数据集
        j=j+window

    #数据归一化过程，除最大值
    for i in range(num_test):
        raw_data[i, 0] = raw_data[i, 0]/198.8755
        raw_data[i, 1] = raw_data[i, 1] / 183.4
        raw_data[i, 2] = raw_data[i, 2] / 183.5725
        raw_data[i, 3] = raw_data[i, 3] / 184.4340
    
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
    train_dataset, val_dataset = torch.utils.data.random_split(test_slide_dataset, [train_size, val_size])

    return train_dataset,val_dataset
    # return train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset
