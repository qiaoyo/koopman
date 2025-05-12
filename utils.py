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
import time
import os
import random
from MiningDataset import MiningDataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.set_printoptions(threshold=np.inf,precision=4,suppress=True)
torch.set_printoptions(threshold=float('inf'), precision=4, sci_mode=False)

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
    return device

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def denormalize_data(data, max_values, min_values):
    """将归一化的数据恢复到原始尺度"""
    denorm_data = data.clone()
    for i in range(len(max_values)): 
        denorm_data[..., i] = data[..., i] * (max_values[i] - min_values[i]) + min_values[i]
    return denorm_data

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
    base_path = '/home/pika/koopman-data/data/flights'
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    # 初始化字典存储结果
    stats = {}
    first_run = True
    
    # 遍历0-53文件夹
    for folder_idx in range(54):
        folder_path = os.path.join(base_path, str(folder_idx))
        if os.path.exists(folder_path):
            # 初始化当前文件夹的统计信息
            folder_stats = {}
            
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
                    
                    # 初始化当前文件夹中该文件的统计信息
                    folder_stats[key] = {
                        'max': np.full(data.shape[1], float('-inf')),
                        'min': np.full(data.shape[1], float('inf'))
                    }
                    
                    # 更新每个维度的最大最小值
                    for dim in range(data.shape[1]):
                        current_max = np.max(data[:, dim])
                        current_min = np.min(data[:, dim])
                        stats[key]['max'][dim] = max(stats[key]['max'][dim], current_max)
                        stats[key]['min'][dim] = min(stats[key]['min'][dim], current_min)
                        folder_stats[key]['max'][dim] = current_max
                        folder_stats[key]['min'][dim] = current_min
            
            # 保存当前文件夹的统计信息
            folder_stats_path = os.path.join(folder_path, 'folder_stats.npy')
            np.save(folder_stats_path, folder_stats)
            print(f"\n文件夹 {folder_idx} 的统计结果已保存至: {folder_stats_path}")
            
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
    
    # 保存总体统计结果
    save_path = os.path.join(base_path, 'flight_stats.npy')
    np.save(save_path, stats)
    print(f"\n总体统计结果已保存至: {save_path}")
    
    # 保存Fisher字典
    fisher_dict = {
        'Motors_CMD': {'mean': np.array([0.5, 0.5, 0.5, 0.5]), 'std': np.array([0.1, 0.1, 0.1, 0.1])},
        'Pos': {'mean': np.array([0.5, 0.5, 0.5]), 'std': np.array([0.1, 0.1, 0.1])},
        'Euler': {'mean': np.array([0.5, 0.5, 0.5]), 'std': np.array([0.1, 0.1, 0.1])}
    }
    fisher_path = os.path.join(base_path, 'fisher_info.npy')
    np.save(fisher_path, fisher_dict)
    print(f"\nFisher字典已保存至: {fisher_path}")
    
    return stats

def load_flight_stats(show_stats=False):
    """
    读取已保存的飞行数据统计结果
    """
    base_path = '/home/pika/koopman-data/data/flights'
    stats_path = os.path.join(base_path, 'flight_stats.npy')
    
    if not os.path.exists(stats_path):
        print("未找到统计文件，请先运行analyze_flight_data()函数")
        return None
    
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # 打印加载的结果
    if show_stats:
        print("\n加载的数据统计结果:")
        for key in stats:
            print(f"\n{key}:")
            print(f"最大值: {stats[key]['max']}")
            print(f"最小值: {stats[key]['min']}")
    
    return stats

def normalize_data(data, max_values=None, min_values=None):
    print(data.shape)
    if type(data) == torch.Tensor:
        normalized_data = data.clone()
    elif type(data) == np.ndarray:
        normalized_data = data.copy()
    else:
        raise ValueError("Invalid data type")
    
    # 只对前4列进行归一化
    for i in range(len(max_values)):
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
    raw_data[:,0:4] = normalize_data(raw_data[:,0:4], motor_cmd_max, motor_cmd_min)
    # raw_data[:,4:7] = normalize_data(raw_data[:,4:7], pos_max, pos_min)
    # raw_data[:,7:10] = normalize_data(raw_data[:,7:10], euler_max, euler_min)    
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

    raw_data[:,0:4] = normalize_data(raw_data[:,0:4], motor_cmd_max, motor_cmd_min)
    # raw_data[:,4:7] = normalize_data(raw_data[:,4:7], pos_max, pos_min)
    # raw_data[:,7:10] = normalize_data(raw_data[:,7:10], euler_max, euler_min)
    
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

def prepare_data_with_folder(folder=0, window=80, return_split=False,return_norm_data=False,augmentation=False):
    """
    Read and process data from specified folder
    Args:
        folder: Folder number, default 0
        window: Time window size, default 80
        return_split: Whether to return split dataset, default False
    Returns:
        slide_dataset: Sliding window dataset
        split_dataset: Split dataset (if return_split=True)
    """ 
    from tqdm import tqdm
    
    base_path = '/home/pika/koopman-data/data/flights'
    if return_norm_data:
        folder_path = os.path.join(base_path, str(folder), 'normed_data')
    else:
        folder_path = os.path.join(base_path, str(folder))
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist")
    
    print(f"Loading data from group {folder}...")
    
    # Read data
    data_dict = {}
    for file_name in tqdm(target_files, desc="Loading data files"):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        data = np.load(file_path)
        print(data.shape,file_name)
        data_dict[file_name.split('.')[0]] = data
        folder_path = os.path.join(base_path, str(folder))
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    loaded_stats=load_flight_stats()
    motor_cmd_max = loaded_stats['Motors_CMD']['max']
    motor_cmd_min = loaded_stats['Motors_CMD']['min']
    
    print("Normalize data...")
    # 归一化Motors_CMD数据
    data_dict['Motors_CMD'] = normalize_data(
        data_dict['Motors_CMD'], 
        motor_cmd_max, 
        motor_cmd_min
    )
    data_length=len(data)
    # 创建滑动窗口数据集
    num_slide_samples = data_length - window - 1
    slide_data = np.zeros((num_slide_samples, window, 10))
    slide_label = np.zeros((num_slide_samples, window, 6))
    
    print("Fill slide window data...")
    # 填充滑动窗口数据
    for i in tqdm(range(num_slide_samples), desc="deal slide data"):
        j = i + 1
        # 填充输入数据
        slide_data[i, :, 0:4] = data_dict['Motors_CMD'][j:j+window]
        # 填充初始状态
        for k in range(window):
            slide_data[i, k, 4:7] = data_dict['Pos'][j-1]
        j = i + 1
        # 填充输入数据
        slide_data[i, :, 0:4] = data_dict['Motors_CMD'][j:j+window]
        # 填充初始状态
        for k in range(window):
            slide_data[i, k, 4:7] = data_dict['Pos'][j-1]
            slide_data[i, k, 7:10] = data_dict['Euler'][j-1]
        # 填充标签
        slide_label[i, :, 0:3] = data_dict['Pos'][j:j+window]
        slide_label[i, :, 3:6] = data_dict['Euler'][j:j+window]
    
    print("Augment data...")
    if augmentation:
        slide_data = time_series_augmentation(slide_data)
    
    # 转换为torch张量
    print("numpy 2 torch tensor...")
    slide_data = torch.from_numpy(slide_data).float()
    slide_label = torch.from_numpy(slide_label).float()

    # 创建数据集
    slide_dataset = MiningDataset(slide_data, slide_label,norm_data=None)
   
    if return_split:
        print("deal split data...")
        # 创建分割数据集
        num_split_samples = (data_length - 1) // window
        split_data = np.zeros((num_split_samples, window, 10))
        split_label = np.zeros((num_split_samples, window, 6))
        
        # 填充分割数据
        for i in tqdm(range(num_split_samples), desc="deal split data"):
            j = i * window + 1
            # 填充输入数据
            split_data[i, :, 0:4] = data_dict['Motors_CMD'][j:j+window]
            # 填充初始状态
            for k in range(window):
                split_data[i, k, 4:7] = data_dict['Pos'][j-1]
                split_data[i, k, 7:10] = data_dict['Euler'][j-1]
            # 填充标签
            split_label[i, :, 0:3] = data_dict['Pos'][j:j+window]
            split_label[i, :, 3:6] = data_dict['Euler'][j:j+window]
    
        split_data = torch.from_numpy(split_data).float()
        split_label = torch.from_numpy(split_label).float()   
        split_dataset = MiningDataset(split_data, split_label,norm_data=None)
        
        print("data done")
        return slide_dataset, split_dataset
    
    print("data done")
    return slide_dataset

def prepare_merged_data(folders=[], window=80, return_split=False, return_norm_data=False,augmentation=False):
    """
    Merge multiple folders' datasets into one
    Args:
        folders: List of folder numbers to merge
        window: Time window size, default 80
        return_split: Whether to return split dataset, default False
    Returns:
        merged_slide_dataset: Merged sliding window dataset
        merged_split_dataset: Merged split dataset (if return_split=True)
    """
    if not folders:
        raise ValueError("Folders list cannot be empty")
    
    print(f"Preparing to merge data from folders: {folders}")
    
    # 存储所有文件夹的数据
    all_slide_data = []
    all_slide_labels = []
    all_split_data = []
    all_split_labels = []
    
    # 遍历每个文件夹并收集数据
    for folder in folders:
        print(f"\nProcessing folder {folder}...")
        if return_split:
            slide_dataset, split_dataset = prepare_data_with_folder(
                folder=folder,
                window=window,
                return_split=True,
                return_norm_data=return_norm_data
            )
            # 收集分割数据集
            all_split_data.append(split_dataset.data)
            all_split_labels.append(split_dataset.label)
        else:
            slide_dataset = prepare_data_with_folder(
                folder=folder,
                window=window,
                return_split=False,
                return_norm_data=return_norm_data,
                augmentation=augmentation
            )
        
        # 收集滑动窗口数据集
        all_slide_data.append(slide_dataset.data)
        all_slide_labels.append(slide_dataset.label)
    
    # 合并滑动窗口数据集
    merged_slide_data = torch.cat(all_slide_data, dim=0)
    merged_slide_labels = torch.cat(all_slide_labels, dim=0)
    merged_slide_dataset = MiningDataset(merged_slide_data, merged_slide_labels)
    
    print(f"\nMerged sliding window dataset size: {len(merged_slide_dataset)}")
    
    if return_split:
        # 合并分割数据集
        merged_split_data = torch.cat(all_split_data, dim=0)
        merged_split_labels = torch.cat(all_split_labels, dim=0)
        merged_split_dataset = MiningDataset(merged_split_data, merged_split_labels)
        print(f"Merged split dataset size: {len(merged_split_dataset)}")
        return merged_slide_dataset, merged_split_dataset
    
    return merged_slide_dataset

def norm_data_and_save():
    base_path = '/home/pika/koopman-data/data/flights'
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']

    for folder in range(54):
        folder_path = os.path.join(base_path, str(folder))
        if os.path.exists(folder_path):
            # 使用.item()获取numpy数组中的字典数据
            norm_data = np.load(os.path.join(folder_path, 'folder_stats.npy'), allow_pickle=True).item()
            norm_file_path = os.path.join(folder_path, 'normed_data')
            if not os.path.exists(norm_file_path):
                os.makedirs(norm_file_path)

            for file_name in target_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    if file_name == 'Motors_CMD.npy':
                        data = normalize_data(data, norm_data['Motors_CMD']['max'], norm_data['Motors_CMD']['min'])
                    elif file_name == 'Pos.npy':
                        data = normalize_data(data, norm_data['Pos']['max'], norm_data['Pos']['min'])
                    elif file_name == 'Euler.npy':
                        data = normalize_data(data, norm_data['Euler']['max'], norm_data['Euler']['min'])
                    np.save(os.path.join(norm_file_path, file_name), data)

if __name__=="__main__":
    # 创建保存目录
    save_dir = '/home/pika/koopman-data/data/processed'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    slide_data=prepare_data_with_folder(folder=0,window=5,return_split=False,return_norm_data=False,augmentation=False)
    
    print(slide_data.len)
    print(slide_data[0][0])
    print(slide_data[0][1])

    base_path='/home/pika/koopman-data/data/flights'
    files_name=['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    data_1=np.load(os.path.join(base_path,'0',files_name[0]))
    data_2=np.load(os.path.join(base_path,'0',files_name[1]))
    data_3=np.load(os.path.join(base_path,'0',files_name[2]))
    print(data_1[0:6])
    print(data_2[0:6])
    print(data_3[0:6])

    # 生成0-53的随机排列
    # all_folders = list(range(54))
    # np.random.shuffle(all_folders)
    
    # # 按照6:2:2的比例划分
    # train_folders = all_folders[:32]  # 60%
    # test_folders = all_folders[32:43]  # 20%
    # online_folders = all_folders[43:]  # 20%
    
    # # 保存划分结果
    # np.save(os.path.join(save_dir, 'train_folders.npy'), train_folders)
    # np.save(os.path.join(save_dir, 'test_folders.npy'), test_folders)
    # np.save(os.path.join(save_dir, 'online_folders.npy'), online_folders)
    
    # 加载并打印结果
    train_folders = np.load(os.path.join(save_dir, 'train_folders.npy'))
    test_folders = np.load(os.path.join(save_dir, 'test_folders.npy'))
    online_folders = np.load(os.path.join(save_dir, 'online_folders.npy'))
    
    print("训练集文件夹:", sorted(train_folders))
    print("测试集文件夹:", sorted(test_folders))
    print("在线训练集文件夹:", sorted(online_folders))
