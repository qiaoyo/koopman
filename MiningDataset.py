import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MiningDataset():
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data, label, folder_idx=None, norm_type='total'):
        """
        初始化数据集
        Args:
            data: 输入数据
            label: 标签数据
            folder_idx: 数据所属的文件夹索引（用于folder归一化）
            norm_type: 归一化类型，'folder'或'total'
        """
        self.data = data
        self.label = label
        self.folder_idx = folder_idx
        self.norm_type = norm_type
        self.len = int(self.data.shape[0])

    def __getitem__(self, i):
        """
        获取数据集中的一个样本
        Args:
            i: 样本索引
        Returns:
            con_data: 输入数据
            label: 标签数据
            folder_idx: 文件夹索引（如果使用folder归一化）
        """
        con_data = self.data[i,:,:]
        label = self.label[i,:,:]
        if self.folder_idx is not None:
            return con_data, label, self.folder_idx[i]
        return con_data, label, 1

    def __len__(self):
        return self.len