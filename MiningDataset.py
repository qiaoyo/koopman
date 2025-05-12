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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MiningDataset():
    """
    A Pytorch Dataset class to be used in PyTorch DataLoader to create batches
    """
    def __init__(self, data, label, norm_data=None):
        self.data = data
        self.label = label
        self.norm_data = norm_data
        self.len = int(self.data.shape[0])

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
        return self.len
