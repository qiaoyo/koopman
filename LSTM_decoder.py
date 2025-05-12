import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

class MultiScaleTimeSeriesModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size=64, lstm_layers=1, fusion_size=128,
                 transformer_d_model=64, transformer_nhead=4, transformer_layers=2, dropout=0.3):
        super(MultiScaleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_dim
        self.transformer_d_model = transformer_d_model  # 将参数保存为类属性
        
        # LSTM编码器部分保持不变
        self.lstm_scale1 = nn.LSTM(input_dim, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale2 = nn.LSTM(input_dim, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale3 = nn.LSTM(input_dim, hidden_size, lstm_layers, batch_first=True)
        
        # Flatten parameters for better memory usage
        self.lstm_scale1.flatten_parameters()
        self.lstm_scale2.flatten_parameters()
        self.lstm_scale3.flatten_parameters()
        
        self.fusion_linear = nn.Linear(hidden_size * 3, fusion_size)
        self.relu = nn.ReLU()
        self.proj_linear = nn.Linear(fusion_size, transformer_d_model)
        
        # 添加LayerNorm层
        self.fusion_layernorm = nn.LayerNorm(fusion_size)
        self.transformer_layernorm = nn.LayerNorm(transformer_d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_layers
        )

        self.ffn1=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )
        
        self.ffn = nn.Sequential(
            # 第一层：扩展维度
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 第二层：保持维度
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第三层：添加残差连接
            ResidualBlock(128, dropout),
            
            # 第四层：降维
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 最后输出层
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        batch_size, seq_length, seq_dim = x.shape

        # Flatten parameters before each forward pass
        self.lstm_scale1.flatten_parameters()
        self.lstm_scale2.flatten_parameters()
        self.lstm_scale3.flatten_parameters()

        # 多尺度LSTM特征提取
        x_scale1 = x
        x_scale2 = x[:, ::2, :]
        x_scale3 = x[:, ::4, :]

        _, (h_n1, _) = self.lstm_scale1(x_scale1)
        feat1 = h_n1[-1]

        _, (h_n2, _) = self.lstm_scale2(x_scale2)
        feat2 = h_n2[-1]

        _, (h_n3, _) = self.lstm_scale3(x_scale3)
        feat3 = h_n3[-1]

        # 特征融合
        fused_feature = torch.cat([feat1, feat2, feat3], dim=-1)
        fused_feature = self.relu(self.fusion_linear(fused_feature))
        fused_feature = self.fusion_layernorm(fused_feature)  # 添加LayerNorm
        
        # 准备Transformer Decoder的输入
        memory = self.proj_linear(fused_feature).unsqueeze(1)  # 编码器输出
        tgt = torch.zeros(batch_size, seq_length, self.transformer_d_model).to(x.device)  # 目标序列初始化
        
        # 生成掩码以确保自回归性质
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)
        
        # Transformer Decoder处理
        transformer_output = self.transformer_decoder(
            tgt=tgt.transpose(0, 1),  # [seq_len, batch_size, d_model]
            memory=memory.transpose(0, 1),  # [1, batch_size, d_model]
            tgt_mask=tgt_mask
        )
        
        # 转换维度并进行最终预测
        x = transformer_output.transpose(0, 1).reshape(-1, transformer_output.size(-1))
        x = self.transformer_layernorm(x)  # 添加LayerNorm
        x = self.ffn(x)
        out = x.view(batch_size, seq_length, -1)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        out = self.layer_norm(out)
        return out

