import torch
import torch.nn as nn
import math

class MultiScaleTimeSeriesModel(nn.Module):
    """极端简化版本 - 仅保留最核心组件"""
    def __init__(self, input_dim=10, output_dim=6, 
                 lstm_hidden=32, trans_d_model=32):
        super().__init__()
        
        # 最小化嵌入
        self.input_proj = nn.Linear(input_dim, trans_d_model)
        
        # 单层LSTM
        self.lstm = nn.LSTM(trans_d_model, lstm_hidden, batch_first=True)
        
        # 单层Transformer解码器
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=trans_d_model,
            nhead=4,
            dim_feedforward=64,
            batch_first=True
        )
        
        # 直接输出
        self.output = nn.Linear(trans_d_model, output_dim)
    
    def forward(self, x):
        x_emb = self.input_proj(x)
        
        # LSTM特征提取
        _, (h_n, _) = self.lstm(x_emb)
        lstm_feat = h_n[-1].unsqueeze(1)
        
        # Transformer解码
        seq_len = x_emb.size(1)
        tgt = torch.zeros_like(x_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        decoded = self.decoder_layer(
            tgt=tgt,
            memory=lstm_feat,
            tgt_mask=tgt_mask
        )
        
        return self.output(decoded)