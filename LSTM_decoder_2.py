import torch
import torch.nn as nn

class MultiScaleTimeSeriesModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=6, hidden_size=64, lstm_layers=2, fusion_size=128,
                 transformer_d_model=64, transformer_nhead=4, transformer_layers=2, dropout=0.2):
        super(MultiScaleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_d_model = transformer_d_model
        
        # 输入特征分组投影
        self.motor_proj = nn.Linear(4, hidden_size // 2)  # 电机信号投影至32维
        self.pose_proj = nn.Linear(6, hidden_size // 2)   # 位姿投影至32维
        
        # 多尺度双向LSTM
        self.lstm_scale1 = nn.LSTM(hidden_size, hidden_size, lstm_layers, 
                                  batch_first=True, bidirectional=True)
        self.lstm_scale2 = nn.LSTM(hidden_size, hidden_size, lstm_layers, 
                                  batch_first=True, bidirectional=True)
        self.lstm_scale3 = nn.LSTM(hidden_size, hidden_size, lstm_layers, 
                                  batch_first=True, bidirectional=True)
        
        # 多尺度特征融合模块（关键修正：调整LayerNorm维度）
        self.scale_fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_size, 2 * hidden_size, kernel_size=1),  # 输入 [B, 3H, L] → 输出 [B, 2H, L]
            nn.BatchNorm1d(2 * hidden_size),  # 对通道维度归一化
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 全局注意力机制
        self.global_attn = nn.MultiheadAttention(
            embed_dim=2 * hidden_size,
            num_heads=transformer_nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * hidden_size,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=2 * hidden_size,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)
        
        # 输出映射层
        self.output_mapping = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.LayerNorm(128),  # 作用于最后一维（通道维度）
            nn.GELU(),
            ResidualBlock(128, dropout),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. 输入分组与投影
        motor_feat = self.motor_proj(x[:, :, :4])     # [B, L, H/2] → [1024, 40, 32]
        pose_feat = self.pose_proj(x[:, :, 4:])       # [B, L, H/2] → [1024, 40, 32]
        combined_feat = torch.cat([motor_feat, pose_feat], dim=-1)  # [B, L, H] → [1024, 40, 64]
        
        # 2. 多尺度LSTM处理（双向特征相加）
        def lstm_forward(lstm, x):
            out, _ = lstm(x)
            return out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]  # 双向融合
        
        scale1_out = lstm_forward(self.lstm_scale1, combined_feat)          # [B, L, H] → [1024, 40, 64]
        scale2_out = lstm_forward(self.lstm_scale2, combined_feat[:, ::2, :])  # [B, L/2, H] → [1024, 20, 64]
        scale3_out = lstm_forward(self.lstm_scale3, combined_feat[:, ::4, :])  # [B, L/4, H] → [1024, 10, 64]
        
        # 3. 上采样至相同序列长度（循环重复）
        scale2_out = scale2_out.repeat(1, 2, 1)[:, :seq_len, :]  # [B, L, H] → [1024, 40, 64]
        scale3_out = scale3_out.repeat(1, 4, 1)[:, :seq_len, :]  # [B, L, H] → [1024, 40, 64]
        
        # 4. 多尺度特征拼接与维度转换（关键修正：先拼接再转换维度）
        multi_scale_feats = torch.cat([scale1_out, scale2_out, scale3_out], dim=-1)  # [B, L, 3H] → [1024, 40, 192]
        multi_scale_feats = multi_scale_feats.transpose(1, 2)  # 转换为 Conv1d 输入格式 [B, 3H, L] → [1024, 192, 40]
        
        # 5. 尺度融合模块（Conv1d + LayerNorm 作用于通道维度）
        fused_feats = self.scale_fusion(multi_scale_feats)  # Conv1d输出 [B, 2H, L] → [1024, 128, 40]
        # 此时 LayerNorm 的输入是 [B, C, L]，通过 normalized_shape=[C] 显式指定归一化通道维度
        # 无需转换维度，LayerNorm 内部会自动处理
        
        # 6. 转换为 [B, L, C] 以适配后续模块（如注意力机制）
        fused_feats = fused_feats.transpose(1, 2)  # [B, L, 2H] → [1024, 40, 128]
        
        # 7. 全局注意力与Transformer编码
        attn_output, _ = self.global_attn(fused_feats, fused_feats, fused_feats)  # [B, L, 2H]
        encoder_output = self.transformer_encoder(attn_output)  # [B, L, 2H]
        
        # 8. 自回归解码
        tgt = torch.zeros_like(encoder_output)  # 初始化目标序列
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)  # [B, L, 2H]
        
        # 9. 输出映射
        output = self.output_mapping(decoder_output)  # [B, L, output_dim] → [1024, 40, 6]
        return output

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # 作用于最后一维（通道维度）
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.net(x)  # 残差连接