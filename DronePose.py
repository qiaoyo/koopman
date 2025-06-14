import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # 计算保持长度不变的填充大小
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 下采样层用于调整通道数
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # 保存填充值用于裁剪
        self.padding = padding
        
    def forward(self, x):
        residual = x
        
        # 第一层卷积
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # 裁剪右侧填充部分
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # 第二层卷积
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # 裁剪右侧填充部分
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        # 调整残差连接的维度
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # 确保残差连接和输出长度匹配
        if residual.size(2) > out.size(2):
            residual = residual[:, :, :out.size(2)]
        elif residual.size(2) < out.size(2):
            out = out[:, :, :residual.size(2)]
        
        return out + residual

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.layer_norm(self.block(x) + x)

class DronePosePredictor(nn.Module):
    def __init__(self, input_dim=10, output_dim=6, 
                 gru_hidden_size=128, gru_layers=2, 
                 tcn_channels=64, tcn_layers=4, 
                 dropout=0.2):
        super().__init__()
        
        # 特征分离层
        self.motor_embed = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU()
        )
        self.pose_embed = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU()
        )
        
        # 双分支特征提取
        # 分支1: GRU处理时序特征
        self.gru = nn.GRU(48, gru_hidden_size, gru_layers, 
                         batch_first=True, bidirectional=True)
        
        # 分支2: TCN处理局部特征
        tcn_blocks = []
        in_ch = 48  # 初始输入通道数
        
        for i in range(tcn_layers):
            dilation = 2 ** (i % 5)  # 循环使用不同扩张率
            tcn_blocks.append(
                TCNBlock(in_ch, tcn_channels, 
                         kernel_size=3, 
                         dilation=dilation, 
                         dropout=dropout)
            )
            in_ch = tcn_channels  # 后续层的输入通道数
            
        self.tcn = nn.Sequential(*tcn_blocks)
        
        # 特征融合与输出
        self.fusion = nn.Sequential(
            nn.Linear(2*gru_hidden_size + tcn_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            ResidualBlock(128, dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # 分离特征 (batch, seq, 10)
        motor = x[..., :4]  # 电机速度
        pose = x[..., 4:]   # 上时刻位姿
        
        # 特征嵌入
        motor_emb = self.motor_embed(motor)  # (batch, seq, 16)
        pose_emb = self.pose_embed(pose)     # (batch, seq, 32)
        combined = torch.cat([motor_emb, pose_emb], dim=-1)  # (batch, seq, 48)
        
        # 分支1: GRU处理
        gru_out, _ = self.gru(combined)  # (batch, seq, 2*hidden)
        
        # 分支2: TCN处理 (需调整维度)
        tcn_in = combined.transpose(1, 2)  # (batch, 48, seq)
        tcn_out = self.tcn(tcn_in)         # (batch, tcn_channels, seq)
        tcn_out = tcn_out.transpose(1, 2)   # (batch, seq, tcn_channels)
        
        # 特征融合
        fused = torch.cat([gru_out, tcn_out], dim=-1)  # (batch, seq, 2*hidden+tcn)
        features = self.fusion(fused)
        
        return self.output(features)

# 修正后的损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss()
        self.alpha = alpha
        
    def forward(self, input, target):
        return self.alpha * self.mse(input, target) + (1 - self.alpha) * self.huber(input, target)
    
class CombinedLoss_without_huber(nn.Module):
    def __init__(self, position_weight=1.0, orientation_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # 分离位置和旋转
        pred_position = pred[:, :, :3]  # 前三个是位置
        pred_orientation = pred[:, :, 3:]  # 后三个是旋转
        target_position = target[:, :, :3]
        target_orientation = target[:, :, 3:]
        
        # 计算位置和旋转的损失
        position_loss = self.mse(pred_position, target_position)
        orientation_loss = self.mse(pred_orientation, target_orientation)
        
        # 应用权重
        total_loss = (self.position_weight * position_loss + 
                     self.orientation_weight * orientation_loss)
        
        return total_loss

class CombinedLoss_total(nn.Module):
    def __init__(self, position_weight=1.0, orientation_weight=1.0, alpha=0.5):
        super(CombinedLoss_total, self).__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.alpha = alpha  # 调节MSE和Huber损失的比例
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss()
        
    def forward(self, pred, target):
        # 分离位置和旋转
        pred_position = pred[:, :, :3]  # 前三个是位置
        pred_orientation = pred[:, :, 3:]  # 后三个是旋转
        target_position = target[:, :, :3]
        target_orientation = target[:, :, 3:]
        
        # 计算位置和旋转的MSE损失
        position_mse = self.mse(pred_position, target_position)
        orientation_mse = self.mse(pred_orientation, target_orientation)
        
        # 计算位置和旋转的Huber损失
        position_huber = self.huber(pred_position, target_position)
        orientation_huber = self.huber(pred_orientation, target_orientation)
        
        # 组合MSE和Huber损失
        position_loss = self.alpha * position_mse + (1 - self.alpha) * position_huber
        orientation_loss = self.alpha * orientation_mse + (1 - self.alpha) * orientation_huber
        
        # 应用位置和旋转的权重
        total_loss = (self.position_weight * position_loss + 
                     self.orientation_weight * orientation_loss)
        
        return total_loss