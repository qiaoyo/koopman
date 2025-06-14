import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 计算padding以保持序列长度不变
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                                self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # 确保输出和残差连接具有相同的长度
        if out.size(-1) != res.size(-1):
            out = out[:, :, :res.size(-1)]
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNPredictor(nn.Module):
    def __init__(self, input_dim=10, output_dim=6, num_channels=[64, 128, 256], kernel_size=2, dropout=0.2):
        super(TCNPredictor, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        
        # 计算TCN输出后的特征维度
        self.feature_dim = num_channels[-1]
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, output_dim)
        )

    def forward(self, x):
        # 输入: [batch_size, seq_len, input_dim]
        # 转换为TCN需要的格式: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # TCN处理
        tcn_out = self.tcn(x)  # [batch_size, num_channels[-1], seq_len]
        
        # 转换回原始格式
        tcn_out = tcn_out.transpose(1, 2)  # [batch_size, seq_len, num_channels[-1]]
        
        # 对每个时间步进行预测
        predictions = []
        for t in range(tcn_out.size(1)):
            current_features = tcn_out[:, t, :]  # [batch_size, num_channels[-1]]
            pred = self.output_layer(current_features)  # [batch_size, output_dim]
            predictions.append(pred)
        
        # 堆叠所有时间步的预测
        predictions = torch.stack(predictions, dim=1)  # [batch_size, seq_len, output_dim]
        
        return predictions

def create_model(input_dim=10, output_dim=6, num_channels=[64, 128, 256], kernel_size=2, dropout=0.2):
    """
    创建TCN模型实例
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        num_channels: TCN每层的通道数列表
        kernel_size: 卷积核大小
        dropout: Dropout比率
        
    Returns:
        TCN模型实例
    """
    model = TCNPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    return model

if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    seq_len = 40
    input_dim = 10
    output_dim = 6
    
    # 创建模型
    model = create_model()
    
    # 创建随机输入数据
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 打印模型结构
    print("\nModel structure:")
    print(model) 