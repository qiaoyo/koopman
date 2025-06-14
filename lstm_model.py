import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.1):
        """
        LSTM预测模型
        
        Args:
            input_dim: 输入维度，默认为10
            hidden_dim: LSTM隐藏层维度，默认为128
            num_layers: LSTM层数，默认为2
            output_dim: 输出维度，默认为6
            dropout: Dropout比率，默认为0.1
        """
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, window_len, input_dim]
            
        Returns:
            输出张量，形状为 [batch_size, window_len, output_dim]
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, window_len, hidden_dim]
        
        # 对每个时间步进行预测
        predictions = []
        for t in range(lstm_out.size(1)):
            # 获取当前时间步的隐藏状态
            current_hidden = lstm_out[:, t, :]  # [batch_size, hidden_dim]
            # 通过全连接层得到预测
            pred = self.fc(current_hidden)  # [batch_size, output_dim]
            predictions.append(pred)
        
        # 将所有时间步的预测堆叠起来
        predictions = torch.stack(predictions, dim=1)  # [batch_size, window_len, output_dim]
        
        return predictions

def create_model(input_dim=10, hidden_dim=128, num_layers=2, output_dim=6, dropout=0.1):
    """
    创建LSTM模型实例
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM层数
        output_dim: 输出维度
        dropout: Dropout比率
        
    Returns:
        LSTM模型实例
    """
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    )
    return model

if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    window_len = 40
    input_dim = 10
    output_dim = 6
    
    # 创建模型
    model = create_model()
    
    # 创建随机输入数据
    x = torch.randn(batch_size, window_len, input_dim)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 打印模型结构
    print("\nModel structure:")
    print(model)