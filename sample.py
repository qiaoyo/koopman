import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from koopman import DT_transformer
import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(42)

# 1. 数据集生成
class SyntheticTimeSeriesDataset(Dataset):
    def __init__(self, seq_length=200, num_samples=1000, noise_std=0.1):
        """
        生成虚拟的多尺度时间序列数据。
        数据由多个正弦波叠加构成，包含低频、中频、高频成分，加上高斯噪声。
        """
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.data = []
        self.targets = []
        t = np.linspace(0, 10, seq_length)
        for _ in range(num_samples):
            A1, f1, phi1 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.0), np.random.uniform(0, 2 * np.pi)
            A2, f2, phi2 = np.random.uniform(0.2, 1.0), np.random.uniform(1.5, 3.0), np.random.uniform(0, 2 * np.pi)
            A3, f3, phi3 = np.random.uniform(0.1, 0.5), np.random.uniform(3.0, 6.0), np.random.uniform(0, 2 * np.pi)
            signal = A1 * np.sin(2 * np.pi * f1 * t + phi1) + \
                     A2 * np.sin(2 * np.pi * f2 * t + phi2) + \
                     A3 * np.sin(2 * np.pi * f3 * t + phi3)
            signal += np.random.normal(0, noise_std, size=seq_length)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-5)
            self.data.append(signal.astype(np.float32))
            self.targets.append(signal[-1].astype(np.float32))

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 2. 模型构建
class MultiScaleTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, output_dim=1,hidden_size=64, lstm_layers=1, fusion_size=128,
                 transformer_d_model=128, transformer_nhead=4, transformer_layers=2, dropout=0.1):
        super(MultiScaleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm_scale1 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale2 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale3 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)

        self.fusion_linear = nn.Linear(hidden_size * 3, fusion_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # self.out_linear = nn.Linear(transformer_d_model, 1)
        self.relu = nn.ReLU()
        self.proj_linear = nn.Linear(fusion_size, transformer_d_model)
        self.ffn=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )

    def forward(self, x):
        batch_size, seq_length, seq_dim = x.shape

        x_scale1 = x
        x_scale2 = x[:, ::2, :]
        x_scale3 = x[:, ::4, :]
        # print(x_scale1.shape,x_scale2.shape,x_scale3.shape)
        _, (h_n1, _) = self.lstm_scale1(x_scale1)
        feat1 = h_n1[-1]

        _, (h_n2, _) = self.lstm_scale2(x_scale2)
        feat2 = h_n2[-1]

        _, (h_n3, _) = self.lstm_scale3(x_scale3)
        feat3 = h_n3[-1]
        # print(feat1.shape,feat2.shape,feat3.shape)
        fused_feature = torch.cat([feat1, feat2, feat3], dim=-1)
        # print(fused_feature.shape)
        fused_feature = self.relu(self.fusion_linear(fused_feature))
        # print(fused_feature.shape)
        transformer_input = fused_feature.unsqueeze(1).repeat(1, seq_length, 1)
        transformer_input = self.proj_linear(transformer_input)
        # transformer_input = transformer_input.transpose(0, 1)
        # print(transformer_input.shape)
        transformer_output = self.transformer_encoder(transformer_input)
        # print(transformer_output.shape)
        # step predict
        x=transformer_output.reshape(-1,transformer_output.size(-1))
        # final step predict
        # x=transformer_output[:,-1,:]
        x=self.ffn(x)
        # final_feature = transformer_output[-1, :, :]
        # out = self.out_linear(final_feature)
        # out = out.squeeze(-1)
        out=x.view(batch_size,seq_length,-1)
        return out

# 3. 训练与评估函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=50, device='cpu'):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return train_losses

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    eval_loss = 0.0
    preds = []
    trues = []
    transformer_outputs = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, trans_out = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item() * inputs.size(0)
            preds.append(outputs.cpu().numpy())
            trues.append(targets.cpu().numpy())
            transformer_outputs.append(trans_out[:, 0, :].cpu().numpy())
    eval_loss = eval_loss / len(dataloader.dataset)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    transformer_outputs = np.concatenate(transformer_outputs, axis=1)
    return eval_loss, preds, trues, transformer_outputs

def plot_results(train_losses, sample_time, sample_signal, transformer_feature, preds, trues):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    axs[0, 0].plot(sample_time, sample_signal, color='tab:blue', linewidth=2)
    axs[0, 0].set_title("Original Time Series Example", fontsize=14)
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Signal Amplitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].text(0.1, 0.9, "Note: Demonstrates the mixture of multi-scale periodic signals and noise",
                  transform=axs[0, 0].transAxes, fontsize=12, color='purple')

    axs[0, 1].plot(np.arange(len(train_losses)), train_losses, color='tab:red', linewidth=2)
    axs[0, 1].set_title("Training Loss Curve", fontsize=14)
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("MSE Loss")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    im = axs[1, 0].imshow(transformer_feature, aspect='auto', cmap='viridis')
    axs[1, 0].set_title("Transformer Feature Heatmap", fontsize=14)
    axs[1, 0].set_xlabel("Sample Index")
    axs[1, 0].set_ylabel("Time Step")
    fig.colorbar(im, ax=axs[1, 0])

    axs[1, 1].plot(preds, label='Predicted', color='tab:green', marker='o', linestyle='--')
    axs[1, 1].plot(trues, label='True', color='tab:orange', marker='x', linestyle='-')
    axs[1, 1].set_title("Prediction vs. True", fontsize=14)
    axs[1, 1].set_xlabel("Sample Index")
    axs[1, 1].set_ylabel("Signal Amplitude")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# 4. 主函数
if __name__ == '__main__':
    seq_length = 50
    num_samples = 60
    batch_size = 4
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    # 构造数据集与DataLoader
    dataset = SyntheticTimeSeriesDataset(seq_length=seq_length, num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model=MultiScaleTimeSeriesModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("开始模型训练：")
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    val_loss, preds, trues, transformer_feature = evaluate_model(model, val_loader, criterion, device=device)
    print(f"验证集损失：{val_loss:.4f}")

    sample_idx = 0
    sample_signal, _ = dataset[sample_idx]
    sample_time = np.linspace(0, 10, seq_length)

    plot_results(train_losses, sample_time, sample_signal, transformer_feature, preds, trues)
