import torch
import numpy as np
from utils import prepare_full_data
from sample import MultiScaleTimeSeriesModel
import os

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def test_model(model, test_loader, device):
    model.eval()
    first_three_errors = []
    last_three_errors = []
    
    with torch.no_grad():
        for ini_datas, labels in test_loader:
            ini_datas = ini_datas.to(device)
            labels = labels.to(device)
            output = model(ini_datas)
            
            # 计算前三个自由度的平均误差
            first_three_error = torch.mean(torch.abs(output[:, :, :3] - labels[:, :, :3]))
            # 计算后三个自由度的平均误差
            last_three_error = torch.mean(torch.abs(output[:, :, 3:] - labels[:, :, 3:]))
            
            first_three_errors.append(first_three_error.item())
            last_three_errors.append(last_three_error.item())
    
    avg_first_three_error = np.mean(first_three_errors)
    avg_last_three_error = np.mean(last_three_errors)
    
    return avg_first_three_error, avg_last_three_error

if __name__ == "__main__":
    # 设置参数
    window = 80
    batch_size = 512
    device = set_device()
    
    # 准备测试数据
    _, _, test_slide_dataset, _ = prepare_full_data(window=window)
    test_slide_loader = torch.utils.data.DataLoader(
        dataset=test_slide_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    
    # 加载模型
    model = MultiScaleTimeSeriesModel(input_size=10, output_dim=6)
    model = model.to(device)
    
    # 加载已保存的模型权重
    checkpoint_path = r'C:\Users\Administrator\Desktop\koopman-data\data\test_decode\best_model_0421.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载成功")
    else:
        print("未找到保存的模型文件")
        exit()
    
    # 测试模型
    first_three_error, last_three_error = test_model(model, test_slide_loader, device)
    
    print(f"前三个自由度的平均误差: {first_three_error:.4f}")
    print(f"后三个自由度的平均误差: {last_three_error:.4f}")