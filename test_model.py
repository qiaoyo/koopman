import torch
import numpy as np
from utils import prepare_full_data
from LSTM_decoder import MultiScaleTimeSeriesModel
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf,precision=4,suppress=True)

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def denormalize_data(data, max_values, min_values, start_idx=4):
    """将归一化的数据恢复到原始尺度"""
    denorm_data = data.clone()
    for i in range(6):  # 6个输出维度
        denorm_data[..., i] = data[..., i] * (max_values[i+start_idx] - min_values[i+start_idx]) + min_values[i+start_idx]
    return denorm_data

def visualize_predictions(output, labels, save_dir):
    """可视化预测结果和真实值"""
    # 只取最后一个时间步的数据
    pred = output[:, -1, :].cpu().numpy()  # 所有样本的最后一步
    true = labels[:, -1, :].cpu().numpy()  # 所有样本的最后一步
    samples = range(len(pred))
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('comparison of prediction and label (final step)')
    
    # 绘制6个自由度的对比图
    for i in range(6):
        row = i // 3
        col = i % 3
        axes[row, col].plot(samples, true[:, i], 'b-', label='label')
        axes[row, col].plot(samples, pred[:, i], 'r--', label='prediction')
        axes[row, col].set_title(f'dof {i+1}')
        axes[row, col].set_xlabel('sample index')
        axes[row, col].set_ylabel('value')
        axes[row, col].legend()
        axes[row, col].grid(True)
        # 设置y轴格式为4位小数
        axes[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()

def test_model(model, test_loader, device, save_dir=None):
    model.eval()
    # 初始化6个自由度的误差列表
    dof_errors = [[] for _ in range(6)]
    
    # max_values = norm_params['max_values']
    # min_values = norm_params['min_values']
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for ini_datas, labels in test_loader:
            ini_datas = ini_datas.to(device)
            labels = labels.to(device)
            output = model(ini_datas)
            
            # 收集所有batch的输出
            all_outputs.append(output)
            all_labels.append(labels)
            
            # 计算每个自由度的误差
            for i in range(6):
                error = torch.mean(torch.abs(output[:, -1, i] - labels[:, -1, i]))
                dof_errors[i].append(error.item())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if save_dir is not None:
        visualize_predictions(all_outputs, all_labels, save_dir)
    # 计算每个自由度的平均误差
    first_three_errors = np.array([np.mean(dof_errors[i]) for i in range(3)])
    last_three_errors = np.array([np.mean(dof_errors[i]) for i in range(3, 6)])
    
    return first_three_errors, last_three_errors

if __name__ == "__main__":
    # 设置参数
    window = 80
    batch_size = 512
    device = set_device()
    
    # 创建保存目录
    save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\LSTM_decoder_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载已保存的模型权重
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    checkpoint_path = os.path.join(save_dir, 'best_model_online.pth')
    
    # 准备测试数据
    train_slide_dataset, _, test_slide_dataset, _ = prepare_full_data(window=window)

    train_slide_loader = torch.utils.data.DataLoader(
        dataset=train_slide_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    test_slide_loader = torch.utils.data.DataLoader(
        dataset=test_slide_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    # 加载模型
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)

    model = model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载成功")
    else:
        print("未找到保存的模型文件")
        exit()
    
    train_save_dir=os.path.join(save_dir, 'predictions_visualization_final_train.png')
    test_save_dir=os.path.join(save_dir, 'predictions_visualization_final_test_online.png')
    # 测试模型
    first_three_error, last_three_error = test_model(model, test_slide_loader, device, test_save_dir)
    
    print(f"test 前三个自由度的平均误差（原始尺度）: {first_three_error}")
    print(f"test 后三个自由度的平均误差（原始尺度）: {last_three_error}")

    # first_three_error, last_three_error = test_model(model, train_slide_loader, device, train_save_dir)
    
    # print(f"train 前三个自由度的平均误差（原始尺度）: {first_three_error}")
    # print(f"train 后三个自由度的平均误差（原始尺度）: {last_three_error}")