import torch
import numpy as np
from utils import *
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
    total_loss = 0.0
    num_batches = 0
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for ini_datas, labels in test_loader:
            ini_datas = ini_datas.to(device)
            labels = labels.to(device)
            output = model(ini_datas)
            
            # 计算MSE损失
            loss = torch.nn.functional.mse_loss(output, labels)
            total_loss += loss.item()
            num_batches += 1
            
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
        
    # 计算平均损失
    avg_loss = total_loss / num_batches
    
    # 计算每个自由度的平均误差
    first_three_errors = np.array([np.mean(dof_errors[i]) for i in range(3)])
    last_three_errors = np.array([np.mean(dof_errors[i]) for i in range(3, 6)])
    
    return first_three_errors, last_three_errors, avg_loss

def test_one_folder(folder=0,save_dir=None):
    window = 80
    batch_size = 512
    
    device = set_device()

    # Create save directory
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load saved model weights
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    # checkpoint_path = os.path.join(save_dir, 'best_model_online.pth')

    # Prepare test data
    test_slide_dataset = prepare_data_with_folder(folder=folder,window=window,     return_split=False)

    test_slide_loader = torch.utils.data.DataLoader(
        dataset=test_slide_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    # Load model
    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)

    model = model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print("Model file not found")
        exit()

    test_save_folder = os.path.join(save_dir, 'predictions_final_test_'+str(folder))
    if not os.path.exists(test_save_folder):
        os.makedirs(test_save_folder)

    test_png_save_dir = os.path.join(test_save_folder, 'predictions_visualization_final_test_'+str(folder)+'.png')
    test_txt_save_dir = os.path.join(test_save_folder,'predictions_final_test_'+str(folder)+'.txt')

    # Test model
    first_three_error, last_three_error, avg_loss = test_model(model,  test_slide_loader, device, test_png_save_dir)

    # Print results
    print(f"Test first three DOF errors (original scale):{first_three_error}   average errors: {np.mean(first_three_error):.4f}m/s")
    print(f"Test last three DOF errors (original scale):{last_three_error}     average errors: {np.mean(last_three_error):.4f}rad/s {np.mean  (last_three_error)*180/np.pi}deg/s")
    print(f"Test average MSE loss: {avg_loss:.6f}")

    # Save results to text file
    with open(test_txt_save_dir, 'w') as f:
        f.write(f"Test folder: {folder} with length {len(test_slide_dataset)}\n")  # Add folder number to the file
        f.write(f"First three DOF average errors: {first_three_error} average   errors: {np.mean(first_three_error):.4f} m/s\n")
        f.write(f"Last three DOF average errors: {last_three_error} average     errors: {np.mean(last_three_error):.4f} rad/s {np.mean(last_three_error)    *180/np.pi}deg/s\n")
        f.write(f"Average MSE loss: {avg_loss:.6f}\n")

def test_all_folders(save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    window = 80
    batch_size = 512
    device = set_device()
    test_save_folder = os.path.join(save_dir, 'predictions_final_test_full')
    if not os.path.exists(test_save_folder):
        os.makedirs(test_save_folder)

    test_txt_save_dir = os.path.join(test_save_folder,'predictions_final_test_.txt')
    f=open(test_txt_save_dir, 'w')
    # Load saved model weights
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)

    model = model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print("Model file not found")
        exit()    

    for folder in range(54):
        test_png_save_dir = os.path.join(test_save_folder, 'predictions_visualization_final_test_'+str(folder)+'.png')
        test_slide_dataset = prepare_data_with_folder(folder=folder,window=window,     return_split=False)

        test_slide_loader = torch.utils.data.DataLoader(
            dataset=test_slide_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        first_three_error, last_three_error, avg_loss = test_model(model,  test_slide_loader, device, test_png_save_dir)

    # Print results
        print(f"Test first three DOF errors (original scale):       {first_three_error}   average errors: {np.mean(first_three_error):.4f}m/s")
        print(f"Test last three DOF errors (original scale):{last_three_error}     average errors: {np.mean(last_three_error):.4f}rad/s {np.mean  (last_three_error)*180/np.pi}deg/s")
        print(f"Test average MSE loss: {avg_loss:.6f}")

    # Save results to text file
        f.write(f"Test folder: {folder} with length {len(test_slide_dataset)}\n")  # Add folder number to the file
        f.write(f"First three DOF average errors: {first_three_error} average   errors: {np.mean(first_three_error):.4f} m/s\n")
        f.write(f"Last three DOF average errors: {last_three_error} average     errors: {np.mean(last_three_error):.4f} rad/s {np.mean(last_three_error)    *180/np.pi}deg/s\n")
        f.write(f"Average MSE loss: {avg_loss:.6f}\n")
        f.write("\n\n")
    # Load saved model weights


    f.close()
        

if __name__ == "__main__":
    # 设置参数
    save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\LSTM_decoder_0502_2'

    test_all_folders(save_dir)