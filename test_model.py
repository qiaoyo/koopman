import torch
import numpy as np
from utils import *
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf,precision=4,suppress=True)
torch.set_printoptions(threshold=float('inf'), precision=4, sci_mode=False)


def visualize_predictions_23(output, labels, save_dir):
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

def visualize_predictions_6(output, labels, save_dir):
    """可视化预测结果和真实值（6*1布局）"""
    # 只取最后一个时间步的数据
    pred = output[:, -1, :].cpu().numpy()  # 所有样本的最后一步
    true = labels[:, -1, :].cpu().numpy()  # 所有样本的最后一步
    samples = range(len(pred))
    
    # 创建6x1的子图布局
    fig, axes = plt.subplots(6, 1, figsize=(10, 20))
    fig.suptitle('comparison of prediction and label (final step)')
    
    # 绘制6个自由度的对比图
    for i in range(6):
        axes[i].plot(samples, true[:, i], 'b-', label='label')
        axes[i].plot(samples, pred[:, i], 'r--', label='prediction')
        axes[i].set_title(f'dof {i+1}')
        axes[i].set_xlabel('sample index')
        axes[i].set_ylabel('value')
        axes[i].legend()
        axes[i].grid(True)
        # 设置y轴格式为4位小数
        axes[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()

def test_model(model, test_loader, device, save_dir=None,norm_data=None,visualize_type='23'):
    model.eval()
    # 初始化6个自由度的误差列表
    dof_errors = [[] for _ in range(6)]
    total_loss = 0
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for ini_datas, labels in test_loader:
            ini_datas = ini_datas.to(device)
            labels = labels.to(device)
            output = model(ini_datas)
            labels=labels[:,-1,:]
            output=output[:,-1,:]
            labels=labels.unsqueeze(1)
            output=output.unsqueeze(1)

            output[:,:,0:3] = denormalize_data(output[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
            output[:,:,3:6] = denormalize_data(output[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])
            labels[:,:,0:3] = denormalize_data(labels[:,:,0:3], norm_data['Pos']['max'], norm_data['Pos']['min'])
            labels[:,:,3:6] = denormalize_data(labels[:,:,3:6], norm_data['Euler']['max'], norm_data['Euler']['min'])

            # 计算MSE损失
            loss = torch.nn.functional.mse_loss(output, labels)
            total_loss+=loss.item()*len(labels)
            
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
        if visualize_type=='23':
            visualize_predictions_23(all_outputs, all_labels, save_dir)
        elif visualize_type=='6':
            visualize_predictions_6(all_outputs, all_labels, save_dir)
    # 计算平均损失
    avg_loss = total_loss / len(test_loader.dataset)
    
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

def test_all_folders(folders=[],save_dir=None,visualize_type='23'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_path = '/home/pika/koopman-data/data/flights'
    window = 80
    batch_size = 512
    device = set_device()
    test_save_folder = os.path.join(save_dir, 'predictions_final_online_new')
    if not os.path.exists(test_save_folder):
        os.makedirs(test_save_folder)

    test_txt_save_dir = os.path.join(test_save_folder,'predictions_final_online.txt')
    f=open(test_txt_save_dir, 'w')
    # Load saved model weights
    checkpoint_path = os.path.join(save_dir, 'best_model_0512.pth')
    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)

    model = model.to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print("Model file not found")
        exit()    

    for folder in folders:
        test_png_save_dir = os.path.join(test_save_folder, 'predictions_visualization_final_test_'+str(folder)+'.png')
        test_slide_dataset = prepare_data_with_folder(folder=folder,window=window, return_split=False,return_norm_data=True)

        test_slide_loader = torch.utils.data.DataLoader(
            dataset=test_slide_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        norm_data = np.load(os.path.join(base_path, str(folder), 'folder_stats.npy'), allow_pickle=True).item()
        first_three_error, last_three_error, avg_loss = test_model(model, test_slide_loader, device, test_png_save_dir,norm_data,visualize_type=visualize_type)

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
    save_dir = r'/home/pika/koopman-data/data/LSTM_decoder_0512'
    train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
    test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
    online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')

    test_all_folders(online_folders.tolist(),save_dir,visualize_type='6')
