import torch
import numpy as np
from utils import *
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
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

def test_model(model, test_loader, device, save_dir=None,visualize_type='23',norm_type='total'):
    model.eval()
    # 初始化每个时间步的误差列表
    window_size = test_loader.dataset[0][0].shape[0]  # 获取时间窗口大小
    first_three_errors = np.zeros((window_size, 3))  # 前三个自由度的误差
    last_three_errors = np.zeros((window_size, 3))   # 后三个自由度的误差
    time_step_losses = np.zeros(window_size)         # 每个时间步的loss
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for ini_datas, labels, folder_idx in tqdm(test_loader, desc='Testing', leave=True):
            # ini_datas=ini_datas.transpose(1,2)
            ini_datas = ini_datas.to(device)
            labels = labels.to(device)
            output = model(ini_datas)
            # print(output.shape)
            # 使用denormalize_batch函数进行反归一化
            denorm_output, denorm_label = denormalize_batch(output, labels, folder_idx, norm_type=norm_type)
            
            # 收集所有batch的输出
            all_outputs.append(denorm_output)
            all_labels.append(denorm_label)
            
            # 计算每个时间步的误差
            for t in range(window_size):
                # 计算前三个自由度（速度）的误差
                vel_error = torch.mean(torch.abs(denorm_output[:, t, :3] - denorm_label[:, t, :3]), dim=0)
                first_three_errors[t] += vel_error.cpu().numpy()
                
                # 计算后三个自由度（角速度）的误差
                ang_vel_error = torch.mean(torch.abs(denorm_output[:, t, 3:] - denorm_label[:, t, 3:]), dim=0)
                last_three_errors[t] += ang_vel_error.cpu().numpy()
                
                # 计算当前时间步的loss
                time_step_loss = torch.nn.functional.mse_loss(denorm_output[:, t], denorm_label[:, t])
                time_step_losses[t] += time_step_loss.item()
    
    # 计算平均值
    num_batches = len(test_loader)
    first_three_errors /= num_batches
    last_three_errors /= num_batches
    time_step_losses /= num_batches
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if save_dir is not None:
        if visualize_type=='23':
            visualize_predictions_23(all_outputs, all_labels, save_dir)
        elif visualize_type=='6':
            visualize_predictions_6(all_outputs, all_labels, save_dir)
    
    return first_three_errors, last_three_errors, time_step_losses

def test_one_folder(folder=0,save_dir=None):
    window = 40
    batch_size = 512
    
    device = set_device()

    # Create save directory
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load saved model weights
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    # checkpoint_path = os.path.join(save_dir, 'best_model_online.pth')

    # Prepare test data
    test_slide_dataset = prepare_data_with_folder(folder=0, window=40, return_split=False, return_norm_data=True, augmentation=False, norm_type='total')

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
    first_three_error, last_three_error, time_step_losses = test_model(model,  test_slide_loader, device, test_png_save_dir)

    # Print results
    print(f"Test first three DOF average error: {np.mean(first_three_error):.4f} m/s")
    print(f"Test last three DOF average error: {np.mean(last_three_error):.4f} rad/s ({np.mean(last_three_error)*180/np.pi:.4f} deg/s)")
    print(f"Test average MSE loss: {np.mean(time_step_losses):.6f}")

    # Save results to text file
    with open(test_txt_save_dir, 'w') as f:
        f.write(f"Test folder: {folder} with length {len(test_slide_dataset)}\n")  # Add folder number to the file
        f.write(f"First three DOF average error: {np.mean(first_three_error):.4f} m/s\n")
        f.write(f"Last three DOF average error: {np.mean(last_three_error):.4f} rad/s ({np.mean(last_three_error)*180/np.pi:.4f} deg/s)\n")
        f.write(f"Average MSE loss: {np.mean(time_step_losses):.6f}\n")

def test_all_folders(folders=[],name='',save_dir=None,window=200,batch_size=1024,return_norm_data=True,norm_type='total',visualize_type='23'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_path = '/home/pika/koopman-data/data/flights'
    device = set_device()
    test_save_folder = os.path.join(save_dir, 'predictions_'+name)
    if not os.path.exists(test_save_folder):
        os.makedirs(test_save_folder)
    # 创建误差可视化目录
    error_vis_dir = os.path.join(save_dir, 'error_visualization_'+name)
    if not os.path.exists(error_vis_dir):
        os.makedirs(error_vis_dir)
    test_txt_save_dir = os.path.join(test_save_folder,'predictions_'+name+'.txt')
    f=open(test_txt_save_dir, 'w')
    # Load saved model weights
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    from DT_Former import DT_transformer
    from lstm_model import LSTMPredictor,create_model
    # model = DT_transformer()
    # model = create_model(
    # input_dim=10,
    # hidden_dim=128,
    # num_layers=2,
    # output_dim=6,
    # dropout=0.1
    # )

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
        test_slide_dataset = prepare_data_with_folder(folder=folder, window=window, return_split=False, return_norm_data=return_norm_data, augmentation=False, norm_type=norm_type)

        test_slide_loader = torch.utils.data.DataLoader(
            dataset=test_slide_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        if return_norm_data:
            first_three_error, last_three_error, time_step_losses = test_model(
                model, test_slide_loader, device, test_png_save_dir,norm_type=norm_type,visualize_type=visualize_type)
        else:
            first_three_error, last_three_error, time_step_losses = test_model(
                model, test_slide_loader, device, test_png_save_dir,visualize_type=visualize_type)

        # Print results
        print(f"Test first three DOF average error: {np.mean(first_three_error):.4f} m/s")
        print(f"Test last three DOF average error: {np.mean(last_three_error):.4f} rad/s ({np.mean(last_three_error)*180/np.pi:.4f} deg/s)")
        print(f"Test average MSE loss: {np.mean(time_step_losses):.6f}")

        # Save results to text file
        f.write(f"Test folder: {folder} with length {len(test_slide_dataset)}\n")  # Add folder number to the file
        f.write(f"First three DOF average error: {np.mean(first_three_error):.4f} m/s\n")
        f.write(f"Last three DOF average error: {np.mean(last_three_error):.4f} rad/s ({np.mean(last_three_error)*180/np.pi:.4f} deg/s)\n")
        f.write(f"Average MSE loss: {np.mean(time_step_losses):.6f}\n")
        f.write("\n\n")
    # Load saved model weights
    f.close()
    


    
    np.save(os.path.join(error_vis_dir, 'first_three_error.npy'), first_three_error)
    np.save(os.path.join(error_vis_dir, 'last_three_error.npy'), last_three_error)
    np.save(os.path.join(error_vis_dir, 'time_step_losses.npy'), time_step_losses)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用新的样式名称
    
    # 1. 前三个维度（速度）的误差图
    plt.figure(figsize=(12, 8))
    time_steps = range(window)
    plt.plot(time_steps, first_three_error[:, 0], label='vel_1', color='#2878B5', linewidth=2)
    plt.plot(time_steps, first_three_error[:, 1], label='vel_2', color='#C82423', linewidth=2)
    plt.plot(time_steps, first_three_error[:, 2], label='vel_3', color='#9AC9DB', linewidth=2)
    plt.xlabel('time step', fontsize=12, fontweight='bold')
    plt.ylabel('error (m/s)', fontsize=12, fontweight='bold')
    plt.title('first three DOF (velocity) error over time steps', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, 'velocity_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 前三个维度的平均误差图
    plt.figure(figsize=(12, 8))
    mean_vel_error = np.mean(first_three_error, axis=1)
    plt.plot(time_steps, mean_vel_error, color='#2878B5', linewidth=2)
    plt.xlabel('time step', fontsize=12, fontweight='bold')
    plt.ylabel('average error (m/s)', fontsize=12, fontweight='bold')
    plt.title('first three DOF (velocity) average error over time steps', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, 'mean_velocity_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 后三个维度（角速度）的误差图
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, last_three_error[:, 0], label='ang_vel_1', color='#2878B5', linewidth=2)
    plt.plot(time_steps, last_three_error[:, 1], label='ang_vel_2', color='#C82423', linewidth=2)
    plt.plot(time_steps, last_three_error[:, 2], label='ang_vel_3', color='#9AC9DB', linewidth=2)
    plt.xlabel('time step', fontsize=12, fontweight='bold')
    plt.ylabel('error (rad/s)', fontsize=12, fontweight='bold')
    plt.title('last three DOF (angular velocity) error over time steps', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, 'angular_velocity_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 后三个维度的平均误差图
    plt.figure(figsize=(12, 8))
    mean_ang_vel_error = np.mean(last_three_error, axis=1)
    plt.plot(time_steps, mean_ang_vel_error, color='#2878B5', linewidth=2)
    plt.xlabel('time step', fontsize=12, fontweight='bold')
    plt.ylabel('average error (rad/s)', fontsize=12, fontweight='bold')
    plt.title('last three DOF (angular velocity) average error over time steps', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, 'mean_angular_velocity_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 总loss随时间步的变化
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, time_step_losses, color='#2878B5', linewidth=2)
    plt.xlabel('time step', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('MSE Loss over time steps', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(error_vis_dir, 'time_step_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 设置参数
    save_dir = '/home/pika/koopman-data/data/LSTM_decoder_0603190_folder'
    train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
    test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
    online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')

    window=190
    norm_type='folder'
    test_all_folders(train_folders.tolist(),name='train',
                     save_dir=save_dir,window=window,batch_size=1024,norm_type=norm_type,
                     visualize_type='6')
    test_all_folders(test_folders.tolist(),name='test',
                     save_dir=save_dir,window=window,batch_size=1024,norm_type=norm_type,
                     visualize_type='6')
    test_all_folders(online_folders.tolist(),name='online',
                     save_dir=save_dir,window=window,batch_size=1024,norm_type=norm_type,
                     visualize_type='6')

