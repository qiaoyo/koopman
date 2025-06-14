import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.nn.utils import weight_norm
from scipy.stats import pearsonr
from scipy import spatial
import time
import os
from utils import *
from DT_Former import DT_transformer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import wandb

if __name__=="__main__":
    wandb.login()
    
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        }
    }
    sweep_config['parameters'] = {}

# 固定不变的超参
    sweep_config['parameters'].update({
    'project_name':{'value':'pelican'},
    'epochs': {'value': 50},
    'ckpt_path': {'value':'best_model.pth'}})

# 离散型分布超参
    sweep_config['parameters'].update({
    'optim_type': {
        'values': ['Adam', 'SGD','AdamW']
        },
    'hidden_layer_width': {
        'values': [16,32,48,64,80,96,112,128]
        }
    })

# 连续型分布超参
    sweep_config['parameters'].update({
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
      },
    
    'batch_size': {
        'distribution': 'q_uniform',
        'q': 8,
        'min': 32,
        'max': 256,
      },
    
    'dropout_p': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.6,
      }
    })
    sweep_config['early_terminate'] = {
    'type':'hyperband',
    'min_iter':3,
    'eta':2,
    's':3
    } 

    sweep_id = wandb.sweep(sweep_config, project='pelican')


    seed=1023
    batch_size=1024
    lr = 1e-3
    num_epochs = 100
    window=40
    norm_type='total'
    require_norm_data=True
    train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
    test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
    online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')
    save_dir = '/home/pika/koopman-data/data/LSTM_decoder_0603_'+str(window)+'_'+norm_type+'_LSTMdecoder_2_1'
    # save_dir = '/home/pika/koopman-data/data/LSTM_decoder_0603_44'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    set_device()
    set_seed(seed=seed)


    train_slide_dataset=prepare_merged_data(folders=train_folders.tolist(),window=window,return_split=False,return_norm_data=require_norm_data,augmentation=True,norm_type=norm_type)
    test_slide_dataset=prepare_merged_data(folders=test_folders.tolist(),window=window,return_split=False,return_norm_data=require_norm_data,augmentation=True,norm_type=norm_type)
    # train_slide_dataset=prepare_merged_data(folders=[44],window=window,return_split=False,return_norm_data=require_norm_data,augmentation=True,norm_type=norm_type)
    # test_slide_dataset=prepare_merged_data(folders=[44],window=window,return_split=False,return_norm_data=require_norm_data,augmentation=True,norm_type=norm_type)

    train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
                           batch_size=batch_size,
                           shuffle=True, 
                           drop_last=True,  
                           pin_memory=True)  
    test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True,
                           pin_memory=True)
    # 加载模型

    from LSTM_decoder_2 import MultiScaleTimeSeriesModel
    from DT_Former import DT_transformer
    from lstm_model import LSTMPredictor, create_model
    # from learned_inertial_model_odometry.src.learning.network.model_tcn import Tcn
    from tsai.models.TST import TST
    from TCN import create_model
# 示例：输入长度=50，变量数=10（4电机+6位姿），输出维度=6（预测位姿）
    from tsai.models.TST import TST
    from tsai.basics import *
    # model=MultiScaleTimeSeriesModel()
    from DronePose import DronePosePredictor,CombinedLoss
    # 初始化模型
    model = DronePosePredictor(
    input_dim=10, 
    output_dim=6,
    gru_hidden_size=128,
    gru_layers=2,
    tcn_channels=64,
    tcn_layers=4,
    dropout=0.1
)

# 训练配置建议
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    loss_function = CombinedLoss()  # 组合损失函数
    # model=create_model()
#     class TST_Seq2Seq(Module):
#         def __init__(self, c_in, c_out, seq_len, d_model=128, n_heads=8, dropout=0.1):
#             super().__init__()
#             self.tst = TST(c_in, c_out=d_model, seq_len=seq_len, 
#                           n_heads=n_heads, dropout=dropout)
#             self.projection = nn.Linear(d_model, c_out)
#             self.seq_len = seq_len

#         def forward(self, x):
#             # x shape: [bs, c_in, seq_len]
#             x = self.tst(x)  # [bs, d_model]
#             x = x.unsqueeze(-1).expand(-1, -1, self.seq_len)  # [bs, d_model, seq_len]
#             x = self.projection(x.transpose(1, 2))  # [bs, seq_len, c_out]
#             return x

# # 使用示例
#     model = TST_Seq2Seq(c_in=10, c_out=6, seq_len=window)  # dim_in=10, dim_out=6
    # model =TCNSeq2Seq(input_dim=10,output_dim=6,
    #                    num_channels=[64, 64, 64, 64, 128, 128, 128],
    #                    kernel_size=2,
    #                    dropout=0.2,
    #                    )

    # model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    # model = DT_transformer()
    # 创建模型
    # model = create_model(
    # input_dim=10,
    # hidden_dim=128,
    # num_layers=2,
    # output_dim=6,
    # dropout=0.1
    # )

    model = model.cuda()
    
    # 加载已保存的模型
    

    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型成功，从第{checkpoint['epoch'] + 1}轮继续训练")
    else:
        print("未找到已保存的模型，将从头开始训练")
        best_loss = float('inf')

    # 添加L2正则化
    weight_decay = 1e-4  # L2正则化系数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 如果存在，加载优化器状态
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # loss_function = torch.nn.MSELoss(reduction="mean")

    best_loss = float('inf')

    train_losses = []
    test_losses = []
    save_freq = 50
    
    # 早停参数
    patience = 10  # 容忍测试集性能不提升的轮数
    patience_counter = 0  # 计数器

    # 初始化训练历史记录
    train_vel_errors = []  # 训练集速度误差
    train_ang_vel_errors = []  # 训练集角速度误差
    test_vel_errors = []  # 测试集速度误差
    test_ang_vel_errors = []  # 测试集角速度误差
        
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_vel_errors = []  # 当前epoch的速度误差
        epoch_ang_vel_errors = []  # 当前epoch的角速度误差
        
        from tqdm import tqdm
        pbar = tqdm(train_slide_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (ini_datas, labels, folder_idx) in enumerate(pbar):

            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            optimizer.zero_grad() 
            output = model(ini_datas)

            # print(output.shape)
            # 使用denormalize_batch函数进行反归一化
            denorm_output, denorm_label = denormalize_batch(output, labels, folder_idx, norm_type)
            
            # 计算前三个自由度（速度）和后三个自由度（角速度）的误差
            vel_error = torch.mean(torch.abs(denorm_output[..., :3] - denorm_label[..., :3]))
            ang_vel_error = torch.mean(torch.abs(denorm_output[..., 3:] - denorm_label[..., 3:]))
            epoch_vel_errors.append(vel_error.item())
            epoch_ang_vel_errors.append(ang_vel_error.item())
            
            loss = loss_function(denorm_output, denorm_label)
            
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()*ini_datas.size(0)
            if i % save_freq == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'vel_err': f'{vel_error.item():.4f}',
                    'ang_vel_err': f'{ang_vel_error.item():.4f}'
                })

        avg_loss = epoch_loss / len(train_slide_loader.dataset)
        train_losses.append(avg_loss)  # 保存训练损失
        
        # 计算并保存当前epoch的平均误差
        avg_vel_error = np.mean(epoch_vel_errors)
        avg_ang_vel_error = np.mean(epoch_ang_vel_errors)
        train_vel_errors.append(avg_vel_error)
        train_ang_vel_errors.append(avg_ang_vel_error)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        print(f"训练集速度误差: {avg_vel_error:.4f} m/s")
        print(f"训练集角速度误差: {avg_ang_vel_error:.4f} rad/s ({avg_ang_vel_error*180/np.pi:.4f} deg/s)")
        
        # 验证阶段
        model.eval()
        test_loss = 0
        epoch_test_vel_errors = []  # 当前epoch的测试集速度误差
        epoch_test_ang_vel_errors = []  # 当前epoch的测试集角速度误差
        
        with torch.no_grad():
            for ini_datas, labels, folder_idx in test_slide_loader:
                
                ini_datas = ini_datas.cuda()
                labels = labels.cuda()
                output = model(ini_datas)

                # 使用denormalize_batch函数进行反归一化
                denorm_output, denorm_label = denormalize_batch(output, labels, folder_idx, norm_type)
                
                # 计算前三个自由度（速度）和后三个自由度（角速度）的误差
                vel_error = torch.mean(torch.abs(denorm_output[..., :3] - denorm_label[..., :3]))
                ang_vel_error = torch.mean(torch.abs(denorm_output[..., 3:] - denorm_label[..., 3:]))
                epoch_test_vel_errors.append(vel_error.item())
                epoch_test_ang_vel_errors.append(ang_vel_error.item())
                
                loss = loss_function(denorm_output, denorm_label)
                
                test_loss+=loss.item()*ini_datas.size(0)
                
        avg_test_loss = test_loss / len(test_slide_loader.dataset)
        test_losses.append(avg_test_loss)
        
        # 计算并保存当前epoch的测试集平均误差
        avg_test_vel_error = np.mean(epoch_test_vel_errors)
        avg_test_ang_vel_error = np.mean(epoch_test_ang_vel_errors)
        test_vel_errors.append(avg_test_vel_error)
        test_ang_vel_errors.append(avg_test_ang_vel_error)
        
        print(f"Test Metrics - Loss: {avg_test_loss:.4f}")
        print(f"测试集速度误差: {avg_test_vel_error:.4f} m/s")
        print(f"测试集角速度误差: {avg_test_ang_vel_error:.4f} rad/s ({avg_test_ang_vel_error*180/np.pi:.4f} deg/s)")
        
        # 检查是否需要保存最佳模型和早停
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0  # 重置计数器
            
            # 保存最佳模型时的详细信息到txt文件
            info_path = os.path.join(save_dir, f'best_model_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"最佳模型信息 (Epoch {epoch+1}):\n")
                f.write(f"训练损失: {avg_loss:.4f}\n")
                f.write(f"测试损失: {avg_test_loss:.4f}\n")
                f.write(f"\n训练集误差:\n")
                f.write(f"速度误差: {avg_vel_error:.4f} m/s\n")
                f.write(f"角速度误差: {avg_ang_vel_error:.4f} rad/s ({avg_ang_vel_error*180/np.pi:.4f} deg/s)\n")
                f.write(f"\n测试集误差:\n")
                f.write(f"速度误差: {avg_test_vel_error:.4f} m/s\n")
                f.write(f"角速度误差: {avg_test_ang_vel_error:.4f} rad/s ({avg_test_ang_vel_error*180/np.pi:.4f} deg/s)\n")
            
            # 保存模型和相关参数
            model_path = os.path.join(save_dir, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'train_vel_error': avg_vel_error,
                'train_ang_vel_error': avg_ang_vel_error,
                'test_vel_error': avg_test_vel_error,
                'test_ang_vel_error': avg_test_ang_vel_error
            }, model_path)
            print(f"Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_vel_errors': train_vel_errors,
        'train_ang_vel_errors': train_ang_vel_errors,
        'test_vel_errors': test_vel_errors,
        'test_ang_vel_errors': test_ang_vel_errors,
        'final_test_loss': avg_test_loss,
        'final_train_vel_error': avg_vel_error,
        'final_train_ang_vel_error': avg_ang_vel_error,
        'final_test_vel_error': avg_test_vel_error,
        'final_test_ang_vel_error': avg_test_ang_vel_error
    }
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.npy')
    np.save(history_path, history)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制损失曲线
    ax.plot(range(1, len(train_losses) + 1), train_losses, 
            label='Training Loss', color='#2878B5', linewidth=2, 
            marker='o', markersize=4, markevery=5)
    ax.plot(range(1, len(test_losses) + 1), test_losses, 
            label='Testing Loss', color='#C82423', linewidth=2,
            marker='s', markersize=4, markevery=5)
    
    # 设置坐标轴
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=20)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置图例
    ax.legend(loc='upper right', fontsize=10, frameon=True, 
             fancybox=True, shadow=True)
    
    # 设置背景
    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('white')
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像（高DPI以确保清晰度）
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    
