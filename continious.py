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

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training_history(save_dir):
    """
    可视化训练历史数据
    
    参数:
        save_dir: 保存历史数据的目录路径
    """
    # 加载训练历史数据
    history_path = os.path.join(save_dir, 'training_history_0430_CONT.npy')
    history = np.load(history_path, allow_pickle=True).item()
    
    # 获取训练损失数据
    train_losses = history['train_losses']
    epochs = range(1, len(train_losses) + 1)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失曲线
    plt.plot(epochs, train_losses, label='训练损失', color='blue', 
             marker='o', linestyle='-', markersize=6, 
             markerfacecolor='white')
    
    # 设置图形属性
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss vs. epoch')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴刻度为整数
    plt.xticks(epochs)
    
    # 保存图形
    plt.savefig(os.path.join(save_dir, 'training_history_visualization.png'))
    plt.close()

# if __name__ == "__main__":
#     # 设置保存目录
#     save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\LSTM_decoder_2'
    
#     # 调用可视化函数
#     visualize_training_history(save_dir)
#     print("可视化完成，图片已保存")

if __name__=="__main__":
    seed=1023
    window=80
    batch_size=1024
    lr = 1e-3
    num_epochs = 50
    
    set_device()
    set_seed(seed=seed)

    train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset=prepare_full_data(window=window)

    train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
                           batch_size=batch_size,
                           shuffle=True, 
                           drop_last=True,  
                           pin_memory=True)  
    train_split_loader = torch.utils.data.DataLoader(dataset=train_split_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True)
    test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
                               batch_size=batch_size,
                               shuffle=False,  
                               drop_last=False,
                               pin_memory=True)
    test_split_loader = torch.utils.data.DataLoader(dataset=test_split_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    # from sample import MultiScaleTimeSeriesModel
    # model = MultiScaleTimeSeriesModel(input_size=10, output_dim=6)
    model = model.cuda()
    
    # 加载已保存的模型
    save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\LSTM_decoder'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'best_model_0421.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型成功，从第{checkpoint['epoch'] + 1}轮继续训练")
        print(f"已加载模型的测试损失: {checkpoint['test_loss']:.4f}")
        print(f"已加载模型的MAE: {checkpoint['test_mae']:.4f}")
        print(f"已加载模型的RMSE: {checkpoint['test_rmse']:.4f}")
        best_loss = checkpoint['test_loss']  # 更新最佳损失值
    else:
        print("未找到已保存的模型，将从头开始训练")
        best_loss = float('inf')

    # 添加L2正则化
    weight_decay = 1e-4  # L2正则化系数
    optimizer = torch.optim.Adam(model.ffn.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 如果存在，加载优化器状态
    if os.path.exists(checkpoint_path):
        model.load_state_dict(checkpoint['model_state_dict'])
    loss_function = torch.nn.MSELoss(reduction="mean")

    best_loss = float('inf')

    train_losses = []
    test_losses = []
    save_freq = 5
    
    # 早停参数
    patience = 10  # 容忍测试集性能不提升的轮数
    patience_counter = 0  # 计数器
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        from tqdm import tqdm
        pbar = tqdm(test_slide_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (ini_datas, labels) in enumerate(pbar):
            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            prediction = model(ini_datas) 
            loss = loss_function(prediction, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            if i % save_freq == 0:
                # train_losses.append(loss.item())  # 保存训练损失
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_loss)
        train_losses.append(avg_loss)  # 保存训练损失
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        

        # 验证阶段前，添加训练集误差计算
        model.eval()  # 临时设置为评估模式以计算训练误差
        train_dimension_errors = [[] for _ in range(6)]  # 存储训练集6个维度的误差
        
        with torch.no_grad():
            for ini_datas, labels in train_slide_loader:
                ini_datas = ini_datas.cuda()
                labels = labels.cuda()
                output = model(ini_datas)
                
                # 计算每个维度的误差
                for dim in range(6):
                    dim_error = torch.mean(torch.abs(output[..., dim] - labels[..., dim]))
                    train_dimension_errors[dim].append(dim_error.item())
        
        # 计算训练集每个维度的平均误差
        train_avg_dimension_errors = [np.mean(errors) for errors in train_dimension_errors]
        train_first_three_avg = np.mean(train_avg_dimension_errors[:3])
        train_last_three_avg = np.mean(train_avg_dimension_errors[3:])
        
        # 验证阶段
        # 计算每个维度的平均误差
       
        
        # 检查是否需要保存最佳模型和早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # 重置计数器
            
            # 保存最佳模型时的详细信息到txt文件
            info_path = os.path.join(save_dir, f'best_model_info_0430_CONTI_ffn.txt')
            with open(info_path, 'w') as f:
                f.write(f"最佳模型信息 (Epoch {epoch+1}):\n")
                f.write(f"训练损失: {avg_loss:.4f}\n")
                
                # 添加训练集误差信息
                f.write("\n训练集每个维度的误差:\n")
                for dim, error in enumerate(train_avg_dimension_errors):
                    if dim < 3:
                        f.write(f"维度 {dim+1} (速度): {error:.4f} m/s\n")
                    else:
                        f.write(f"维度 {dim+1} (角速度): {error:.4f} rad/s ({error*180/3.14:.4f} deg/s)\n")
                f.write(f"\n训练集前三个维度平均误差(速度): {train_first_three_avg:.4f} m/s\n")
                f.write(f"训练集后三个维度平均误差(角速度): {train_last_three_avg:.4f} rad/s ({train_last_three_avg*180/3.14:.4f} deg/s)\n")
                
                # 添加测试集误差信息
                
            
            # 保存模型和相关参数
            model_path = os.path.join(save_dir, f'best_model_0430_conti_ffn.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss
                
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
        
    }
    np.save(os.path.join(save_dir, 'training_history_0430_CONT_ffn.npy'), history)
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    epochs = range(1, epoch + 2)
    
    # 绘制训练损失，使用蓝色线条和圆形标记
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', 
             linestyle='-', markersize=8, markerfacecolor='white')
    
    # 绘制测试损失，使用红色线条和方形标记
    # plt.plot(epochs, test_losses, label='Test Loss', color='red', marker='s', 
    #          linestyle='-', markersize=8, markerfacecolor='white')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴刻度为整数
    plt.xticks(epochs)
    
    plt.savefig(os.path.join(save_dir, 'loss_curves_0430_conti_ffn.png'))
    plt.show()
    plt.close()  # 关闭图像

    