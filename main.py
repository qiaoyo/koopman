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

if __name__=="__main__":
    seed=1023
    window=80
    batch_size=1024
    lr = 1e-3
    num_epochs = 100
    
    set_device()
    set_seed(seed=seed)

    train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
    test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
    online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')

    train_slide_dataset=prepare_merged_data(folders=train_folders.tolist(),window=window,return_split=False,return_norm_data=True,augmentation=False)
    test_slide_dataset=prepare_merged_data(folders=test_folders.tolist(),window=window,return_split=False,return_norm_data=True,augmentation=False)

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

    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    
    model = model.cuda()
    
    # 加载已保存的模型
    save_dir = '/home/pika/koopman-data/data/LSTM_decoder_0512'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'best_model_0512.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型成功，从第{checkpoint['epoch'] + 1}轮继续训练")
        # print(f"已加载模型的测试损失: {checkpoint['test_loss']:.4f}")
        # print(f"已加载模型的MAE: {checkpoint['test_mae']:.4f}")
        # print(f"已加载模型的RMSE: {checkpoint['test_rmse']:.4f}")
        # best_loss = checkpoint['test_loss']  # 更新最佳损失值
    else:
        print("未找到已保存的模型，将从头开始训练")
        best_loss = float('inf')

    # 添加L2正则化
    weight_decay = 1e-5  # L2正则化系数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 如果存在，加载优化器状态
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_function = torch.nn.MSELoss(reduction="mean")

    best_loss = float('inf')

    train_losses = []
    test_losses = []
    save_freq = 50
    
    # 早停参数
    patience = 10  # 容忍测试集性能不提升的轮数
    patience_counter = 0  # 计数器

    train_dimension_errors = [[] for _ in range(6)]  # 存储训练集6个维度的误差
        
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        from tqdm import tqdm
        pbar = tqdm(train_slide_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (ini_datas, labels) in enumerate(pbar):
            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            prediction = model(ini_datas) 
            loss = loss_function(prediction[:,-1,:], labels[:,-1,:])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss+=loss.item()*ini_datas.size(0)
            
            if i % save_freq == 0:
                # train_losses.append(loss.item())  # 保存训练损失
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_slide_loader.dataset)
        train_losses.append(avg_loss)  # 保存训练损失
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        
        # 验证阶段
        model.eval()
        test_loss = 0
        test_mae = []
        test_rmse = []
        # dimension_errors = [[] for _ in range(6)]  # 存储6个维度的误差
        
        with torch.no_grad():
            for ini_datas, labels in test_slide_loader:
                ini_datas = ini_datas.cuda()
                labels = labels.cuda()
                output = model(ini_datas)
                
                loss = loss_function(output[:,-1,:], labels[:,-1,:])
                mae = torch.mean(torch.abs(output[:,-1,:] - labels[:,-1,:]))
                rmse = torch.sqrt(torch.mean((output[:,-1,:] - labels[:,-1,:]) ** 2))
                
                test_loss+=loss.item()*ini_datas.size(0)
                test_mae.append(mae.item())
                test_rmse.append(rmse.item())
                
                # 计算每个维度的误差
                # for dim in range(6):
                #     dim_error = torch.mean(torch.abs(output[..., dim] - labels[..., dim]))
                #     dimension_errors[dim].append(dim_error.item())
        
        avg_test_loss = test_loss / len(test_slide_loader.dataset)
        test_losses.append(avg_test_loss)
        avg_test_mae = np.mean(test_mae)
        avg_test_rmse = np.mean(test_rmse)
        
        # # 验证阶段前，添加训练集误差计算
        # model.eval()  # 临时设置为评估模式以计算训练误差
        # train_dimension_errors = [[] for _ in range(6)]  # 存储训练集6个维度的误差
        
        # with torch.no_grad():
        #     for ini_datas, labels in train_slide_loader:
        #         ini_datas = ini_datas.cuda()
        #         labels = labels.cuda()
        #         output = model(ini_datas)
                
        #         # 计算每个维度的误差
        #         for dim in range(6):
        #             dim_error = torch.mean(torch.abs(output[..., dim] - labels[..., dim]))
        #             train_dimension_errors[dim].append(dim_error.item())
        
        # 计算训练集每个维度的平均误差
        # train_avg_dimension_errors = [np.mean(errors) for errors in train_dimension_errors]
        # train_first_three_avg = np.mean(train_avg_dimension_errors[:3])
        # train_last_three_avg = np.mean(train_avg_dimension_errors[3:])
        
        # 验证阶段
        # 计算每个维度的平均误差
        # avg_dimension_errors = [np.mean(errors) for errors in dimension_errors]
        # # 计算前三个维度和后三个维度的平均误差
        # first_three_avg = np.mean(avg_dimension_errors[:3])
        # last_three_avg = np.mean(avg_dimension_errors[3:])
        
        print(f"Test Metrics - Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}")
        # print(f"前三个维度平均误差: {first_three_avg:.4f} m/s")
        # print(f"后三个维度平均误差: {last_three_avg:.4f} rad/s {last_three_avg*180/3.14 }deg/s")
        
        # 检查是否需要保存最佳模型和早停
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0  # 重置计数器
            
            # 保存最佳模型时的详细信息到txt文件
            info_path = os.path.join(save_dir, f'best_model_info_012.txt')
            with open(info_path, 'w') as f:
                f.write(f"最佳模型信息 (Epoch {epoch+1}):\n")
                f.write(f"训练损失: {avg_loss:.4f}\n")
                f.write(f"测试损失: {avg_test_loss:.4f}\n")
                f.write(f"测试MAE: {avg_test_mae:.4f}\n")
                f.write(f"测试RMSE: {avg_test_rmse:.4f}\n")
                
                # 添加训练集误差信息
                # f.write("\n训练集每个维度的误差:\n")
                # for dim, error in enumerate(train_avg_dimension_errors):
                #     if dim < 3:
                #         f.write(f"维度 {dim+1} (速度): {error:.4f} m/s\n")
                #     else:
                #         f.write(f"维度 {dim+1} (角速度): {error:.4f} rad/s ({error*180/3.14:.4f} deg/s)\n")
                # f.write(f"\n训练集前三个维度平均误差(速度): {train_first_three_avg:.4f} m/s\n")
                # f.write(f"训练集后三个维度平均误差(角速度): {train_last_three_avg:.4f} rad/s ({train_last_three_avg*180/3.14:.4f} deg/s)\n")
                
                # 添加测试集误差信息
                # f.write("\n测试集每个维度的误差:\n")
                # for dim, error in enumerate(avg_dimension_errors):
                #     if dim < 3:
                #         f.write(f"维度 {dim+1} (速度): {error:.4f} m/s\n")
                #     else:
                #         f.write(f"维度 {dim+1} (角速度): {error:.4f} rad/s ({error*180/3.14:.4f} deg/s)\n")
                # f.write(f"\n测试集前三个维度平均误差(速度): {first_three_avg:.4f} m/s\n")
                # f.write(f"测试集后三个维度平均误差(角速度): {last_three_avg:.4f} rad/s ({last_three_avg*180/3.14:.4f} deg/s)\n")
            
            # 保存模型和相关参数
            model_path = os.path.join(save_dir, f'best_model_0512.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'test_mae': avg_test_mae,
                'test_rmse': avg_test_rmse
                # 'dimension_errors': avg_dimension_errors,
                # 'first_three_avg': first_three_avg,
                # 'last_three_avg': last_three_avg
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
        'final_test_loss': avg_test_loss,
        'final_test_mae': avg_test_mae,
        'final_test_rmse': avg_test_rmse
    }
    np.save(os.path.join(save_dir, 'training_history_0512.npy'), history)
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    epochs = range(1, epoch + 2)
    
    # 绘制训练损失，使用蓝色线条和圆形标记
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', 
             linestyle='-', markersize=8, markerfacecolor='white')
    
    # 绘制测试损失，使用红色线条和方形标记
    plt.plot(epochs, test_losses, label='Test Loss', color='red', marker='s', 
             linestyle='-', markersize=8, markerfacecolor='white')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴刻度为整数
    plt.xticks(epochs)
    
    plt.savefig(os.path.join(save_dir, 'loss_curves_0512.png'))
    plt.show()
    plt.close()  # 关闭图像

    