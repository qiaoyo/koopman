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
    lr = 1e-4
    num_epochs = 50
    
    device=set_device()
    set_seed(seed=seed)

    test_slide_dataset=prepare_merged_data(folders=[31],window=window,return_split=False,return_norm_data=True)

    
    test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
                               batch_size=batch_size,
                               shuffle=False,  
                               drop_last=False,
                               pin_memory=True)
    base_path = '/home/pika/koopman-data/data/flights'
    folder=31
    norm_data = np.load(os.path.join(base_path, str(folder), 'folder_stats.npy'), allow_pickle=True).item()

    from LSTM_decoder import MultiScaleTimeSeriesModel
    model = MultiScaleTimeSeriesModel(input_dim=10, output_dim=6)
    # from sample import MultiScaleTimeSeriesModel
    # model = MultiScaleTimeSeriesModel(input_size=10, output_dim=6)
    model = model.cuda()
    
    # 加载已保存的模型
    
    save_dir = '/home/pika/koopman-data/data/LSTM_decoder_0503_3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'best_model_0503.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,weights_only=False)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 如果存在，加载优化器状态
    if os.path.exists(checkpoint_path):
        model.load_state_dict(checkpoint['model_state_dict'])
    loss_function = torch.nn.MSELoss(reduction="mean")

    best_loss = float('inf')

    train_losses = []

    save_freq = 5
    
    # 早停参数
    patience = 20  # 容忍测试集性能不提升的轮数
    patience_counter = 0  # 计数器
    
    from test_model import test_model
    
    print(f"device:{device}")
    first_three_errors, last_three_errors, avg_loss=test_model(model,test_slide_loader,device,save_dir=None,norm_data=norm_data)

    print(f"first_three_errors:{first_three_errors},last_three_errors:{last_three_errors},avg_loss:{avg_loss}")
    best_loss=avg_loss
    train_losses.append(avg_loss)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        from tqdm import tqdm
        pbar = tqdm(test_slide_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (ini_datas, labels) in enumerate(pbar):
            model.train()
            ini_datas = ini_datas.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            prediction = model(ini_datas)
            
            loss = loss_function(prediction, labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            
            if i % save_freq == 0:
                # train_losses.append(loss.item())  # 保存训练损失
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_loss)
        # train_losses.append(avg_loss)  # 保存训练损失
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        
        first_three_errors, last_three_errors, av_loss=test_model(model,test_slide_loader,device,save_dir=None,norm_data=norm_data)
        train_losses.append(av_loss)
        if av_loss<best_loss:
            best_loss=av_loss
            png_save_path=os.path.join(save_dir,'test.png')
            first_three_errors, last_three_errors, _=test_model(model,test_slide_loader,device,save_dir=png_save_path,norm_data=norm_data)

            patience_counter = 0  # 重置计数器

                # 添加测试集误差信息
                
            
            # 保存模型和相关参数
            model_path = os.path.join(save_dir, f'best_model_0503_conti_ffn.pth')
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
        'train_losses': train_losses
        
        
    }
    np.save(os.path.join(save_dir, 'training_history_0430_CONT_ffn.npy'), history)
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    epochs = range(0, epoch + 2)
    
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

    