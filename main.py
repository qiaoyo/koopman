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
    batch_size=512
    lr = 6e-5
    num_epochs = 30
    
    set_device()
    set_seed(seed=seed)

    train_slide_dataset,train_split_dataset,test_slide_dataset,test_split_dataset=prepare_full_data(window=window)
    # train_slide_dataset,test_slide_dataset=prepare_data(window=window)
    # train_slide_loader = torch.utils.data.DataLoader(dataset=train_slide_dataset,
    #                        batch_size=batch_size,
    #                        shuffle=True, 
    #                        drop_last=True,  
    #                        pin_memory=True)  
    # test_slide_loader = torch.utils.data.DataLoader(dataset=test_slide_dataset,
    #                        batch_size=batch_size,
    #                        shuffle=True, 
    #                        drop_last=True,  
    #                        pin_memory=True)  

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
    from sample import MultiScaleTimeSeriesModel
    model=MultiScaleTimeSeriesModel(input_size=10,output_dim=6)
    model=model.cuda()

    # 添加L2正则化
    weight_decay = 1e-4  # L2正则化系数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss(reduction="mean")

    best_loss = float('inf')
    save_dir = r'C:\Users\Administrator\Desktop\koopman-data\data\test'
    train_losses = []
    test_losses = []
    save_freq = 50
    
    # 早停参数
    patience = 5  # 容忍测试集性能不提升的轮数
    patience_counter = 0  # 计数器
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        from tqdm import tqdm
        pbar = tqdm(train_slide_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
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
        
        # 验证阶段
        model.eval()
        test_loss = []
        test_mae = []
        test_rmse = []
        
        with torch.no_grad():
            for ini_datas, labels in test_slide_loader:
                ini_datas = ini_datas.cuda()
                labels = labels.cuda()
                output = model(ini_datas)
                
                loss = loss_function(output, labels)
                mae = torch.mean(torch.abs(output - labels))
                rmse = torch.sqrt(torch.mean((output - labels) ** 2))
                
                test_loss.append(loss.item())
                test_mae.append(mae.item())
                test_rmse.append(rmse.item())
        
        avg_test_loss = np.mean(test_loss)
        test_losses.append(avg_test_loss)
        avg_test_mae = np.mean(test_mae)
        avg_test_rmse = np.mean(test_rmse)
        
        print(f"Test Metrics - Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}, RMSE: {avg_test_rmse:.4f}")
        
        # 检查是否需要保存最佳模型和早停
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0  # 重置计数器
            model_path = os.path.join(save_dir, f'best_model_0421.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': avg_test_loss,
                'test_mae': avg_test_mae,
                'test_rmse': avg_test_rmse
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
    np.save(os.path.join(save_dir, 'training_history_0421.npy'), history)
    
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
    
    plt.savefig(os.path.join(save_dir, 'loss_curves_0421.png'))
    plt.show()
    plt.close()  # 关闭图像

    