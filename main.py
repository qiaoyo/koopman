import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.nn.utils import weight_norm
from scipy.stats import pearsonr
from scipy import spatial
import time
from utils import *
from DT_Former import DT_transformer
from tqdm import tqdm
import wandb
from config import sweep_config, wandb_config
from torch.utils.data import DataLoader, TensorDataset
from DronePose import DronePosePredictor, CombinedLoss
from multiprocessing import Process

def train_model(config=None):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化wandb，增加超时时间
    with wandb.init(config=config, settings=wandb.Settings(init_timeout=300)) as run:
        # 获取当前配置
        config = run.config
        
        # 设置随机种子
        set_seed(42)

        # 加载数据
        train_folders = np.load('/home/pika/koopman-data/data/processed/train_folders.npy')
        test_folders = np.load('/home/pika/koopman-data/data/processed/test_folders.npy')
        online_folders = np.load('/home/pika/koopman-data/data/processed/online_folders.npy')
        train_slide_dataset = prepare_merged_data(
            folders=train_folders.tolist(),
            window=config.window,
            return_split=False,
            return_norm_data=config.require_norm_data,
            augmentation=True,
            norm_type=config.norm_type
        )
        test_slide_dataset = prepare_merged_data(
            folders=test_folders.tolist(),
            window=config.window,
            return_split=False,
            return_norm_data=config.require_norm_data,
            augmentation=True,
            norm_type=config.norm_type
        )

        train_loader = DataLoader(train_slide_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(test_slide_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
        
        # 初始化模型
        model = DronePosePredictor(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            gru_hidden_size=config.gru_hidden_size,
            gru_layers=config.gru_layers,
            tcn_channels=config.tcn_channels,
            tcn_layers=config.tcn_layers,
            dropout=config.dropout
        ).to(device)

        # 定义损失函数和器优化
        loss_function = CombinedLoss(alpha=config.alpha).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
            for input_data, labels, folder_idx in pbar:
                optimizer.zero_grad()
                input_data = input_data.to(device)
                labels = labels.to(device)
                output_data = model(input_data)
                
                denorm_output, denorm_label = denormalize_batch(output_data, labels, folder_idx, config.norm_type)
                denorm_output = denorm_output.to(device)
                denorm_label = denorm_label.to(device)
                loss = loss_function(denorm_output, denorm_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * input_data.size(0)
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for input_data, labels, folder_idx in val_loader:
                    input_data = input_data.to(device)
                    labels = labels.to(device)
                    output_data = model(input_data)
                    denorm_output, denorm_label = denormalize_batch(output_data, labels, folder_idx, config.norm_type)
                    denorm_output = denorm_output.to(device)
                    denorm_label = denorm_label.to(device)
                    loss = loss_function(denorm_output, denorm_label)
                    val_loss += loss.item() * input_data.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset           )
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录指标
            run.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                run.save('best_model.pth')
            
            # 早停检查
            if epoch > 20 and val_loss > best_val_loss * 1.15:
                print(f"Early stopping at epoch {epoch}")
                break

def run_experiment(sweep_id):
    # 启动一个实验
    wandb.agent(sweep_id, function=train_model, count=1)

def main():
    # 初始化wandb sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'])
    
    # 并行运行多个实验
    # 总实验次数
    total_experiments = 52
    # 并行运行的实验数量
    parallel_experiments = 4
    
    # 分批运行实验
    for i in range(0, total_experiments, parallel_experiments):
        processes = []
        for _ in range(parallel_experiments):
            if i + _ < total_experiments:  # 确保不会超出总实验次数
                p = Process(target=run_experiment, args=(sweep_id,))
                p.start()
                processes.append(p)
        
        # 等待当前批次的所有实验完成
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()