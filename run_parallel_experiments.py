import subprocess
import os
import time
from config import sweep_config, wandb_config

def run_experiment(exp_id, gpu_id):
    """运行单个实验"""
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 修改wandb配置
    exp_config = sweep_config.copy()
    exp_config['name'] = f'experiment_{exp_id}'
    
    # 根据实验ID修改一些参数
    if exp_id == 0:
        exp_config['parameters']['learning_rate']['min'] = -4  # 1e-4
        exp_config['parameters']['learning_rate']['max'] = -3  # 1e-3
    elif exp_id == 1:
        exp_config['parameters']['hidden_size']['values'] = [64, 128]
        exp_config['parameters']['batch_size']['values'] = [32, 64]
    elif exp_id == 2:
        exp_config['parameters']['dropout']['min'] = 0.2
        exp_config['parameters']['dropout']['max'] = 0.4
    elif exp_id == 3:
        exp_config['parameters']['position_weight']['min'] = 1.0
        exp_config['parameters']['position_weight']['max'] = 2.0
    
    # 创建实验目录
    exp_dir = f'experiment_{exp_id}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # 运行实验
    cmd = f'python main.py --exp_id {exp_id} --gpu_id {gpu_id}'
    process = subprocess.Popen(cmd, shell=True, env=env)
    return process

def main():
    # 检查可用的GPU数量
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
    except:
        print("No GPU found, using CPU")
        num_gpus = 0
    
    # 启动4个并行实验
    processes = []
    for i in range(4):
        gpu_id = i % max(1, num_gpus)  # 如果没有GPU，使用CPU
        print(f"Starting experiment {i} on {'GPU' if num_gpus > 0 else 'CPU'} {gpu_id}")
        process = run_experiment(i, gpu_id)
        processes.append(process)
        time.sleep(5)  # 等待5秒再启动下一个实验
    
    # 等待所有实验完成
    for i, process in enumerate(processes):
        process.wait()
        print(f"Experiment {i} completed")

if __name__ == "__main__":
    main() 