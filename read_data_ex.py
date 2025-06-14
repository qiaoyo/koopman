import numpy as np
import os
np.set_printoptions(suppress=True,precision=3)
def read_npy_files(directory):
    # 获取目录下所有的.npy文件
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # 按文件名排序
    # ["Motors.npy","Vel.npy","pqr.npy"]
    npy_files.sort()
    
    for file_name in npy_files:
        if file_name!='folder_stats.npy' and file_name!='len.npy':
            print(f"\n文件名: {file_name}")
            file_path = os.path.join(directory, file_name)
        # 读取.npy文件
            data = np.load(file_path)
        
        
            print("前三列数据:")
            print(data[:4, :])  # 显示所有行的前三列

if __name__ == "__main__":
    directory = "/home/pika/koopman-data/data/flights/1"
    # read_npy_files(directory)


# 读取.npy文件
    folder_stats = np.load(os.path.join(directory,'folder_stats.npy'), allow_pickle=True).item()

# 打印字典内容
    print("\n文件夹统计信息：")
    print("-" * 50)
    for folder_name, stats in folder_stats.items():
        print(f"\n文件夹: {folder_name}")
        print("-" * 30)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")