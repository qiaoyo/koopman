import numpy as np
import os

def read_folder_lengths(base_path, start_folder=0, end_folder=53):
    """
    读取指定范围内每个文件夹的len.npy值
    
    Args:
        base_path: 基础路径
        start_folder: 起始文件夹编号
        end_folder: 结束文件夹编号
    """
    # 创建输出文件
    output_file = 'folder_lengths.txt'
    
    with open(output_file, 'w') as f:
        f.write("Folder Lengths:\n")
        f.write("-" * 20 + "\n")
        
        for folder in range(start_folder, end_folder + 1):
            len_file = os.path.join(base_path, str(folder), 'len.npy')
            if os.path.exists(len_file):
                length = np.load(len_file)
                f.write(f"Folder {folder:2d}: {length}\n")
            else:
                f.write(f"Folder {folder:2d}: File not found\n")
    
    print(f"Results have been saved to {output_file}")

if __name__ == "__main__":
    base_path = '/home/pika/koopman-data/data/flights'
    read_folder_lengths(base_path) 