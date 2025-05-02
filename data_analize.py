import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_flight_data():
    """
    Traverse specific npy files in all subfolders under the flights folder and save statistics of max/min values for each sequence
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    # Initialize dictionary to store results
    stats = {}
    
    # Traverse folders 0-53
    for folder_idx in range(54):
        folder_path = os.path.join(base_path, str(folder_idx))
        if os.path.exists(folder_path):
            # Create sub-dictionary for each folder
            stats[str(folder_idx)] = {}
            
            # Traverse target files
            for file_name in target_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    key = file_name.split('.')[0]
                    
                    # Calculate max/min values for current sequence
                    stats[str(folder_idx)][key] = {
                        'max': np.max(data, axis=0),
                        'min': np.min(data, axis=0)
                    }
    
    # Print results
    print("\nData statistics:")
    for folder_idx in stats:
        print(f"\nFolder {folder_idx}:")
        for file_key in stats[folder_idx]:
            print(f"\n{file_key}:")
            for dim in range(len(stats[folder_idx][file_key]['max'])):
                print(f"Dimension {dim}:")
                print(f"  Max value: {stats[folder_idx][file_key]['max'][dim]:.4f}")
                print(f"  Min value: {stats[folder_idx][file_key]['min'][dim]:.4f}")
    
    # Save statistics
    save_path = os.path.join(base_path, 'flight_stats_by_sequence.npy')
    np.save(save_path, stats)
    print(f"\nStatistics saved to: {save_path}")
    
    return stats

def visualize_motors_cmd_range():
    """
    Visualize the distribution of max/min values of Motors_CMD first dimension across sequences 0-53
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    stats_path = os.path.join(base_path, 'flight_stats_by_sequence.npy')
    
    # Load statistical data
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # Extract Motors_CMD first dimension max/min values
    sequence_nums = []
    max_values = []
    min_values = []
    
    for folder_idx in range(54):
        if str(folder_idx) in stats:
            if 'Motors_CMD' in stats[str(folder_idx)]:
                sequence_nums.append(folder_idx)
                max_values.append(stats[str(folder_idx)]['Motors_CMD']['max'][0])
                min_values.append(stats[str(folder_idx)]['Motors_CMD']['min'][0])
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Draw max and min values
    plt.plot(sequence_nums, max_values, 'r-', label='最大值', marker='o')
    plt.plot(sequence_nums, min_values, 'b-', label='最小值', marker='o')
    
    # Fill max and min values between areas
    plt.fill_between(sequence_nums, max_values, min_values, alpha=0.2, color='gray')
    
    # Set plot properties
    plt.xlabel('folder idx')
    plt.ylabel('Motors_CMD first dim')
    plt.title('Motors_CMD range visualization')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    save_path = os.path.join(base_path, 'motors_cmd_range_visualization.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"可视化结果已保存至: {save_path}")

def visualize_all_dimensions():
    """
    Visualize the distribution of max/min values for all data types and dimensions, combining them into one large figure
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    stats_path = os.path.join(base_path, 'flight_stats_by_sequence.npy')
    
    # Load statistical data
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # Define data types and corresponding titles
    data_types = {
        'Motors_CMD': 'Motors Command',
        'Pos': 'Position',
        'Euler': 'Euler Angles'
    }
    
    # Create a large figure containing all subplots
    plt.figure(figsize=(20, 15))
    
    # Calculate total number of subplots needed
    total_subplots = sum(len(next(iter(stats.values()))[dt]['max']) for dt in data_types)
    
    # Calculate number of rows and columns for subplots
    n_rows = 4  # Set 4 rows
    n_cols = 3  # Set 3 columns
    
    # Current subplot index
    subplot_idx = 1
    
    # Iterate through each data type
    for data_type, title in data_types.items():
        first_folder = next(iter(stats.values()))
        n_dims = len(first_folder[data_type]['max'])
        
        for dim in range(n_dims):
            sequence_nums = []
            max_values = []
            min_values = []
            
            for folder_idx in range(54):
                if str(folder_idx) in stats:
                    if data_type in stats[str(folder_idx)]:
                        sequence_nums.append(folder_idx)
                        max_values.append(stats[str(folder_idx)][data_type]['max'][dim])
                        min_values.append(stats[str(folder_idx)][data_type]['min'][dim])
            
            plt.subplot(n_rows, n_cols, subplot_idx)
            
            plt.plot(sequence_nums, max_values, 'r-', label='Max', marker='o', markersize=3)
            plt.plot(sequence_nums, min_values, 'b-', label='Min', marker='o', markersize=3)
            
            plt.fill_between(sequence_nums, max_values, min_values, alpha=0.2, color='gray')
            
            plt.xlabel('Folder Index', fontsize=8)
            plt.ylabel(f'{data_type} Dim {dim}', fontsize=8)
            plt.title(f'{title} Dimension {dim}', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=8)
            plt.tick_params(axis='both', which='major', labelsize=8)
            
            subplot_idx += 1
    
    plt.tight_layout()
    
    save_path = os.path.join(base_path, 'all_dimensions_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization results for all dimensions have been saved to: {save_path}")

def visualize_single_group(folder):
    """
    可视化单个指定文件夹的所有维度数据
    Args:
        folder: 文件夹编号
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    stats_path = os.path.join(base_path, 'flight_stats_by_sequence.npy')
    
    # 加载统计数据
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # 创建图形
    plt.figure(figsize=(15, 16))
    
    # 准备数据
    group = str(folder)
    
    # 绘制Motors_CMD的最大值和最小值
    plt.subplot(2, 1, 1)
    motors_max = stats[group]['Motors_CMD']['max']
    motors_min = stats[group]['Motors_CMD']['min']
    dims_motors = range(len(motors_max))
    
    plt.plot(dims_motors, motors_max, marker='o', linestyle='-', markersize=6, label='Motors_CMD Max')
    plt.plot(dims_motors, motors_min, marker='s', linestyle='--', markersize=6, label='Motors_CMD Min')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Dimension Index', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.title(f'Group {group} Motors_CMD Max/Min Values', fontsize=12)
    plt.legend(fontsize=8, loc='upper left')
    
    # 绘制Pos和Euler的最大值和最小值
    plt.subplot(2, 1, 2)
    pos_max = stats[group]['Pos']['max']
    pos_min = stats[group]['Pos']['min']
    euler_max = stats[group]['Euler']['max']
    euler_min = stats[group]['Euler']['min']
    
    dims_pos_euler = range(len(pos_max) + len(euler_max))
    
    plt.plot(range(len(pos_max)), pos_max, marker='o', linestyle='-', markersize=6, label='Pos Max')
    plt.plot(range(len(pos_max)), pos_min, marker='s', linestyle='--', markersize=6, label='Pos Min')
    plt.plot(range(len(pos_max), len(pos_max) + len(euler_max)), euler_max, marker='o', linestyle='-', markersize=6, label='Euler Max')
    plt.plot(range(len(pos_max), len(pos_max) + len(euler_max)), euler_min, marker='s', linestyle='--', markersize=6, label='Euler Min')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Dimension Index', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.title(f'Group {group} Pos/Euler Max/Min Values', fontsize=12)
    plt.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(base_path, f'group_{folder}_split_lines.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已将第 {folder} 组的数据可视化保存至: {save_path}")

def visualize_two_groups(folder1, folder2):
    """
    同时可视化两个文件夹的数据，将同种数据放在同一个子图中
    Args:
        folder1: 第一个文件夹编号
        folder2: 第二个文件夹编号
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    stats_path = os.path.join(base_path, 'flight_stats_by_sequence.npy')
    
    # 加载统计数据
    stats = np.load(stats_path, allow_pickle=True).item()
    
    # 创建图形
    plt.figure(figsize=(15, 16))
    
    # 准备数据
    group1 = str(folder1)
    group2 = str(folder2)
    
    # 定义两个组的颜色
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    
    # 绘制Motors_CMD的最大值和最小值
    plt.subplot(2, 1, 1)
    motors_max1 = stats[group1]['Motors_CMD']['max']
    motors_min1 = stats[group1]['Motors_CMD']['min']
    motors_max2 = stats[group2]['Motors_CMD']['max']
    motors_min2 = stats[group2]['Motors_CMD']['min']
    dims_motors = range(len(motors_max1))
    
    # 绘制第一组Motors_CMD数据
    plt.plot(dims_motors, motors_max1, color=colors[0], linestyle='-', marker='o', 
             label=f'Group {folder1} Motors_CMD')
    plt.plot(dims_motors, motors_min1, color=colors[0], linestyle='--', marker='s', markersize=6)
    plt.fill_between(dims_motors, motors_max1, motors_min1, alpha=0.2, color=colors[0])
    
    # 绘制第二组Motors_CMD数据
    plt.plot(dims_motors, motors_max2, color=colors[1], linestyle='-', marker='o',
             label=f'Group {folder2} Motors_CMD')
    plt.plot(dims_motors, motors_min2, color=colors[1], linestyle='--', marker='s', markersize=6)
    plt.fill_between(dims_motors, motors_max2, motors_min2, alpha=0.2, color=colors[1])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Dimension Index', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.title(f'Motors_CMD Max/Min Values Comparison', fontsize=12)
    plt.legend(fontsize=8, loc='upper left')
    
    # 绘制Pos和Euler的最大值和最小值
    plt.subplot(2, 1, 2)
    pos_max1 = stats[group1]['Pos']['max']
    pos_min1 = stats[group1]['Pos']['min']
    euler_max1 = stats[group1]['Euler']['max']
    euler_min1 = stats[group1]['Euler']['min']
    
    pos_max2 = stats[group2]['Pos']['max']
    pos_min2 = stats[group2]['Pos']['min']
    euler_max2 = stats[group2]['Euler']['max']
    euler_min2 = stats[group2]['Euler']['min']
    
    # 绘制第一组Pos数据
    plt.plot(range(len(pos_max1)), pos_max1, color=colors[0], linestyle='-', marker='o', 
             markersize=6, label=f'Group {folder1} Pos')
    plt.plot(range(len(pos_min1)), pos_min1, color=colors[0], linestyle='--', marker='s', markersize=6)
    plt.fill_between(range(len(pos_max1)), pos_max1, pos_min1, alpha=0.2, color=colors[0])
    
    # 绘制第二组Pos数据
    plt.plot(range(len(pos_max2)), pos_max2, color=colors[1], linestyle='-', marker='o',
             markersize=6, label=f'Group {folder2} Pos')
    plt.plot(range(len(pos_min2)), pos_min2, color=colors[1], linestyle='--', marker='s', markersize=6)
    plt.fill_between(range(len(pos_max2)), pos_max2, pos_min2, alpha=0.2, color=colors[1])
    
    # 绘制第一组Euler数据
    offset = len(pos_max1)
    plt.plot(range(offset, offset + len(euler_max1)), euler_max1, color=colors[0], linestyle='-',
             marker='o', markersize=6, label=f'Group {folder1} Euler')
    plt.plot(range(offset, offset + len(euler_min1)), euler_min1, color=colors[0], linestyle='--',
             marker='s', markersize=6)
    plt.fill_between(range(offset, offset + len(euler_max1)), euler_max1, euler_min1, 
                    alpha=0.2, color=colors[0])
    
    # 绘制第二组Euler数据
    plt.plot(range(offset, offset + len(euler_max2)), euler_max2, color=colors[1], linestyle='-',
             marker='o', markersize=6, label=f'Group {folder2} Euler')
    plt.plot(range(offset, offset + len(euler_min2)), euler_min2, color=colors[1], linestyle='--',
             marker='s', markersize=6)
    plt.fill_between(range(offset, offset + len(euler_max2)), euler_max2, euler_min2,
                    alpha=0.2, color=colors[1])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Dimension Index', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.title(f'Pos/Euler Max/Min Values Comparison', fontsize=12)
    plt.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(base_path, f'groups_{folder1}_{folder2}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已将第 {folder1} 组和第 {folder2} 组的数据对比可视化保存至: {save_path}")

def visualize_time_series():
    """
    可视化每个文件夹中的时序数据，每个文件夹生成一张图，包含所有维度的子图
    每行显示3个子图
    """
    base_path = r"C:\Users\Administrator\Desktop\koopman-data\data\flights"
    # 创建新的保存目录
    save_dir = os.path.join(base_path, 'time_series_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    target_files = ['Motors_CMD.npy', 'Pos.npy', 'Euler.npy']
    
    # 遍历0-53文件夹
    for folder_idx in range(54):
        folder_path = os.path.join(base_path, str(folder_idx))
        if not os.path.exists(folder_path):
            continue
            
        # 读取数据
        length=0
        data_dict = {}
        total_dims = 0
        for file_name in target_files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                data = np.load(file_path)
                length=len(data)
                key = file_name.split('.')[0]
                data_dict[key] = data
                total_dims += data.shape[1]
        
        if not data_dict:
            continue
            
        # 计算需要的行数（每行3个子图）
        n_cols = 3
        n_rows = (total_dims + n_cols - 1) // n_cols
        
        # 创建子图
        fig = plt.figure(figsize=(15, 5 * n_rows))
        current_subplot = 1
        
        # 为每个数据类型创建子图
        for data_type, data in data_dict.items():
            for dim in range(data.shape[1]):
                plt.subplot(n_rows, n_cols, current_subplot)
                
                # 绘制时序数据
                time_steps = range(len(data))
                plt.plot(time_steps, data[:, dim], 'b-', linewidth=1)
                
                # 设置子图属性
                plt.title(f'{data_type} Dimension {dim}', fontsize=10)
                plt.xlabel('Time Steps', fontsize=8)
                plt.ylabel('Values', fontsize=8)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tick_params(axis='both', which='major', labelsize=8)
                
                current_subplot += 1
        
        plt.tight_layout()
        
        # 保存图形到新目录
        save_path = os.path.join(save_dir, f'time_series_group_{folder_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已将第 {folder_idx} 组的 len: {length}时序数据可视化保存至: {save_path}")
if __name__ == "__main__":
    visualize_time_series()
