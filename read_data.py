import numpy as np
import pandas as pd
if __name__ == '__main__':
    file_path =  r'C:\Users\Administrator\Desktop\koopman-data\data\train.xlsx'   # r对路径进行转义，windows需要
    raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    # print(raw_data)
    raw_data=np.array(raw_data)
    raw_train_data=raw_data[0:60000,:]
    

    file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\50-hour-test.xlsx'   # r对路径进行转义，windows需要
    raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    raw_data=np.array(raw_data)
    raw_test_data=raw_data[0:26000,:]

    # 计算训练数据前4列的最大值和最小值
    train_max_values = np.max(raw_train_data[:, 0:4], axis=0)
    train_min_values = np.min(raw_train_data[:, 0:4], axis=0)
    print("\n训练数据前4列的最大值和最小值：")
    for i in range(4):
        print(f"第{i}列 - 最小值: {train_min_values[i]:.4f}, 最大值: {train_max_values[i]:.4f}")
    
    # 计算测试数据前4列的最大值和最小值
    test_max_values = np.max(raw_test_data[:, 0:4], axis=0)
    test_min_values = np.min(raw_test_data[:, 0:4], axis=0)
    print("\n测试数据前4列的最大值和最小值：")
    for i in range(4):
        print(f"第{i}列 - 最小值: {test_min_values[i]:.4f}, 最大值: {test_max_values[i]:.4f}")

    