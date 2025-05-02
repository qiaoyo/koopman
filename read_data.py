import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # 读取npy文件
# 读取训练历史数据
    history_path =  r'C:\Users\Administrator\Desktop\koopman-data\data\test\training_history_0421.npy'
    history = np.load(history_path, allow_pickle=True).item()

    # 提取数据
    train_losses = history['train_losses']
    test_losses = history['test_losses']
    final_test_loss = history['final_test_loss']
    final_test_mae = history['final_test_mae']
    final_test_rmse = history['final_test_rmse']

    # 打印最终的评估指标
    print("Training Results Summary:")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test MAE: {final_test_mae:.4f}")
    print(f"Final test RMSE: {final_test_rmse:.4f}")

    # 计算并打印最佳epoch和对应的损失
    best_epoch = np.argmin(test_losses)
    best_test_loss = np.min(test_losses)
    print(f"\nBest Performance:")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best test loss: {best_test_loss:.4f}")

    # 计算训练过程中的一些统计信息
    print("\nTraining Process Statistics:")
    print(f"Average training loss: {np.mean(train_losses):.4f}")
    print(f"Average test loss: {np.mean(test_losses):.4f}")
    print(f"Training loss std: {np.std(train_losses):.4f}")
    print(f"Test loss std: {np.std(test_losses):.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    # 绘制训练损失
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    # 绘制测试损失
    plt.plot(epochs, test_losses, 'r-s', label='Test Loss', markersize=4)

    # 标注最佳点
    plt.plot(best_epoch + 1, best_test_loss, 'g*', markersize=15, label='Best Test Loss')

    plt.title('Training and Test Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 保存图像
    plt.savefig(r'C:\Users\Administrator\Desktop\koopman-data\data\test\loss_analysis_0421.jpg')
    plt.show()
    # 读取Excel文件
    # file_path =  r'C:\Users\Administrator\Desktop\koopman-data\data\train.xlsx'   # r对路径进行转义，windows需要
    # raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    # # print(raw_data)
    # raw_data=np.array(raw_data)
    # raw_train_data=raw_data[0:60000,:]
    

    # file_path = r'C:\Users\Administrator\Desktop\koopman-data\data\50-hour-test.xlsx'   # r对路径进行转义，windows需要
    # raw_data = pd.read_excel(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
    # raw_data=np.array(raw_data)
    # raw_test_data=raw_data[0:26000,:]

    # # 计算训练数据前4列的最大值和最小值
    # train_max_values = np.max(raw_train_data[:, 0:10], axis=0)
    # train_min_values = np.min(raw_train_data[:, 0:10], axis=0)
    # print("\n训练数据前4列的最大值和最小值：")
    # for i in range(10):
    #     print(f"第{i}列 - 最小值: {train_min_values[i]:.4f}, 最大值: {train_max_values[i]:.4f}")
    
    # # 计算测试数据前4列的最大值和最小值
    # test_max_values = np.max(raw_test_data[:, 0:10], axis=0)
    # test_min_values = np.min(raw_test_data[:, 0:10], axis=0)
    # print("\n测试数据前4列的最大值和最小值：")
    # for i in range(10):
    #     print(f"第{i}列 - 最小值: {test_min_values[i]:.4f}, 最大值: {test_max_values[i]:.4f}")

