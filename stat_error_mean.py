import numpy as np

base_path = "/home/pika/koopman-data/data"
windows = [40, 60, 80, 100]
results = {}

for w in windows:
    folder = f"DronePose_{w}/error_visualization_test"
    first_path = f"{base_path}/{folder}/first_three_error.npy"
    last_path = f"{base_path}/{folder}/last_three_error.npy"
    loss_path = f"{base_path}/{folder}/time_step_losses.npy"
    try:
        first = np.load(first_path)
        last = np.load(last_path)
        losses = np.load(loss_path)
        # 后三个自由度误差从rad转为deg
        last_deg = last * 180 / np.pi
        results[w] = {
            "first_three_error_mean": np.mean(first),
            "last_three_error_mean_deg": np.mean(last_deg),
            "time_step_losses_mean": np.mean(losses)
        }
        print(f"窗口{w}:")
        print(f"  first_three_error均值: {np.mean(first):.6f}")
        print(f"  last_three_error均值: {np.mean(last_deg):.6f} (deg)")
        print(f"  time_step_losses均值: {np.mean(losses):.6f}")
    except Exception as e:
        print(f"窗口{w} 读取失败: {e}")

print("全部完成！")