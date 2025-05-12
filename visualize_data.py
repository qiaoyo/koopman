import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成数据
group1 = np.random.uniform(30000, 50000, 10)
group2 = group1 * 0.9 + np.random.normal(0, 1000, 10)
group3 = group1 * 0.5 + np.random.normal(0, 2000, 10)
group4 = group1 * 0.7 + np.random.normal(0, 1500, 10)
group5 = group1 * 0.9 + np.random.normal(0, 1200, 10)

# 创建图形
plt.figure(figsize=(15, 8))

# 设置颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# 为每个位置创建一组数据点
for i in range(10):
    values = [group1[i], group2[i], group3[i], group4[i], group5[i]]
    x = np.ones(5) * (i + 1)  # x坐标
    for j in range(5):
        plt.scatter(x[j], values[j], c=colors[j], marker=markers[j], s=100, label=f'Group {j+1}' if i == 0 else "")

# 添加连接线
for i in range(10):
    values = [group1[i], group2[i], group3[i], group4[i], group5[i]]
    x = np.ones(5) * (i + 1)
    plt.plot(x, values, 'k--', alpha=0.3)

# 设置图表属性
plt.title('数据可视化 - 10组数据的对比', fontsize=14)
plt.xlabel('数据组', fontsize=12)
plt.ylabel('数值', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'])

# 设置x轴刻度
plt.xticks(range(1, 11))

# 添加数值标签
for i in range(10):
    values = [group1[i], group2[i], group3[i], group4[i], group5[i]]
    x = np.ones(5) * (i + 1)
    for j, v in enumerate(values):
        plt.text(x[j], v, f'{v:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
plt.close()