import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成主要分布在0-0.2之间的数据
main_data = np.random.beta(2, 10, 9) * 0.2  # 使用beta分布生成9个0-0.2之间的数

# 生成1个0.4-0.7之间的数据
mid_data = np.random.uniform(0.4, 0.7, 1)

# 合并数据并打乱顺序
original_data = np.concatenate([main_data, mid_data])
np.random.shuffle(original_data)

# 生成第二组数据（小变动）
small_change_data = original_data.copy()
# 对小值添加小的随机变动
small_indices = np.where(small_change_data < 0.2)[0]
small_change_data[small_indices] += np.random.normal(0, 0.02, len(small_indices))
small_change_data[small_indices] = np.clip(small_change_data[small_indices], 0, 0.2)

# 生成第三组数据（突变）
mutation_data = original_data.copy()
# 找到最大的值
max_index = np.argmax(mutation_data)
# 将最大值突变到很小
mutation_data[max_index] = np.random.uniform(0.01, 0.05)

# 打印数据
print("\n原始数据：")
for i, num in enumerate(original_data, 1):
    print(f"第{i}个数: {num:.4f}")

print("\n小变动数据：")
for i, num in enumerate(small_change_data, 1):
    print(f"第{i}个数: {num:.4f}")

print("\n突变数据：")
for i, num in enumerate(mutation_data, 1):
    print(f"第{i}个数: {num:.4f}")

# 可视化数据分布
plt.figure(figsize=(15, 8))

# 绘制三组数据的叠加柱状图
x = np.arange(1, 11)
width = 0.8  # 增加柱子宽度

# 创建叠加柱状图
plt.bar(x, original_data, width, label='原始数据', alpha=0.3, color='blue')
plt.bar(x, small_change_data, width, label='小变动数据', alpha=0.3, color='red')
plt.bar(x, mutation_data, width, label='突变数据', alpha=0.3, color='green')

# 设置坐标轴标签和刻度
# plt.xlabel('序号', fontsize=16, weight='bold')
# plt.ylabel('数值', fontsize=16, weight='bold')
plt.xticks(range(1, 11), fontsize=18, weight='bold')
plt.yticks(fontsize=18, weight='bold')

plt.grid(True, linestyle='--', alpha=0.7)

# 添加数值标签
for i in range(10):
    # 在柱子上方显示三组数据的值
    plt.text(i+1, original_data[i], f'{original_data[i]:.4f}', ha='center', va='bottom', fontsize=16, color='blue', weight='bold')
    plt.text(i+1, small_change_data[i], f'{small_change_data[i]:.4f}', ha='center', va='bottom', fontsize=16, color='red', weight='bold')
    plt.text(i+1, mutation_data[i], f'{mutation_data[i]:.4f}', ha='center', va='bottom', fontsize=16, color='green', weight='bold')

plt.tight_layout()
plt.savefig('random_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()    