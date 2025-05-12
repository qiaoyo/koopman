import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成第一组数据 (3w-5w)
group1 = np.random.uniform(30000, 50000, 10)

# 生成第二组数据 (第一组的0.9左右)
group2 = group1 * 0.9 + np.random.normal(0, 1000, 10)

# 生成第三组数据 (第一组的0.5左右)
group3 = group1 * 0.5 + np.random.normal(0, 2000, 10)

# 生成第四组数据 (第一组的0.7左右)
group4 = group1 * 0.7 + np.random.normal(0, 1500, 10)

# 生成第五组数据 (第一组的0.9左右)
group5 = group1 * 0.9 + np.random.normal(0, 1200, 10)

# 打印每组数据的统计信息
for i, group in enumerate([group1, group2, group3, group4, group5], 1):
    print(f"\n第{i}组数据:")
    print(f"数据: {group}")
    print(f"平均值: {np.mean(group):.2f}")
    print(f"标准差: {np.std(group):.2f}")
    print(f"方差: {np.var(group):.2f}")
    print(f"与第一组的比例: {np.mean(group)/np.mean(group1):.2f}") 