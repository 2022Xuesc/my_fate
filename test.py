import torch

# 创建一个矩阵 A 和一个向量 b
A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = torch.tensor([3.0, 7.0, 11.0])

# 使用 torch.linalg.lstsq 求解最小二乘解
x, residuals, rank, s = torch.linalg.lstsq(A, b)

# 输出结果
print("最小二乘解 x:")
print(x)

print("残差:")
print(residuals)

print("矩阵的秩:")
print(rank)

print("奇异值:")
print(s)
