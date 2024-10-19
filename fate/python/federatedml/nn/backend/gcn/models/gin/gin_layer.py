import torch
import torch.nn as nn
from torch.nn import Parameter


# 定义GIN层
class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Todo: 中间暂定为in * 2
        self.mid_features = out_features * 2

        self.fc1 = torch.nn.Linear(self.in_features, self.mid_features, bias)
        self.fc2 = torch.nn.Linear(self.mid_features, self.out_features, bias)

        self.bn = nn.BatchNorm1d(self.out_features)

        # epsilon初始化为0
        self.epsilon = Parameter(torch.zeros(1), requires_grad=True)

    # 前向传播，权重与输入相乘、结果再与邻接矩阵adj相乘。
    def forward(self, x, adj):
        x = (1 + self.epsilon) * x + torch.matmul(adj, x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x
