import torch
import torch.nn as nn
from torch.nn import Parameter


# 自定义的矩阵
class CustomMatrix(nn.Module):
    def __init__(self, adjList, needTranspose=False):
        super(CustomMatrix, self).__init__()
        num_labels = len(adjList)
        self.fixed_diagonal = Parameter(torch.tensor([1. for _ in range(num_labels)]))
        self.fixed_diagonal.requires_grad_(False)

        self.off_diagonal = Parameter(torch.Tensor(num_labels, num_labels))
        adj = torch.from_numpy(adjList)
        if needTranspose:
            adj = torch.transpose(adj, 0, 1)
        self.off_diagonal.data.copy_(adj)
        # self.mask不在
        self.mask = Parameter(torch.eye(num_labels))
        self.mask.requires_grad_(False)

    def forward(self):
        return self.fixed_diagonal * self.mask + self.off_diagonal * (1 - self.mask)


def genAdj(dynamic_adj):
    num_labels = len(dynamic_adj[0])
    device = dynamic_adj[0].device
    fixed_diagonal = torch.tensor([1. for _ in range(num_labels)]).to(device)
    mask = torch.eye(num_labels).to(device)
    # fixed_dynamic_adjs
    return fixed_diagonal * mask + dynamic_adj * (1 - mask)
