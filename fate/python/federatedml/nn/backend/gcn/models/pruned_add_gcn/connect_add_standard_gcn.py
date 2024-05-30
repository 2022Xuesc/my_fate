import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# 动态图卷积层
class DynamicGraphConvolution(nn.Module):
    # 节点的输入特征
    # 节点的输出特征
    def __init__(self, in_features, out_features, num_nodes, adjList=None, needOptimize=True, constraint=False):
        super(DynamicGraphConvolution, self).__init__()
        # Todo: 静态相关性矩阵随机初始化得到
        self.constraint = constraint

        if adjList is not None:
            self.static_adj = Parameter(torch.Tensor(num_nodes, num_nodes))
            # 标准gcn无需进行转置
            adj = torch.from_numpy(adjList)
            if constraint:
                adj = self.un_sigmoid(adj)
            self.static_adj.data.copy_(adj)

        # Todo: 残差连接要求维度相同
        self.static_weight = Parameter(torch.Tensor(in_features, in_features))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层
        self.dynamic_weight = Parameter(torch.Tensor(in_features, out_features))

        # 手动创建的网络参数，需要进行reset
        self.reset_weight_parameters()

    def updateA(self, adjList):
        self.static_adj.data.copy_(torch.from_numpy(adjList).float())

    def getAdj(self):
        return self.static_adj.data.cpu().numpy()

    def reset_weight_parameters(self):
        # 为static_weight规范化
        static_stdv = 1. / math.sqrt(self.static_weight.size(1))
        self.static_weight.data.uniform_(-static_stdv, static_stdv)

        # 为dynamic_weight规范化
        dynamic_stdv = 1. / math.sqrt(self.dynamic_weight.size(1))
        self.dynamic_weight.data.uniform_(-dynamic_stdv, dynamic_stdv)

    def un_sigmoid(self, adjList):
        un_adjList = -np.log(1 / (adjList + np.exp(-8)) - 1)
        # un_adjList中可能出现nan，则替换成100即可
        un_adjList[np.isnan(un_adjList)] = 100
        return un_adjList

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        # 这一步执行的运算其实是D^T * A^T * D
        # 前面已经经过转置了，那这一步应该不用对A进行转置
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def gen_adjs(self, A):
        batch_size = A.size(0)
        adjs = torch.zeros_like(A)
        for i in range(batch_size):
            # 这里对行求和
            # Todo: 问题出在这里，怎么能对其进行约束
            D = torch.pow(A[i].sum(1).float(), -0.5)
            # 将其转换成对角矩阵
            D = torch.diag(D)
            adj = torch.matmul(torch.matmul(A[i], D).t(), D)
            adjs[i] = adj
        return adjs

    def forward_gcn(self, input, weight, adj):
        output = torch.matmul(adj, torch.transpose(input, 1, 2))
        output = self.relu(output)
        output = torch.matmul(output, weight)
        return torch.transpose(output, 1, 2)  # 再次进行转置

    def forward_static_gcn(self, x):
        if self.constraint:
            adj = torch.sigmoid(self.static_adj)
        else:
            adj = self.static_adj
        returned_adj = adj
        adj = self.gen_adj(adj)
        x = self.forward_gcn(x, self.static_weight, adj)
        return x, returned_adj

    def forward_construct_dynamic_graph(self, x, connect_vec):
        ### Construct the dynamic correlation matrix ###
        connect_vec = connect_vec.unsqueeze(-1).expand(connect_vec.size(0), connect_vec.size(1), x.size(2))
        x = torch.cat((connect_vec, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        # Todo: 这个是原来的sigmoid
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        transformed_adjs = self.gen_adjs(dynamic_adj)
        x = self.forward_gcn(x, self.dynamic_weight, transformed_adjs)
        x = self.relu(x)
        return x

    def forward(self, x, connect_vec, out1=None, prob=False, gap=False):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        x = x.transpose(1, 2)
        out_static, static_adj = self.forward_static_gcn(x)
        x = x + out_static  # Todo: 残差连接
        dynamic_adj = self.forward_construct_dynamic_graph(x, connect_vec)

        # Todo: 这里引入动态损失
        #  1. 第一部分，概率向量损失
        num_classes = out1.size(1)
        dynamic_adj_loss = torch.tensor(0.).to(out1.device)
        if prob:
            transformed_out1 = torch.matmul(out1.unsqueeze(1), dynamic_adj).squeeze(1)
            transformed_out1 /= num_classes
            # 第0维是batch_size，非对称损失也不求平均；因此，无需torch.mean
            dynamic_adj_loss += torch.sum(torch.norm(out1 - transformed_out1, dim=1))
        if gap:
            diff = dynamic_adj - static_adj
            # 第0维是batch_size，非对称损失也不求平均；因此，无需torch.mean
            dynamic_adj_loss += torch.sum(torch.norm(diff.reshape(diff.size(0), -1), dim=1)) / num_classes
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x, dynamic_adj_loss


class CONNECT_ADD_STANDARD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None, needOptimize=True,
                 constraint=False, prob=False, gap=False):
        super(CONNECT_ADD_STANDARD_GCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gcn = DynamicGraphConvolution(in_features, out_features, num_classes, adjList, needOptimize,
                                           constraint)
        # 这里的fc是1000维的，改成num_classes维
        feat_dim = 2048

        self.connect = torch.nn.Linear(in_features=feat_dim, out_features=in_features, bias=True)

        self.fc = torch.nn.Linear(in_features=feat_dim, out_features=num_classes, bias=True)
        self.prob = prob
        self.gap = gap

    def updateA(self, adjList):
        self.gcn.updateA(adjList)

    def getAdj(self):
        return self.gcn.getAdj()

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_dgcn(self, x, connect_vec, out1, prob, gap):
        # Todo: 在这里调用，在这里配置
        x, dynamic_loss = self.gcn(x, connect_vec, out1, prob, gap)
        return x, dynamic_loss

    def forward(self, x, inp):
        x = self.forward_feature(x)
        # Todo: x展平+池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 计算得到图像特征之后，将其传给dgcn
        out1 = self.fc(x)
        # 经过一个全连接层作用
        connect_vec = self.connect(x)
        # Todo: 第一维应该是batch，这里将二维特征展平
        z, dynamic_adj_loss = self.forward_dgcn(inp, connect_vec, out1, self.prob, self.gap)
        z = z.transpose(1, 2)

        # output = torch.cat([z, x], dim=-1)
        # output = self.fc(output)
        # output = self.classifier(output)

        # z的维度是batch_size * num_classes * feat_dim
        # x的维度是batch_size * feat_dim
        # Todo: 采用直接求点积的方式,out2算下来太大了

        out2 = torch.matmul(z, x.unsqueeze(-1)).squeeze(-1) / z.size(2)
        # Todo: dynamic_adj_loss也不一定有值，
        return out1, out2, dynamic_adj_loss

    def get_config_optim(self, lr, lrp):
        # 与GCN类似
        # 特征层的参数使用较小的学习率
        # 其余层的参数使用较大的学习率
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]
