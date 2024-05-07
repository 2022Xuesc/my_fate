import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


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
            # Todo: 注意这里需要进行转置
            adj = torch.transpose(torch.from_numpy(adjList), 0, 1)
            if constraint:
                adj = self.un_sigmoid(adj)
            self.static_adj.data.copy_(adj)

        self.static_weight = nn.Sequential(  # 静态图卷积的变换矩阵，将in_features变换到out_features
            nn.Conv1d(in_features, in_features, 1),  # 残差连接要求维度相同
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)  # 动态图卷积的变换矩阵

    def un_sigmoid(self, adjList):
        un_adjList = -np.log(1 / (adjList + np.exp(-8)) - 1)
        # un_adjList中可能出现nan，则替换成100即可
        un_adjList[np.isnan(un_adjList)] = 100
        return un_adjList

    def forward_static_gcn(self, x):
        if self.constraint:
            adj = torch.sigmoid(self.static_adj)
        else:
            adj = self.static_adj
        x = torch.matmul(adj, x.transpose(1, 2))  # Todo: 注意这里的static_adj是原先概率矩阵的转置
        # 将static_adj和relu拆分开来
        x = self.relu(x)
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        # Todo: 这个是原来的sigmoid
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        x = x.transpose(1, 2)
        out_static = self.forward_static_gcn(x)
        x = x + out_static  # Todo: 残差连接
        dynamic_adj = self.forward_construct_dynamic_graph(x)

        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class PRUNED_ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None, needOptimize=True,
                 constraint=False):
        super(PRUNED_ADD_GCN, self).__init__()
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
        self.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_features * 2, out_features),
        #     nn.Tanh()
        # )
        # self.classifier = Element_Wise_Layer(num_classes, out_features)

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x, inp):
        x = self.forward_feature(x)
        # Todo: x展平+池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc(x)
        # Todo: 第一维应该是batch，这里将二维特征展平
        z = self.forward_dgcn(inp)
        z = z.transpose(1, 2)

        # output = torch.cat([z, x], dim=-1)
        # output = self.fc(output)
        # output = self.classifier(output)

        # z的维度是batch_size * num_classes * feat_dim
        # x的维度是batch_size * feat_dim
        # Todo: 采用直接求点积的方式,out2算下来太大了

        out2 = torch.matmul(z, x.unsqueeze(-1)).squeeze(-1) / z.size(2)
        return out1, out2

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
