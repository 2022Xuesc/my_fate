import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


class GINLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Todo: 中间暂定为in
        self.mid_features = out_features

        self.fc1 = torch.nn.Linear(self.in_features, self.mid_features, bias)
        self.fc2 = torch.nn.Linear(self.mid_features, self.out_features, bias)

        self.bn = nn.BatchNorm1d(self.out_features)

        # epsilon初始化为0
        self.epsilon = Parameter(torch.zeros(1), requires_grad=True)

    # 前向传播，权重与输入相乘、结果再与邻接矩阵adj相乘。
    def forward(self, x, adj):
        x = torch.transpose(x, 1, 2)
        x = (1 + self.epsilon) * x + torch.matmul(adj, x)
        x = self.fc1(x)
        x = self.fc2(x)
        # Todo: 这个bn怎么添加？
        batch_size, num_classes, feature_dim = x.size()
        x_reshaped = x.view(-1, feature_dim)
        x_reshaped = self.bn(x_reshaped)
        x = x_reshaped.view(batch_size, num_classes, feature_dim)
        return torch.transpose(x, 1, 2)


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

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层

        # 定义gin相关的层
        # 一个行不行？
        self.static_gin1 = GINLayer(in_features, in_features)
        self.static_gin2 = GINLayer(in_features, in_features)

        # 两个gin，之间建立残差连接
        self.dynamic_gin1 = GINLayer(in_features, in_features)
        self.dynamic_gin2 = GINLayer(in_features, in_features)

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
            
        x = self.static_gin1(x, adj)
        x = self.relu(x)
        identity = x
        x = self.static_gin2(x, adj)
        x = self.relu(x + identity)
        return x, adj

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

    def forward_dynamic_gcn(self, x, dynamic_adjs):
        x = self.dynamic_gin1(x, dynamic_adjs)
        x = self.relu(x)
        identity = x
        x = self.dynamic_gin2(x, dynamic_adjs)
        x = self.relu(x + identity)
        return x

    def forward(self, x, out1=None, prob=False, gap=False):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        x = x.transpose(1, 2)
        out_static, static_adj = self.forward_static_gcn(x)
        x = x + out_static  # Todo: 残差连接
        dynamic_adj = self.forward_construct_dynamic_graph(x)

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


class PRUNED_ADD_GIN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None, needOptimize=True,
                 constraint=False, prob=False, gap=False):
        super(PRUNED_ADD_GIN, self).__init__()
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
        self.prob = prob
        self.gap = gap

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_dgcn(self, x, out1, prob, gap):
        # Todo: 在这里调用，在这里配置
        x, dynamic_loss = self.gcn(x, out1, prob, gap)
        return x, dynamic_loss

    def forward(self, x, inp):
        x = self.forward_feature(x)
        # Todo: x展平+池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc(x)
        # Todo: 第一维应该是batch，这里将二维特征展平
        z, dynamic_adj_loss = self.forward_dgcn(inp, out1, self.prob, self.gap)
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
