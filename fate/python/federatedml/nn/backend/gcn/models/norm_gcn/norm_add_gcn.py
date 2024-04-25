import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# Todo: 主体部分已经完成
#  BN层
#  bias的设置;
# 动态图卷积层


class DynamicGraphConvolution(nn.Module):
    # 节点的输入特征
    # 节点的输出特征
    def __init__(self, in_features, out_features, num_nodes, adjList=None, needOptimize=True, norm_method="standard",
                 ):
        super(DynamicGraphConvolution, self).__init__()
        # 可优化的adj参数
        self.adj_param = Parameter(torch.Tensor(num_nodes, num_nodes))

        if norm_method == 'sigmoid':
            # Todo: 需要进行un_sigmoid
            self.adj_param.data.copy_(self.un_sigmoid(torch.from_numpy(adjList)))
        else:
            # Todo: 无需进行
            self.adj_param.data.copy_(torch.from_numpy(adjList))

        self.static_weight = Parameter(torch.Tensor(in_features, out_features))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层
        self.dynamic_weight = Parameter(torch.Tensor(in_features, out_features))  # 动态图卷积的变换矩阵

        self.reset_weight_parameters()

        self.normalizations = {'standard': self.standard_normalization,
                               'sigmoid': torch.sigmoid}
        self.norm_method = norm_method

        self.bn = nn.BatchNorm1d(in_features)

    def standard_normalization(self, x):
        # x是一个20*20的矩阵
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return x

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

    # 根据标签相关性矩阵计算adj
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
        support = torch.matmul(torch.transpose(input, 1, 2),
                               weight)  # batch * feat * num_classes ; weight = num_classes * num_classes
        output = torch.matmul(adj, support)
        return torch.transpose(output, 1, 2)  # 再次进行转置

    def forward_static_gcn(self, x):
        # 先对adj_param的参数进行sigmoid，将其归一化到【0，1】之间，然后再进行计算
        # self.adj_param进行sigmoid，然后走标准gcn的方式
        static_adj = self.normalizations[self.norm_method](self.adj_param)
        # adj_param = self.adj_param
        adj = self.gen_adj(static_adj)
        x = self.forward_gcn(x, self.static_weight, adj)
        # Todo: 需要进行BatchNorm吗？
        x = torch.transpose(x, 1, 2)
        batch_size, num_classes, feature_dim = x.size()
        x_reshaped = x.view(-1, feature_dim)
        x_reshaped = self.bn(x_reshaped)
        x = x_reshaped.view(batch_size, num_classes, feature_dim)
        x = self.relu(x)
        x = torch.transpose(x, 1, 2)
        return x, static_adj

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
        dynamic_adj = self.normalizations[self.norm_method](dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adjs):
        # 对于动态图来说，每张图像都有一个图
        transformed_adjs = self.gen_adjs(dynamic_adjs)
        x = self.forward_gcn(x, self.dynamic_weight, transformed_adjs)


        x = torch.transpose(x, 1, 2)
        batch_size, num_classes, feature_dim = x.size()
        x_reshaped = x.view(-1, feature_dim)
        x_reshaped = self.bn(x_reshaped)
        x = x_reshaped.view(batch_size, num_classes, feature_dim)

        x = torch.transpose(x, 1, 2)

        x = self.relu(x)
        return x

    def forward(self, x, out1):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static, static_adj = self.forward_static_gcn(x)
        x = x + out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        # 计算dynamic_adj在out1上经过一次作用后对out1的改变情况，使用余弦相似度作为度量
        # out1的维度 batch_size * num_classes， dynamic_adj的维度 batch_size * num_classes * num_classes

        transformed_out1 = torch.matmul(out1.unsqueeze(1), dynamic_adj).squeeze(1)

        # Todo: 余弦相似度不可取，为什么不可取？
        # dynamic_adj_loss = 1 - torch.mean(torch.cosine_similarity(out1, transformed_out1, dim=1))

        # Todo: 使用平方损失？不该求平均的，因为asym_loss就是未平均的
        #  需要对变换的标签值进行平均，直接除以num_classes即可
        transformed_out1 /= transformed_out1.size(1)
        dynamic_adj_loss = torch.sum(torch.norm(out1 - transformed_out1, dim=1))
        diff = dynamic_adj - static_adj

        dynamic_adj_loss += torch.sum(torch.norm(diff.reshape(diff.size(0), -1), dim=1))
        # Todo: 归一化后求2范数？
        x = self.forward_dynamic_gcn(x, dynamic_adj)

        # Todo: 动态对称损失要求和静态约束损失接近，这个感觉得加

        return x, dynamic_adj_loss


class NORM_ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=1024, out_features=1024, adjList=None, needOptimize=True):
        super(NORM_ADD_GCN, self).__init__()
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
        # 1和1的卷积核
        # 输入H * W * in_features
        # 输出H * W * num_classes，相当于每个类别一个注意力图
        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1, 1), bias=False)

        self.conv_transform = nn.Conv2d(2048, in_features, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        self.gcn = DynamicGraphConvolution(in_features, out_features, num_classes, adjList, needOptimize)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())  # 单位矩阵，自相关性
        self.last_linear = nn.Conv1d(out_features, self.num_classes, 1)  # 最终的分类层

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)  # batch_size, num_classes , H, W
        x = x.view(x.size(0), x.size(1), -1)  # batch_size, num_classes, H * W
        x = x.topk(1, dim=-1)[0].mean(dim=-1)  # 从topk1这里可以看出是最大池化
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)  # 求sigmoid，将其归为0到1之间，注意力值
        mask = mask.transpose(1, 2)  # B, H*W, N

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x, out1):
        x, dynamic_adj_loss = self.gcn(x, out1)
        return x, dynamic_adj_loss

    def forward(self, x):
        y = self.forward_feature(x)

        out1 = self.forward_classification_sm(y)  # 计算初始分类器的预测向量

        v = self.forward_sam(y)  # B*1024*num_classes，每个类别的注意力向量
        z, dynamic_adj_loss = self.forward_dgcn(v, out1)
        z = v + z

        out2 = self.last_linear(z)  # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
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
