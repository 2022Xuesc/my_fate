import torch
import torch.nn as nn

from torch.nn import Parameter

# 动态图卷积层
class DynamicGraphConvolution(nn.Module):
    # 节点的输入特征
    # 节点的输出特征
    def __init__(self, in_features, out_features, num_nodes, adjList=None, needOptimize=True):
        super(DynamicGraphConvolution, self).__init__()
        # Todo: 静态相关性矩阵随机初始化得到
        if adjList is not None:
            self.static_adj = Parameter(torch.Tensor(num_nodes, num_nodes))
            adj = torch.transpose(torch.from_numpy(adjList), 0, 1)
            self.static_adj.data.copy_(adj)
            
        self.static_weight = nn.Sequential(  # 静态图卷积的变换矩阵，将in_features变换到out_features
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)  # 动态图卷积的变换矩阵

    def forward_static_gcn(self, x):
        x = torch.matmul(self.static_adj, x.transpose(1, 2))
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

    def forward(self, x, out1):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static  # Todo: 其实就是自己本身的特征向量不变
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=1024, out_features=1024, adjList=None, needOptimize=True):
        super(ADD_GCN, self).__init__()
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
        x = self.gcn(x, out1)
        return x

    def forward(self, x):
        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)  # 计算初始分类器的预测向量

        v = self.forward_sam(x)  # B*1024*num_classes，每个类别的注意力向量
        z = self.forward_dgcn(v, out1)
        z = v + z

        out2 = self.last_linear(z)  # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
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
