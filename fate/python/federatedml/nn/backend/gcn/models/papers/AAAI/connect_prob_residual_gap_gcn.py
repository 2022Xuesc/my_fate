import torch
import torch.nn as nn


class DynamicGraphConvolution(nn.Module):
    # 节点的输入特征
    # 节点的输出特征
    def __init__(self, in_features, out_features, num_nodes, adjList=None):
        super(DynamicGraphConvolution, self).__init__()

        self.num_nodes = num_nodes

        # 可优化的adj参数
        self.static_adj = nn.Conv1d(num_nodes, num_nodes, 1, bias=False)

        adj = torch.from_numpy(adjList)
        # Todo: 检查是否需要进行转置
        adj = torch.transpose(adj, 0, 1)
        self.static_adj.weight.data.copy_(adj.unsqueeze(-1))

        self.static_adj = nn.Sequential(
            self.static_adj,
            nn.LeakyReLU(0.2)
        )

        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, in_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)  # 生成动态图的卷积层
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2)) / self.num_nodes
        x = self.static_weight(x.transpose(1, 2))
        return x, list(self.static_adj.modules())[1].weight.squeeze(-1)

    def forward_construct_dynamic_graph(self, x, connect_vec):
        connect_vec = connect_vec.unsqueeze(-1).expand(connect_vec.size(0), connect_vec.size(1), x.size(2))
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((connect_vec, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        # Todo: 学习关于static_adj的残差
        dynamic_adj = torch.sigmoid(dynamic_adj) + self.static_adj[0].weight.squeeze(-1)
        dynamic_adj /= 2
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj) / self.num_nodes
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, connect_vec, out1):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        x = x.transpose(1, 2)
        out_static, static_adj = self.forward_static_gcn(x)
        x = x + out_static
        dynamic_adj = self.forward_construct_dynamic_graph(x, connect_vec)

        # 计算动态标签相关性矩阵损失
        dynamic_adj_loss = torch.tensor(0.).to(out1.device)

        # Todo: 基于prob的动态损失计算
        out1 = nn.Sigmoid()(out1)
        transformed_out1 = torch.matmul(out1.unsqueeze(1), dynamic_adj).squeeze(1)
        transformed_out1 /= self.num_nodes
        # 第0维是batch_size，非对称损失也不求平均；因此，无需torch.mean
        dynamic_adj_loss += torch.sum(torch.norm(out1 - transformed_out1, dim=1))

        # Todo: 基于gap的动态损失计算，在残差的基础上使用损失函数进一步控制残差学习的范围
        diff = dynamic_adj - static_adj
        dynamic_adj_loss += torch.sum(torch.norm(diff.reshape(diff.size(0), -1), dim=1)) / self.num_nodes

        x = self.forward_dynamic_gcn(x, dynamic_adj)

        return x, dynamic_adj_loss


class AAAI_CONNECT_PROB_RESIDUAL_GAP_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None):
        super(AAAI_CONNECT_PROB_RESIDUAL_GAP_GCN, self).__init__()
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

        self.gcn = DynamicGraphConvolution(in_features, out_features, num_classes, adjList)

        feat_dim = 2048
        self.connect = torch.nn.Linear(in_features=feat_dim, out_features=in_features, bias=True)
        self.fc = torch.nn.Linear(in_features=feat_dim, out_features=num_classes, bias=True)

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_dgcn(self, x, connect_vec, out1):
        x = self.gcn(x, connect_vec, out1)
        return x

    def forward(self, x, inp):
        x = self.forward_feature(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc(x)

        connect_vec = self.connect(x)
        z, dynamic_adj_loss = self.forward_dgcn(inp, connect_vec, out1)
        z = z.transpose(1, 2)

        out2 = torch.matmul(z, x.unsqueeze(-1)).squeeze(-1) / z.size(2)

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
