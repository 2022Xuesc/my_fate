import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# 动态图卷积层
class DynamicGraphConvolution(nn.Module):
    # 节点的输入特征
    # 节点的输出特征
    def __init__(self, in_features, out_features, num_nodes, adjList=None, needOptimize=True, constraint=False,
                 label_prob_vec=None):
        super(DynamicGraphConvolution, self).__init__()
        # Todo: 静态相关性矩阵随机初始化得到
        self.constraint = constraint
        self.label_prob_vec = label_prob_vec
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
        # Todo: 和global相关的下列参数都不使用了
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
        return x, adj  # 将静态相关性矩阵返回

    def forward_construct_dynamic_graph(self, x, connect_vec):
        # Todo: 扩展后拼接
        connect_vec = connect_vec.unsqueeze(-1).expand(connect_vec.size(0), connect_vec.size(1), x.size(2))
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((connect_vec, x), dim=1)
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

        num_classes = out1.size(1)
        dynamic_adj_loss = torch.tensor(0.).to(out1.device)
        # Todo: 概率向量损失不太好，怎么改进一下？
        #   1. 改进1：out1不太稳定，对其进行sigmoid
        #   2. 改进2：使用更加精细的控制代替矩阵乘法？之前的工作
        out1 = nn.Sigmoid()(out1)
        if prob:
            # Todo: 最简单的计算过渡输出的方法
            # transformed_out1 = torch.matmul(out1.unsqueeze(1), dynamic_adj).squeeze(1)
            # # Todo: 这里的话求过和，需要除以类别数量
            # transformed_out1 /= num_classes
            # # 第0维是batch_size，非对称损失也不求平均；因此，无需torch.mean
            # dynamic_adj_loss += torch.sum(torch.norm(out1 - transformed_out1, dim=1))
            # Todo: 使用正负相关的方法
            device = out1.device
            batch_size = len(out1)
            candidates = torch.zeros((batch_size, num_classes), dtype=torch.float64).to(device)
            exists_lower_bound = 0.5  # 如果大于0.5，则认为存在
            relation_gap = 0
            for b in range(batch_size):
                predict_vec = out1[b]
                for lj in range(num_classes):
                    relation_num = 0
                    incr_lower_bound = self.label_prob_vec[lj] + relation_gap
                    decr_upper_bound = self.label_prob_vec[lj] - relation_gap
                    for li in range(num_classes):
                        if li == lj:  # 是同一个标签，则1转移
                            relation_num += 1
                            candidates[b][lj] += predict_vec[li]
                            continue
                        if predict_vec[li] > exists_lower_bound:
                            # Todo: 这里使用网络生成的动态相关性矩阵
                            a = dynamic_adj[b][li][lj].item()
                            relation_num += 1
                            # 标签li对标签lj起促进作用，直接相乘即可
                            if a > incr_lower_bound:
                                # 仅当起促进作用时，才累加
                                candidates[b][lj] += predict_vec[li] * a
                            elif a < decr_upper_bound:  # 标签li对lj起抑制作用
                                candidates[b][lj] += 1 - predict_vec[li] * (1 - a)
                    # 按照转移的标签数量进行平均
                    # 里边既有促进作用的部分，也有抑制作用的部分
                    candidates[b][lj] /= relation_num
            # Todo: 使用生成的candidates来计算过渡损失
            dynamic_adj_loss += torch.sum(torch.norm(out1 - candidates, dim=1))
        if gap:
            diff = dynamic_adj - static_adj
            # 第0维是batch_size，非对称损失也不求平均；因此，无需torch.mean
            dynamic_adj_loss += torch.sum(torch.norm(diff.reshape(diff.size(0), -1), dim=1)) / num_classes

        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x, dynamic_adj_loss


class CONNECTED_ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, in_features=300, out_features=2048, adjList=None, needOptimize=True,
                 constraint=False, prob=False, gap=False,label_prob_vec=None):
        super(CONNECTED_ADD_GCN, self).__init__()
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
                                           constraint, label_prob_vec=label_prob_vec)
        # 这里的fc是1000维的，改成num_classes维
        feat_dim = 2048
        # 将CNN特征经过FC后拼接到静态图特征后面
        self.connect = torch.nn.Linear(in_features=feat_dim, out_features=in_features, bias=True)
        self.fc = torch.nn.Linear(in_features=feat_dim, out_features=num_classes, bias=True)
        self.prob = prob
        self.gap = gap

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
        x = torch.flatten(x, 1)
        out1 = self.fc(x)

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
