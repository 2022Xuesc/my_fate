import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models


# 根据传入的数据返回分类类别数
def get_outdim(dset):
    if dset == 'imagenet':
        return 1000
    elif dset == 'cifar100':
        return 100
    elif dset == 'cifar10' or dset == 'mnist':
        return 10
    elif dset == 'emnist':
        return 62
    else:
        raise NotImplementedError()


class SingleSubModel(nn.Module):
    # 单个子模型仅仅生成一个输出

    def __init__(self, enc, head, strategy, ind):
        super(SingleSubModel, self).__init__()

        m_list = nn.ModuleList()
        for m in enc:
            m_list.append(m)

        self.enc = m_list
        self.head = head
        self.strategy = strategy
        self.ind = ind

        # Todo: 新阶段的enc和head的参数进行初始化
        for m in self.enc[ind].modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, verbose=False):
        feats = []
        out = x
        for m in self.enc:
            out = m(out)
            feats.append(out)

        if not verbose:
            return self.head(feats[-1])
        else:
            return self.head(feats[-1]), feats

    def print_weight(self):
        for n, p in self.named_parameters():
            print(n, p)

    def trainable_parameters(self):
        # svcca gradually fix the lower-level layers to reduce the cost
        if self.strategy == 'svcca':
            for i in range(self.ind, 4):
                for name, param in self.enc[i].named_parameters():
                    yield param
        else:
            for i in range(self.ind + 1):
                for name, param in self.enc[i].named_parameters():
                    yield param

        for name, param in self.head.named_parameters():
            yield param

    def latest_parameters(self):
        # 返回新添加的层以及之前层的参数
        for name, param in self.enc[self.ind].named_parameters():
            yield param

        if isinstance(self.head, list):
            for h in self.head:
                for name, param in h.named_parameters():
                    yield param
        else:
            for name, param in self.head.named_parameters():
                yield param

    def return_num_parameters(self):
        total = 0
        # Since enc and dec are module lists, we have to travese every model in them.
        for p in self.trainable_parameters():
            total += torch.numel(p)

        return total


class MultiSubModel(SingleSubModel):
    """Submodels that produce multiple outputs"""

    def __init__(self, enc, head, strategy, ind):
        super(MultiSubModel, self).__init__(enc, head, strategy, ind)

        m_list = nn.ModuleList()
        for m in enc:
            m_list.append(m)
        self.enc = m_list

        h_list = nn.ModuleList()
        for h in head:
            h_list.append(h)
        self.head = h_list

        self.strategy = strategy
        self.ind = ind

    def forward(self, x, verbose=False):
        feats = []
        outs = []

        feat = x
        # 同时计算多个子模型的输出
        for m_f, m_o in zip(self.enc, self.head):
            feat = m_f(feat)
            out = m_o(feat)

            outs.append(out)
            feats.append(feat)

        if not verbose:
            return outs
        else:
            return outs, feats
