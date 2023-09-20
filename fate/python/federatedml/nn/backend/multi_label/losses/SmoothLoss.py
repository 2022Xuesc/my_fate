import torch
import torch.nn as nn

__all__ = ['SmoothLoss', 'LabelSmoothLoss', 'FeatureSmoothLoss']


class SmoothLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(SmoothLoss, self).__init__()
        # 对于负类样本的损失权重的惩罚因子
        self.gamma_neg = gamma_neg
        # 对于正类样本的损失权重的惩罚因子
        # 对于负类样本主导的数据集，设置惩罚因子大于正类样本的惩罚因子
        self.gamma_pos = gamma_pos
        # 截断阈值，对于负类样本，如果gamma_pos<=0.05，则认为其预测准确，不计算其损失值
        self.clip = clip
        # Todo: 该设置项的作用是什么？
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # 计算概率
        x_sigmoid = torch.sigmoid(x)
        # 预测为正类的概率
        xs_pos = x_sigmoid
        # 预测为负类的概率
        xs_neg = 1 - x_sigmoid

        # 非对称截断
        if self.clip is not None and self.clip > 0:
            # 将xs_neg加上截断值，再设置其上界为1
            # 比如给定某个类的输出概率为0.01，转换成预测负类的概率为0.99，加上clip=0.05，截断到1
            # 这样将该类预测为负类的概率为1，将其排除到了损失函数的计算之外
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 标准交叉熵计算公式，log前面是没有权重系数的
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # 非对称关注
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Todo: 这里用来控制是否计算梯度？
            #  禁止损失在focal loss上进行梯度传播
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            # 如果y=1，则pt为第0项pt0 = xs_pos，这个概率越大，对应权重应越小
            pt0 = xs_pos * y
            # 如果y=0，则pt为第1项pt1 = xs_neg，这个概率月大，对应权重应越小
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            # 根据标签y得到gamma值，这里没有使用if语句，而是加法
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            # 计算权重系数
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            # 为计算好的交叉熵损失乘以权重系数
            loss *= one_sided_w
        # 返回损失的负值
        return -loss.sum()





class LabelSmoothLoss(nn.Module):
    # 进行相关参数的设置
    def __init__(self):
        super(LabelSmoothLoss, self).__init__()

    # 传入
    # 1. 预测概率y：b * 80
    # 2. 图像语义相似度: b * b
    # 3. 标签相关性: 80 * 80
    def forward(self, predicts, similarities, A):
        batch_size = len(predicts)
        # 相似图像的个数是 batch_size // 2
        k = batch_size // 2
        num_labels = 80
        # 这里除以(num_labels - 1)而不是除以num_labels是因为不考虑自转移的情况
        total_loss = torch.norm(predicts - torch.matmul(torch.matmul(similarities, predicts), A) / k / (num_labels - 1), p=2)
        return total_loss / batch_size

class FeatureSmoothLoss(nn.Module):
    # 进行相关参数的设置
    def __init__(self):
        super(FeatureSmoothLoss, self).__init__()

    # 传入
    # 1. CNN输出的该批次图像的特征，需要带梯度
    # 2. 图像之间的相似性
    def forward(self, features, similarities):
        batch_size = len(features)
        features = features.reshape(batch_size, -1)
        total_loss = torch.norm(features - torch.matmul(similarities, features), p=2)
        # Todo: 这里用总损失还是平均损失
        return total_loss / batch_size
