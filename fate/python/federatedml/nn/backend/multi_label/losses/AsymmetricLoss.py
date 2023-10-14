import torch
import torch.nn as nn

__all__ = ['AsymmetricLoss', 'AsymmetricLossOptimized']


# Todo: 对损失进行修改,从外部传入x的sigmoid值,而不是在里边进行计算
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
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
        x_sigmoid = x
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


class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # 计算预测为正标签的概率和预测为负标签的概率
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            # 这里使用原地操作
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # 基本的交叉熵损失计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        # 和未优化的版本逻辑相同
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
