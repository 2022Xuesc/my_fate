import torch


# noinspection PyProtectedMember
class SplicingPruner(object):
    def __init__(self, name, sensitivities, low_thresh_mult, hi_thresh_mult, sensitivity_multiplier=0):
        self.name = name
        self.sensitivities = sensitivities
        self.low_thresh_mult = low_thresh_mult
        self.hi_thresh_mult = hi_thresh_mult
        self.sensitivity_multiplier = sensitivity_multiplier

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.sensitivities:
            if '*' not in self.sensitivities:
                return
            else:
                sensitivity = self.sensitivities['*']
        else:
            sensitivity = self.sensitivities[param_name]

        # 关于敏感度因子的动态变化
        if self.sensitivity_multiplier > 0:
            starting_epoch = meta['starting_epoch']
            current_epoch = meta['current_epoch']
            sensitivity *= (current_epoch - starting_epoch) * self.sensitivity_multiplier + 1

        if zeros_mask_dict[param_name].mask is None:
            zeros_mask_dict[param_name].mask = torch.ones_like(param)
        # 计算当前epoch的mask需要用到上一个epoch的mask
        zeros_mask_dict[param_name].mask = self.create_mask(param,
                                                            zeros_mask_dict[param_name].mask,
                                                            sensitivity,
                                                            self.low_thresh_mult,
                                                            self.hi_thresh_mult)

    @staticmethod
    def create_mask(param, current_mask, sensitivity, low_thresh_mult, hi_thresh_mult):
        with torch.no_grad():
            if not hasattr(param, '_std'):
                param._std = torch.std(param.abs()).item()
                param._mean = torch.mean(param.abs()).item()
            threshold_low = (param._mean + param._std * sensitivity) * low_thresh_mult
            threshold_hi = (param._mean + param._std * sensitivity) * hi_thresh_mult
            # 代码实现的是Dynamic Network Surgery文章中的等式(3)
            #           0    if a  > |W|
            # h(W) =    mask if a <= |W| < b
            #           1    if b <= |W|
            # h(W) 是所谓的网络手术函数
            # mask 是之前迭代中使用的掩膜矩阵
            # a 和 b分别是剪枝阈值
            zeros, ones = torch.zeros_like(current_mask), torch.ones_like(current_mask)
            weights_abs = param.abs()
            # Todo: torch.where(condition,x,y)用法，如果满足条件，则返回x，否则返回y
            new_mask = torch.where(threshold_low >= weights_abs, zeros, current_mask)
            new_mask = torch.where(threshold_hi <= weights_abs, ones, new_mask)
            return new_mask
