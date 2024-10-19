from .regularizer import _Regularizer
from .. import create_mask_threshold_criterion


class L1Regularizer(_Regularizer):
    def __init__(self, name, model, reg_regims, threshold_criteria=None):
        super(L1Regularizer, self).__init__(name, model, reg_regims, threshold_criteria)

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        if param_name in self.reg_regims:
            strength = self.reg_regims[param_name]
            regularizer_loss += L1Regularizer.__add__l1(param, strength)
        return regularizer_loss

    # Todo: 这里的阈值是计算什么的？
    # 根据配置好的阈值计算
    def threshold(self, param, param_name, zeros_mask_dict):
        if self.threshold_criteria is None or param_name not in self.reg_regims:
            return
        strength = self.reg_regims[param_name]
        zeros_mask_dict[param_name].mask = create_mask_threshold_criterion(param, threshold=strength)
        zeros_mask_dict[param_name].is_regularization_mask = True

    @staticmethod
    def __add__l1(var, strength):
        return var.abs().sum() * strength

    @staticmethod
    def __add_l1_all(loss, model, reg_regims):
        for param_name, param in model.named_parameters():
            if param_name in reg_regims.keys():
                strength = reg_regims[param_name]
                loss += L1Regularizer.__add__l1(param, strength)
