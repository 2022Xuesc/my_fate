import torch
import torch.nn

EPSILON = 1e-8


class _Regularizer(object):
    def __init__(self, name, model, reg_regims, threshold_criteria):
        """正则化器的基类

        :param name: 正则化器的名称
        :param model: 应用正则化的模型
        :param reg_regims: 量化损失的强度：str-->float或tuple[float]的字典
        :param threshold_criteria: 计算阈值的度量
        """
        self.name = name
        self.model = model
        self.reg_regims = reg_regims
        self.threshold_criteria = threshold_criteria

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        """应用正则化损失

        :param param: 计算正则项的参数
        :param param_name: 参数名称
        :param regularizer_loss: 之前计算好的正则项损失
        :param zeros_mask_dict: 掩膜矩阵的配置
        :return: 对当前参数应用额外损失的正则化损失
        """
        raise NotImplementedError

    def threshold(self, param, param_name, zeros_mask_dict):
        """计算剪枝阈值

        :param param: 参数张量
        :param param_name: 参数名称
        :param zeros_mask_dict: 掩膜矩阵的配置
        :return:
        """
        return NotImplementedError
