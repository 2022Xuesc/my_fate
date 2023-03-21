from .regularizer import _Regularizer, EPSILON
import federatedml.nn.backend.distiller
from ..thresholding import GroupThresholdMixin


class GroupLassoRegularizer(GroupThresholdMixin, _Regularizer):
    def __init__(self, name, model, reg_regims, threshold_criteria=None):
        super(GroupLassoRegularizer, self).__init__(name, model, reg_regims, threshold_criteria)
        assert threshold_criteria in [None, "Max", "Mean_Abs"]

    def loss(self, param, param_name, regularizer_loss, zeros_mask_dict):
        if param_name in self.reg_regims.keys():
            group_type = self.reg_regims[param_name][1]
            strength = self.reg_regims[param_name][0]
            if group_type == '2D':
                regularizer_loss += GroupLassoRegularizer.__2d_kernelwise_reg(param, strength)
            elif group_type == 'Channels':
                regularizer_loss += GroupLassoRegularizer.__3d_channelwise_reg(param, strength)
            elif group_type == 'Filters':
                regularizer_loss += GroupLassoRegularizer.__3d_filterwise_reg(param, strength)

    @staticmethod
    def __2d_kernelwise_reg(layer_weights, strength):
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"
        view_2d = layer_weights.view(-1, layer_weights.size(2) * layer_weights.size(3))
        return GroupLassoRegularizer.__grouplasso_reg(view_2d, strength, dim=1)

    @staticmethod
    def __3d_channelwise_reg(layer_weights, strength):
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"
        layer_channels_l2 = GroupLassoRegularizer._channels_l2(layer_weights).sum().mul_(strength)
        return layer_channels_l2

    @staticmethod
    def __3d_filterwise_reg(layer_weights, strength):
        assert layer_weights.dim() == 4, "This regularization is only supported for 4D weights"
        layer_filters_l2 = GroupLassoRegularizer._filters_l2(layer_weights).sum().mul_(strength)
        return layer_filters_l2

    @staticmethod
    def _filters_l2(layer_weights, strength):
        filters_view = layer_weights.view(layer_weights.size(0), -1)
        return GroupLassoRegularizer.__grouplasso_reg(filters_view, strength, dim=1)

    @staticmethod
    def _channels_l2(layer_weights):
        # 获取滤波器的个数
        num_filters = layer_weights.size(0)
        num_kernels_per_filter = layer_weights.size(1)
        view_2d = layer_weights.view(-1, layer_weights.size(2) * layer_weights.size(3))
        k_sq_sums = view_2d.pow(2).sum(dim=1)
        k_sq_sums_mat = k_sq_sums.view(num_filters, num_kernels_per_filter).t()
        channels_l2 = k_sq_sums_mat.sum(dim=1).add(EPSILON).pow(1 / 2.)
        return channels_l2

    @staticmethod
    def __grouplasso_reg(groups, strength, dim):
        if dim == -1:
            return groups.norm(2) * strength
        return groups.norm(2, dim=dim).sum().mul_(strength)
