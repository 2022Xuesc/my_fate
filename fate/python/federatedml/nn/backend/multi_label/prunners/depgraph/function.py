import torch
import torch.nn as nn
import federatedml.nn.backend.multi_label.prunners.depgraph.ops as ops
from copy import deepcopy
from functools import reduce
from operator import mul

from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from typing import Callable, Sequence, Tuple, Dict

# 执行导出过程
__all__ = [
    'BasePruningFunc',
    'PrunerBox',
    'prune_conv_out_channels',
    'prune_conv_in_channels',
    'prune_batchnorm_out_channels',
    'prune_batchnorm_in_channels',
    'prune_linear_out_channels',
    'prune_linear_in_channels',
    'prune_parameter_out_channels',
    'prune_parameter_in_channels',

    'prune_depthwise_conv_out_channels',
    'prune_depthwise_conv_in_channels',
    'prune_prelu_out_channels',
    'prune_prelu_in_channels',
    'prune_layernorm_out_channels',
    'prune_layernorm_in_channels',
    'prune_embedding_out_channels',
    'prune_embedding_in_channels',
    'prune_multihead_attention_out_channels',
    'prune_multihead_attention_in_channels',
    'prune_groupnorm_out_channels',
    'prune_groupnorm_in_channels',
    'prune_instancenorm_out_channels',
    'prune_instancenorm_in_channels',
]


class BasePruningFunc(ABC):
    TARGET_MODULES = ops.TORCH_OTHERS

    def __init__(self, pruning_dim=1):
        self.pruning_dim = pruning_dim

    # 定义对输入通道、输出通道进行剪枝的抽象方法
    @abstractclassmethod
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def get_out_channels(self, layer: nn.Module):
        raise NotImplementedError

    @abstractclassmethod
    def get_in_channels(self, layer: nn.Module):
        raise NotImplementedError

    # 检查需要剪枝通道的索引是否合法
    def check(self, layer, idxs, to_output):
        # 验证目标模块的合法性
        if self.TARGET_MODULES is not None:
            assert isinstance(layer, self.TARGET_MODULES), 'Mismatched pruner {} and module {}'.format(
                self.__str__, layer)
        # to_output表示对输出通道进行剪枝
        if to_output:
            # 获取可剪枝的通道数
            prunable_channels = self.get_out_channels(layer)
        else:
            prunable_channels = self.get_in_channels(layer)
        # 验证idxs是否合法
        if prunable_channels is not None:
            assert all(idx < prunable_channels and idx >= 0 for idx in
                       idxs), "All pruning indices should fall into [{}, {})".format(0, prunable_channels)

    # 调用，进行剪枝
    def __call__(self, layer: nn.Module, idxs: Sequence[int], to_output: bool = True, inplace: bool = True,
                 dry_run: bool = False) -> Tuple[nn.Module, int]:
        idxs.sort()
        self.check(layer, idxs, to_output)
        # 获取剪枝函数
        pruning_fn = self.prune_out_channels if to_output else self.prune_in_channels
        # 如果非原地进行剪枝，则进行深拷贝
        # Todo: 聚合前的选择传输可以认为是拷贝后的剪枝
        if not inplace:
            layer = deepcopy(layer)
        # 执行剪枝，并返回剪枝后的层
        layer = pruning_fn(layer, idxs)
        return layer

    # 对参数和梯度进行剪枝
    # 这里keep_idxs给出了需要保留的参数的索引
    def _prune_parameter_and_grad(self, weight, keep_idxs, pruning_dim):
        pruned_weight = torch.nn.Parameter(
            torch.index_select(weight, pruning_dim, torch.LongTensor(keep_idxs).to(weight.device))
        )
        # 此外，还需要保存梯度
        if weight.grad is not None:
            pruned_weight.grad = torch.index_select(weight.grad, pruning_dim,
                                                    torch.LongTensor(keep_idxs).to(weight.device))
        return pruned_weight.to(weight.device)


class ConvPruner(BasePruningFunc):
    # 该剪枝器的目标模块是TORCH_CONV
    TARGET_MODULES = ops.TORCH_CONV

    # Todo: 对输出通道的剪枝，是在dim=0上进行
    #  Why? Debug一下，看看卷积层参数的形状
    def prune_out_channels(self, layer: nn.Module, masks, idxs: Sequence[int]):
        # 获取需要保留的通道索引
        # keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        # keep_idxs.sort()
        # # 计算剪枝后的输出通道的数量
        # layer.out_channels = layer.out_channels - len(idxs)
        # if not layer.transposed:
        #     layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        # else:
        #     layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        # if layer.bias is not None:
        #     layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        # return layer
        # 卷积层的形状 (out_channels,in_channels,k,k)
        masks[0][idxs] = 0
        if len(masks) > 1:
            masks[1][idxs] = 0

    def prune_in_channels(self, layer: nn.Module, masks, idxs: Sequence[int]):
        # keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        # keep_idxs.sort()
        # layer.in_channels = layer.in_channels - len(idxs)
        # # Todo: 输入可以进行分组
        # if layer.groups > 1:
        #     keep_idxs = keep_idxs[:len(keep_idxs) // layer.groups]
        # if not layer.transposed:
        #     layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        # else:
        #     layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        # # 不对偏置进行剪枝因为它并不改变输出通道
        # return layer
        masks[0][:, idxs] = 0

    def get_out_channels(self, layer: nn.Module):
        return layer.out_channels

    def get_in_channels(self, layer: nn.Module):
        return layer.in_channels


# 对线性层的剪枝器
class LinearPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def prune_out_channels(self, layer: nn.Module, masks, idxs: Sequence[int]):
        # keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        # keep_idxs.sort()
        # layer.out_features = layer.out_features - len(idxs)
        # layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        # if layer.bias is not None:
        #     layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        # 返回剪枝后的层
        # Todo: 对输出通道进行剪枝
        # 注意，权重矩阵的形状为(out_features,in_features)
        masks[0][idxs] = 0
        if len(masks) > 1:
            masks[1][idxs] = 0

    def prune_in_channels(self, layer: nn.Module, masks, idxs: Sequence[int]):
        # keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        # keep_idxs.sort()
        # layer.in_features = layer.in_features - len(idxs)
        # layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        # return layer
        # 对输入通道的剪枝是在维度1上进行，且对输入通道的剪枝不涉及对偏置矩阵的剪枝

        masks[0][:, idxs] = 0

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


# 对BatchNorm层的剪枝
class BatchNormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_BATCHNORM

    def prune_out_channels(self, layer: nn.Module, masks, idxs: Sequence[int]):
        # keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        # keep_idxs.sort()
        # layer.num_features = layer.num_features - len(idxs)
        # # 只保留未剪枝索引的统计数据
        # layer.running_mean = layer.running_mean.data[keep_idxs]
        # layer.running_var = layer.running_var[keep_idxs]
        # # 如果需要经过仿射，则进行参数和梯度的剪枝
        # if layer.affine:
        #     layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        #     layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        # return layer
        masks[0][idxs] = 0
        if len(masks) > 1:
            masks[1][idxs] = 0

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


# 参数剪枝器
class ParameterPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER

    def __init__(self, pruning_dim=-1):
        super().__init__(pruning_dim=pruning_dim)

    def prune_out_channels(self, tensor: nn.Module, idxs: Sequence[int]):
        keep_idxs = list(set(range(tensor.data.shape[self.pruning_dim])))
        keep_idxs.sort()
        pruned_parameter = self._prune_parameter_and_grad(tensor, keep_idxs, self.pruning_dim)
        return pruned_parameter

    prune_in_channels = prune_out_channels

    def get_out_channels(self, parameter):
        return parameter.shape[self.pruning_dim]

    def get_in_channels(self, parameter):
        return parameter.shape[self.pruning_dim]


# Todo: Depthwise修饰的卷积层有什么特点吗？
class DepthwiseConvPruner(ConvPruner):
    TARGET_MODULES = ops.TORCH_CONV

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()

        layer.out_channels = layer.out_channels - len(idxs)
        # 这里层的输入通道数也会发生变化
        layer.in_channels = layer.in_channels - len(idxs)
        layer.groups = layer.groups - len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels


# Todo: 其他高阶网络结构的剪枝器

class LayernormPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LAYERNORM

    def __init__(self, metrcis=None, pruning_dim=-1):
        super().__init__(metrcis)
        self.pruning_dim = pruning_dim

    def check(self, layer, idxs):
        layer.dim = self.pruning_dim

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        pruning_dim = self.pruning_dim
        if len(layer.normalized_shape) < -pruning_dim:
            return layer
        num_features = layer.normalized_shape[pruning_dim]
        keep_idxs = torch.tensor(list(set(range(num_features)) - set(idxs)))
        keep_idxs.sort()
        if layer.elementwise_affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, pruning_dim)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, pruning_dim)
        if pruning_dim != -1:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (
                keep_idxs.size(0),) + layer.normalized_shape[pruning_dim + 1:]
        else:
            layer.normalized_shape = layer.normalized_shape[:pruning_dim] + (
                keep_idxs.size(0),)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]


class GroupNormPruner(BasePruningFunc):
    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.num_channels)) - set(idxs))
        keep_idxs.sort()
        layer.num_channels = layer.num_channels - len(idxs)
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_channels

    def get_in_channels(self, layer):
        return layer.num_channels


class InstanceNormPruner(BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        keep_idxs.sort()
        layer.num_features = layer.num_features - len(idxs)
        if layer.affine:
            layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.num_features

    def get_in_channels(self, layer):
        return layer.num_features


class PReLUPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_PRELU

    def prune_out_channels(self, layer: nn.PReLU, idxs: list) -> nn.Module:
        if layer.num_parameters == 1:
            return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        keep_idxs.sort()
        layer.num_parameters = layer.num_parameters - len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer:nn.Module, idxs: Sequence[int]) -> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        if layer.num_parameters == 1:
            return None
        else:
            return layer.num_parameters

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)


class EmbeddingPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_EMBED

    def prune_out_channels(self, layer: nn.Embedding, idxs: list) -> nn.Module:
        num_features = layer.embedding_dim
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        layer.embedding_dim = len(keep_idxs)
        return layer

    prune_in_channels = prune_out_channels

    # def prune_in_channels(self, layer: nn.Embedding, idxs: list)-> nn.Module:
    #    return self.prune_out_channels(layer=layer, idxs=idxs)

    def get_out_channels(self, layer):
        return layer.embedding_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer=layer)


class LSTMPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LSTM

    def prune_out_channels(self, layer: nn.LSTM, idxs: list) -> nn.Module:
        assert layer.num_layers == 1
        num_layers = layer.num_layers
        num_features = layer.hidden_size
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        keep_idxs = torch.tensor(keep_idxs)
        expanded_keep_idxs = torch.cat([keep_idxs + i * num_features for i in range(4)], dim=0)
        if layer.bidirectional:
            postfix = ['', '_reverse']
        else:
            postfix = ['']
        # for l in range(num_layers):
        for pf in postfix:
            setattr(layer, 'weight_hh_l0' + pf, self._prune_parameter_and_grad(
                getattr(layer, 'weight_hh_l0' + pf), keep_idxs, 0))
            if layer.bias:
                setattr(layer, 'bias_hh_l0' + pf, self._prune_parameter_and_grad(
                    getattr(layer, 'bias_hh_l0' + pf), keep_idxs, 0))
            setattr(layer, 'weight_hh_l0' + pf, self._prune_parameter_and_grad(
                getattr(layer, 'weight_hh_l0' + pf), keep_idxs, 0))
            setattr(layer, 'weight_ih_l0' + pf, self._prune_parameter_and_grad(
                getattr(layer, 'weight_ih_l0' + pf), expanded_keep_idxs, 1))
            if layer.bias:
                setattr(layer, 'bias_ih_l0' + pf, self._prune_parameter_and_grad(
                    getattr(layer, 'bias_ih_l0' + pf), keep_idxs, 0))
        layer.hidden_size = len(keep_idxs)

    def prune_in_channels(self, layer: nn.LSTM, idxs: list):
        num_features = layer.input_size
        keep_idxs = list(set(range(num_features)) - set(idxs))
        keep_idxs.sort()
        setattr(layer, 'weight_ih_l0', self._prune_parameter_and_grad(
            getattr(layer, 'weight_ih_l0'), keep_idxs, 1))
        if layer.bidirectional:
            setattr(layer, 'weight_ih_l0_reverse', self._prune_parameter_and_grad(
                getattr(layer, 'weight_ih_l0_reverse'), keep_idxs, 1))
        layer.input_size = len(keep_idxs)

    def get_out_channels(self, layer):
        return layer.hidden_size

    def get_in_channels(self, layer):
        return layer.input_size


class MultiheadAttentionPruner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_MHA

    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        assert (layer.embed_dim - len(
            idxs)) % layer.num_heads == 0, "embed_dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)" % (
            layer.embed_dim, layer.num_heads)

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.embed_dim)) - set(idxs))
        keep_idxs.sort()

        if layer.q_proj_weight is not None:
            layer.q_proj_weight = self._prune_parameter_and_grad(layer.q_proj_weight, keep_idxs, 0)
        if layer.k_proj_weight is not None:
            layer.k_proj_weight = self._prune_parameter_and_grad(layer.k_proj_weight, keep_idxs, 0)
        if layer.v_proj_weight is not None:
            layer.v_proj_weight = self._prune_parameter_and_grad(layer.v_proj_weight, keep_idxs, 0)

        pruning_idxs_repeated = idxs + \
                                [i + layer.embed_dim for i in idxs] + \
                                [i + 2 * layer.embed_dim for i in idxs]
        keep_idxs_3x_repeated = list(
            set(range(3 * layer.embed_dim)) - set(pruning_idxs_repeated))
        keep_idxs_3x_repeated.sort()
        if layer.in_proj_weight is not None:
            layer.in_proj_weight = self._prune_parameter_and_grad(layer.in_proj_weight, keep_idxs_3x_repeated, 0)
            layer.in_proj_weight = self._prune_parameter_and_grad(layer.in_proj_weight, keep_idxs, 1)
        if layer.in_proj_bias is not None:
            layer.in_proj_bias = self._prune_parameter_and_grad(layer.in_proj_bias, keep_idxs_3x_repeated, 0)

        if layer.bias_k is not None:
            layer.bias_k = self._prune_parameter_and_grad(layer.bias_k, keep_idxs, 2)
        if layer.bias_v is not None:
            layer.bias_v = self._prune_parameter_and_grad(layer.bias_v, keep_idxs, 2)

        linear = layer.out_proj
        keep_idxs = list(set(range(linear.out_features)) - set(idxs))
        keep_idxs.sort()
        linear.out_features = linear.out_features - len(idxs)
        linear.weight = self._prune_parameter_and_grad(linear.weight, keep_idxs, 0)
        if linear.bias is not None:
            linear.bias = self._prune_parameter_and_grad(linear.bias, keep_idxs, 0)
        keep_idxs = list(set(range(linear.in_features)) - set(idxs))
        keep_idxs.sort()
        linear.in_features = linear.in_features - len(idxs)
        linear.weight = self._prune_parameter_and_grad(linear.weight, keep_idxs, 1)
        layer.embed_dim = layer.embed_dim - len(idxs)
        layer.head_dim = layer.embed_dim // layer.num_heads
        layer.kdim = layer.embed_dim
        layer.vdim = layer.embed_dim
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.embed_dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)


PrunerBox = {
    ops.OPTYPE.CONV: ConvPruner(),
    ops.OPTYPE.LINEAR: LinearPruner(),
    ops.OPTYPE.BN: BatchNormPruner(),
    ops.OPTYPE.PARAMETER: ParameterPruner(),
    # 高阶的剪枝器
    ops.OPTYPE.DEPTHWISE_CONV: DepthwiseConvPruner(),
    ops.OPTYPE.PRELU: PReLUPruner(),
    ops.OPTYPE.LN: LayernormPruner(),
    ops.OPTYPE.EMBED: EmbeddingPruner(),
    ops.OPTYPE.MHA: MultiheadAttentionPruner(),
    ops.OPTYPE.LSTM: LSTMPruner(),
    ops.OPTYPE.GN: GroupNormPruner(),
    ops.OPTYPE.IN: InstanceNormPruner(),
}

# 下面导出剪枝函数的别名
prune_conv_out_channels = PrunerBox[ops.OPTYPE.CONV].prune_out_channels
# 卷积层输入通道的剪枝器
prune_conv_in_channels = PrunerBox[ops.OPTYPE.CONV].prune_in_channels

# 对batchnorm输出通道和输入通道的剪枝
# 即batchnorm层前或后的神经元剪枝
prune_batchnorm_out_channels = PrunerBox[ops.OPTYPE.BN].prune_out_channels
prune_batchnorm_in_channels = PrunerBox[ops.OPTYPE.BN].prune_in_channels

# 对线性层输出和输入通道的剪枝
prune_linear_out_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_out_channels
prune_linear_in_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_in_channels

prune_parameter_out_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_out_channels
prune_parameter_in_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_in_channels

prune_depthwise_conv_out_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_out_channels
prune_depthwise_conv_in_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_in_channels

# Todo: prelu
prune_prelu_out_channels = PrunerBox[ops.OPTYPE.PRELU].prune_out_channels
prune_prelu_in_channels = PrunerBox[ops.OPTYPE.PRELU].prune_in_channels

# 对layernorm的剪枝
prune_layernorm_out_channels = PrunerBox[ops.OPTYPE.LN].prune_out_channels
prune_layernorm_in_channels = PrunerBox[ops.OPTYPE.LN].prune_in_channels

prune_embedding_out_channels = PrunerBox[ops.OPTYPE.EMBED].prune_out_channels
prune_embedding_in_channels = PrunerBox[ops.OPTYPE.EMBED].prune_in_channels

# 对多头注意力的剪枝
prune_multihead_attention_out_channels = PrunerBox[ops.OPTYPE.MHA].prune_out_channels
prune_multihead_attention_in_channels = PrunerBox[ops.OPTYPE.MHA].prune_in_channels

prune_lstm_out_channels = PrunerBox[ops.OPTYPE.LSTM].prune_out_channels
prune_lstm_in_channels = PrunerBox[ops.OPTYPE.LSTM].prune_in_channels

prune_groupnorm_out_channels = PrunerBox[ops.OPTYPE.GN].prune_out_channels
prune_groupnorm_in_channels = PrunerBox[ops.OPTYPE.GN].prune_in_channels

prune_instancenorm_out_channels = PrunerBox[ops.OPTYPE.IN].prune_out_channels
prune_instancenorm_in_channels = PrunerBox[ops.OPTYPE.IN].prune_in_channels
