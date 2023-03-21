import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import argparse
from collections import OrderedDict, namedtuple
from functools import reduce, partial, update_wrapper
import logging
import os
from copy import deepcopy
import warnings

import federatedml.nn.backend.distiller.utils as utils
from .quantizer import Quantizer, QBits
from .q_utils import *

import torch.quantization
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq

import federatedml.nn.backend.distiller.modules as modules

quant_logger = logging.getLogger('quant_logger')


# 将量化参数转为字符串，使得可读更高
def _quant_param_to_str(val):
    """将数值参数量化为字符串

    :param val: 数值参数，tensor或者实数
    :return: 量化后的字符串
    """
    if isinstance(val, torch.Tensor):
        # 如果是列表，则直接返回‘PerCh’
        if val.numel() > 1:
            return 'PerCh'
        else:
            return '{:.6f}'.format(val.item())
    return '{:.6f}'.format(val)


def _enum_to_str(enum_val):
    """将枚举类型转为字符串

    :param enum_val: 枚举值
    :return:
    """
    if isinstance(enum_val, str):
        return enum_val
    # Todo: enum_val的示例是怎样的？为什么需要用'.'分割
    return str(enum_val).split('.')[1]


# Todo: 这个是继承吗？trick
class ModuleQuantMode(namedtuple('ModuleQuantMode', ['activations', 'weights'])):
    def __new__(cls, activations, weights):
        if not isinstance(activations, LinearQuantMode) or not isinstance(weights, LinearQuantMode):
            raise ValueError('ModuleQuantMode must receive LinearQuantMode values')
        return super(ModuleQuantMode, cls).__new__(cls, activations, weights)


# Todo: 剪切模式在哪里使用？
# 定义剪切模式，其属于一个枚举类
class ClipMode(Enum):
    # 不使用剪切，使用绝对值的min/max
    NONE = 0
    # 通过平均一个批次内样本的最大绝对值作为剪切值
    AVG = 1
    # 使用tensor+N个标准差的平均值计算剪切值，N应该单独指定
    N_STD = 2
    # Todo: 暂时不考虑ACIQ剪切模式
    GAUSS = 3
    LAPLACE = 4


# 验证给定的值val是否在enum_cls中或是否属于enum_cls实例
def _verify_enum_value(val, enum_cls):
    cls_name = enum_cls.__name__
    if isinstance(val, str):
        try:
            return enum_cls[val]
        except KeyError:
            raise ValueError("Input string '{0}' doesn't match any of the values of {1}: {2}"
                             .format(val, cls_name, [e.name for e in enum_cls]))
    elif isinstance(val, enum_cls):
        return val
    else:
        raise TypeError("Argument can be either a string or member of {0} (got {1})".format(cls_name, val))


# Todo: 根据具体的示例结合理解
def verify_quant_mode(mode):
    if isinstance(mode, ModuleQuantMode):
        return mode
    if isinstance(mode, dict):
        # Todo: 里面存的是key吗？对应_verify中的第一种情况
        acts = _verify_enum_value(mode['activations'], LinearQuantMode)
        wts = _verify_enum_value(mode['weights'], LinearQuantMode)
    else:
        acts = wts = _verify_enum_value(mode, LinearQuantMode)
    return ModuleQuantMode(acts, wts)


def verify_clip_mode(mode):
    return _verify_enum_value(mode, ClipMode)


def _get_saturation_fn(quant_mode, clip_mode, num_stds, num_bits=None):
    """根据量化模式，剪切模式以及标准差数据判断计算饱和值的函数

    :param quant_mode: 量化模式
    :param clip_mode: 剪切模式
    :param num_stds: 标准差的数量
    :param num_bits: 量化比特数
    :return:
    """
    if is_linear_quant_mode_symmetric(quant_mode):
        # 对称: max_abs
        fns = {ClipMode.NONE: get_tensor_max_abs,
               ClipMode.AVG: get_tensor_avg_max_abs,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_max_abs, n_stds=num_stds)}
    else:
        # 非对称: min_max
        fns = {ClipMode.NONE: get_tensor_min_max,
               ClipMode.AVG: get_tensor_avg_min_max,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_min_max, n_stds=num_stds)}
    # 根据clip_mode返回具体的修剪函数
    return fns[clip_mode]


class UnsatisfiedRequirements(Exception):
    pass


def _get_quant_params_from_tensor(tensor, num_bits, mode=LinearQuantMode.SYMMETRIC, clip=ClipMode.NONE,
                                  per_channel=False, num_stds=None,
                                  half_range=False, scale_approx_mult_bits=None):
    # per_channel为True时的要求
    if per_channel and tensor.dim() not in [2, 4]:
        raise UnsatisfiedRequirements('Per channel quantization possible only with '
                                      '2D or 4D tensors (linear or conv layer weights)')
    # ClipMode.N_STD时的一些要求
    if clip == ClipMode.N_STD:
        if per_channel:
            raise ValueError('N_STD clipping not supported with per-channel quantization')
        if num_stds is None:
            raise UnsatisfiedRequirements('Clip mode set top N_STD but \'num_stds\' parameter not provided')
    # 如果是AVG剪切模式或者per_channel为True
    dim = 0 if clip == ClipMode.AVG or per_channel else None
    sat_fn = _get_saturation_fn(mode, clip, num_stds, num_bits)
    # 如果是对称量化
    if is_linear_quant_mode_symmetric(mode):
        sat_val = sat_fn(tensor, dim)
        if isinstance(sat_val, tuple):
            assert len(sat_val) == 2
            # 对称情况下，只需要max即可
            sat_val = torch.max(*sat_val)
        # 获取对称线性量化参数
        scale, zp = symmetric_linear_quantization_params(num_bits, sat_val,
                                                         restrict_qrange=mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
    else:
        # half_range是GAUSS和LAPLACE模式下需要的参数
        sat_min, sat_max = sat_fn(tensor, dim) if clip not in [ClipMode.GAUSS, ClipMode.LAPLACE] \
            else sat_fn(tensor, dim, half_range=half_range)
        # 获取符号标志
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)
    if per_channel:
        # 对scale和zero_points进行reshape，使得其可以同权重向量正确广播
        dims = [scale.shape[0]] + [1] * (tensor.dim() - 1)
        scale = scale.view(dims)
        zp = zp.view(dims)
    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)
    return scale, zp


# 从已经保存的状态字典中得到量化的参数
# Todo: 注意stats中值包含min、max、mean等信息
def _get_quant_params_from_stats_dict(stats, num_bits, mode, clip=ClipMode.NONE, num_stds=None, half_range=False,
                                      scale_approx_mult_bits=None):
    if clip == ClipMode.N_STD:
        if num_stds is None:
            raise ValueError('Clip mode set to N_STD but \'num_stds\' parameter not provided')
        if num_stds <= 0:
            raise ValueError('n_stds must be > 0, got {}'.format(num_stds))
    # 根据ClipMode修改前缀
    prefix = 'avg_' if clip == ClipMode.AVG else ''
    sat_min = torch.tensor(float(stats[prefix + 'min']))
    sat_max = torch.tensor(float(stats[prefix + 'max']))
    if clip == ClipMode.N_STD:
        mean = torch.tensor(float(stats['mean']))
        std = torch.tensor(float(stats['std']))
        sat_min = torch.max(sat_min, mean - num_stds * std)
        sat_max = torch.min(sat_max, mean + num_stds * std)
    # Todo: 先不考虑LAPLACE以及GAUSS模式
    # elif clip in (ClipMode.LAPLACE, ClipMode.GAUSS):
    #     clip = AciqClipper.AciqClippingType.Laplace if clip == ClipMode.LAPLACE else AciqClipper.AciqClippingType.Gauss
    #     if is_linear_quant_mode_symmetric(mode):
    #         sat_min, sat_max = AciqSymmetricClipper(num_bits, clip)(stats)
    #     else:
    #         sat_min, sat_max = AciqAsymmetricClipper(num_bits, clip)(stats, half_range=half_range)
    if is_linear_quant_mode_symmetric(mode):
        scale, zp = symmetric_linear_quantization_params(num_bits, torch.max(sat_min.abs_(), sat_max.abs_()),
                                                         restrict_qrange=mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
    else:
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)
    # Todo: 暂时不看近似的方法
    if scale_approx_mult_bits is not None:
        scale = approx_scale_as_mult_and_shift(scale, scale_approx_mult_bits)

    return scale, zp


# 和get_quant_param是一个互逆的过程
def _get_clipping_value(scale, zp, num_bits, mode):
    """
    通过解量化得到量化参数诱导的饱和值
    Args:
        scale, zp (torch.Tensor or float): 量化参数
        num_bits (int): 量化后的比特数目
        mode (LinearQuantMode): 量化的模型
    Returns:
        min, max : tuple[float, float]
    """
    device = scale.device if isinstance(scale, torch.Tensor) else 'cpu'
    # 非对称量化模型，场景是对relu后的激活单元进行量化
    if is_linear_quant_mode_asymmetric(mode):
        t = torch.tensor([0, 2 ** num_bits - 1], device=device)
    else:
        # 这里对称量化并没有对量化范围进行限制，限制后应该为:-(2 ** num_bits - 1) / 2
        t = torch.tensor([-2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1], device=device)
    # 对已经量化的值进行解量化
    sat_min, sat_max = linear_dequantize(t, scale, zp)  # type: torch.Tensor
    return sat_min, sat_max


# Todo: 下一部分是 Post Training

# 定义量化张量的元数据类
class TensorQuantMetadata(namedtuple('TensorQuantMetadata', ['scale', 'zero_point', 'min_q_val', 'max_q_val'])):
    __slots__ = ()

    # 输出人类易读的方法
    def __str__(self):
        return '(scale={} ; zero_point={})'.format(_quant_param_to_str(self.scale),
                                                   _quant_param_to_str(self.zero_point))


# 定义量化设置类
# Todo: 与量化张量元数据类的关系：QuantSetting经过计算可以得到Metadata
class QuantSettings(object):
    def __init__(self, num_bits, quant_mode, clip_mode, clip_n_stds, clip_half_range, per_channel):
        self.num_bits = num_bits
        self.quant_mode = quant_mode
        self.clip_mode = clip_mode
        self.clip_n_stds = clip_n_stds
        self.clip_half_range = clip_half_range
        self.per_channel = per_channel

    def __str__(self):
        return '(num_bits={} ; quant_mode={} ; clip_mode={} ; clip_n_stds={} ; clip_half_range={}' \
               ' ; per_channel={})'.format(self.num_bits, _enum_to_str(self.quant_mode),
                                           _enum_to_str(self.clip_mode), self.clip_n_stds, self.clip_half_range,
                                           self.per_channel)


def linear_quantize_clamp_with_metadata(t, inplace=False):
    assert hasattr(t, 'quant_metadata')
    qmd = t.quant_metadata
    t = linear_quantize_clamp(t, *qmd, inplace)
    # 如果不是原地操作，则对象t中的qmd信息丢失，需要重新赋值
    if not inplace:
        t.quant_metadata = qmd
    return t


def linear_dequantize_with_metadata(t, inplace=False):
    assert hasattr(t, 'quant_metadata')
    qmd = t.quant_metadata
    t = linear_dequantize(t, qmd.scale, qmd.zero_point, inplace)
    if not inplace:
        t.quant_metadata = qmd
    return t


# 对要剪切的值进行检查
def _check_clipping_val(val, quant_mode, half_range):
    if isinstance(val, float):
        if is_linear_quant_mode_symmetric(quant_mode):
            return -val, val
        elif half_range:
            return 0, val
        raise ValueError('For asymmetric quantization, setting clipping values only allowed '
                         'using both min/max values.')
    if isinstance(val, (tuple, list, np.ndarray, torch.Tensor)):
        assert all(utils.is_scalar(v) for v in val), 'Elements of the clipping value must be scalar-like.'
        assert len(val) == 2, 'Clipping value must have 2 elements.'
        return tuple(val)
    raise TypeError('Clipping value should be a scalar or an iterable of these')


class RangeLinearQuantWrapper(nn.Module):
    """
        使用基于范围的线性量化函数的包装基类
        Args:
            wrapped_module (torch.nn.Module): 需要包装的模型
            num_bits_acts (int): 量化输入和输出使用的比特数量
            num_bits_accum (int): 分配给中间整型结果的累加器的比特数量,一般比num_bits_acts要大，否则容易发生溢出现象
            mode (ModuleQuantMode / LinearQuantMode): 使用的量化模式 (symmetric / asymmetric-signed / unsigned)，也可以是字典
            clip_acts (ClipMode): 使用的激活单元剪切模型
            activation_stats (dict): 包含激活单元状态的字典，用于量化参数的静态计算
                                    字典应该是distiller.data_loggers.QuantCalibrationStatsCollector导出的格式
                                    如果是该属性为None，则表示动态计算量化参数
            # Todo: 高阶可选参数，暂时不使用。
            clip_n_stds (float): When clip_acts == ClipMode.N_STD, this is the number of standard deviations to use
            clip_half_range (bool): use half range clipping.
                NOTE - this only works with ACIQ clip modes i.e. GAUSS and LAPLACE
            scale_approx_mult_bits (int): If not None, scale factors will be approximated using an integer multiplication
                followed by a bit-wise shift. This eliminates floating-point scale factors, replacing them with integer
                calculations.
                If None, scale factors will be kept in their original FP32 values.
        """

    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32, mode=LinearQuantMode.SYMMETRIC,
                 clip_acts=ClipMode.NONE, activation_stats=None, clip_n_stds=None, clip_half_range=False,
                 scale_approx_mult_bits=None, input_overrides=None, inputs_quant_auto_fallback=False):
        super(RangeLinearQuantWrapper, self).__init__()
        input_overrides = input_overrides or OrderedDict()

        mode = verify_quant_mode(mode)
        self.mode = mode

        self.wrapped_module = wrapped_module
        self.clip_half_range = clip_half_range
        self.scale_approx_mult_bits = scale_approx_mult_bits
        self.inputs_quant_auto_fallback = inputs_quant_auto_fallback

        # 创建一些设置对象
        self.output_quant_settings = QuantSettings(num_bits_acts, mode.activations, clip_acts, clip_n_stds,
                                                   clip_half_range, False)
        self.accum_quant_settings = QuantSettings(num_bits_accum, LinearQuantMode.SYMMETRIC,
                                                  ClipMode.NONE, None, False, False)
        self.preset_act_stats = False

        # 注册buffer
        self.register_buffer('num_forwards', torch.zeros(1, dtype=torch.long))
        self.register_buffer('force_readjust', torch.tensor(False))

        # 获取累加器的量化范围
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True,
                                                                         signed_restrict_qrange=False)
        # 如果不需要量化激活单元，则直接return
        if num_bits_acts is None:
            return
        # 设置输入的量化设置
        self.inputs_quant_settings_overrides = OrderedDict()
        for k, v in input_overrides.items():
            idx = int(k)

            # Todo: 关于此处的input和output的含义
            # 将output的设置应用到input上
            if v.pop('from.output', None):
                quant_settings = deepcopy(self.output_quant_settings)
                quant_settings.clip_half_range = False
            else:
                # 从value中取出对应的设置值
                quant_settings = QuantSettings(
                    v.pop('bits_activations', self.output_quant_settings.num_bits),
                    verify_quant_mode(v.pop('mode', self.output_quant_settings.quant_mode)),
                    verify_clip_mode(v.pop('clip_acts', self.output_quant_settings.clip_mode)),
                    v.pop('clip_n_stds', self.output_quant_settings.clip_n_stds),
                    False, False
                )
            self.inputs_quant_settings_overrides[idx] = quant_settings
        # 该字段控制前向操作结束时输出是否解量化
        # 如果设置为False，则返回量化后的输出，但不返回任何的量化参数
        self._dequant_out = True
        signed = mode.activations != LinearQuantMode.ASYMMETRIC_UNSIGNED
        restrict_qrange = mode.activations == LinearQuantMode.SYMMETRIC_RESTRICTED
        # 获取激活单元/输出的量化范围
        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=signed,
                                                                       signed_restrict_qrange=restrict_qrange)
        # 如果存在预设的字典
        if activation_stats:
            self.preset_act_stats = True

            # Todo: 计算输入的备用设置
            self.inputs_quant_metadata_fallback = OrderedDict()
            for idx, stats in activation_stats['inputs'].items():
                settings = self.inputs_quant_settings_overrides.get(idx, self.output_quant_settings)
                # 计算量化参数scale和zp
                scale, zp = _get_quant_params_from_stats_dict(stats, settings.num_bits, settings.quant_mode,
                                                              settings.clip_mode,
                                                              settings.clip_n_stds, settings.clip_half_range,
                                                              self.scale_approx_mult_bits)
                # 获取量化范围
                min_q_val, max_q_val = get_quantized_range(
                    settings.num_bits, settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED,
                                       settings.quant_mode != LinearQuantMode.SYMMETRIC_RESTRICTED
                )
                # 构造量化元数据
                qmd = TensorQuantMetadata(scale, zp, min_q_val, max_q_val)
                self.inputs_quant_metadata_fallback[idx] = qmd

            # 计算输出量化参数
            scale, zp = _get_quant_params_from_stats_dict(activation_stats['output'], num_bits_acts, mode.activations,
                                                          clip_acts, clip_n_stds, clip_half_range,
                                                          scale_approx_mult_bits)
            if not isinstance(scale, torch.Tensor):
                scale, zp = torch.tensor(scale), torch.tensor(zp)
            # 注册输出的量化参数buffer
            self.register_buffer('output_scale', scale)
            self.register_buffer('output_zero_point', zp)
        else:
            self.preset_act_stats = False

    # 定义输出量化参数的生成器,返回的是__init__()函数中注册的buffer
    # Todo: 这里filter的含义是什么？
    def named_linear_quant_params(self, filter=False):
        if self.output_quant_settings.num_bits is not None and self.preset_act_stats:
            # 在__init__()函数中注册的buffer
            yield 'output_scale', self.output_scale
            if not filter or (is_linear_quant_mode_asymmetric(self.mode.activations) and not self.clip_half_range):
                yield 'output_zero_point', self.output_zero_point

    def _check_requirements_output_clipping(self):
        if not self.output_quant_settings.num_bits:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values because '
                                          'the activations aren\'t quantized.')
        if not self.preset_act_stats:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values '
                                          'because the activations stats were not provided.')

    @property
    def output_clipping(self):
        self._check_requirements_output_clipping()
        bits = self.output_quant_settings.num_bits
        scale, zp = self.output_scale, self.output_zero_point
        return _get_clipping_value(scale, zp, bits, self.output_quant_settings.quant_mode)

    @output_clipping.setter
    def output_clipping(self, val):
        """从所传值val中计算得到val_min和val_max，然后计算量化参数并进行设置

        :param val:
        :return:
        """
        self._check_requirements_output_clipping()
        qset = self.output_quant_settings
        val_min, val_max = _check_clipping_val(val, qset.quant_mode, self.clip_half_range)
        # 对qset的相关属性进行重置
        qset.clip_mode, qset.clip_half_range, qset.clip_n_stds = ClipMode.NONE, None, None
        scale, zp = _get_quant_params_from_stats_dict({'min': val_min, 'max': val_max}, qset.num_bits, qset.quant_mode,
                                                      scale_approx_mult_bits=self.scale_approx_mult_bits)
        self.set_linear_quant_param('output_scale', scale.item())
        self.set_linear_quant_param('output_zero_point', zp.item())

    def named_clipping(self, filter=False):
        val = self.output_clipping
        if filter and (is_linear_quant_mode_symmetric(self.mode.activations) or self.clip_half_range):
            val = val[1]
        yield 'output_clipping', val

    def set_linear_quant_param(self, name, val):
        if name in dict(self.named_clipping()):
            setattr(self, name, val)
        elif name not in dict(self.named_linear_quant_params()):
            raise ValueError('%s is not a quantization parameter.' % name)
        else:
            getattr(self, name).data.fill_(val)
        self.force_readjust.fill_(True)

    def update_linear_quant_params(self, new_config):
        """更新配置字典中所有的量化参数

        :param new_config: 新的配置字典
        :return:
        """
        for name, val in new_config.items():
            self.set_linear_quant_param(name, val)

    # noinspection PyArgumentList
    def forward(self, *inputs):
        # 属于Post-Training Quantization，仅可用于评估模式
        if self.training:
            raise RuntimeError(self.__class__.__name__ + " can only be used in eval mode")
        if self.output_quant_settings.num_bits is None:
            # 不需要进行量化，则使用包装模块计算输出
            out = self.wrapped_module(*inputs)
            if self.clip_half_range:
                out = f.relu(out)
            return out
        device = inputs[0].device
        # 将buffer中的配置设置为属性，并移动到输入的计算设备上
        # Todo: 为什么不一开始就设置为属性呢？attr和_buffer的区别？
        for buffer_name, buffer in self._buffers.items():
            setattr(self, buffer_name, buffer.to(device))
        # 处理得到量化后的输入
        inputs_q = [self._prepare_input(idx, input) for idx, input in enumerate(inputs)]

        accum = self.quantized_forward(*inputs_q)

        # Todo: clip_half_range在神经网络情景中是执行relu的标志
        if self.clip_half_range:
            accum = f.relu(accum)

        # Todo: 注意理解下面多组量化参数的含义
        out_scale, out_zero_point = self.get_output_quantization_params(accum)
        requant_scale, requant_zero_point = self.get_accum_to_output_re_quantization_params(out_scale, out_zero_point)
        # 执行量化，得到量化后的输出
        out_q = linear_quantize_clamp(accum.data, requant_scale, requant_zero_point,
                                      self.acts_min_q_val, self.acts_max_q_val, inplace=True)
        if not self._dequant_out:
            return torch.autograd.Variable(out_q)
        # 进行解量化的一些操作
        out_f = linear_dequantize(out_q, out_scale, out_zero_point, inplace=True)
        out_f.quant_metadata = TensorQuantMetadata(out_scale, out_zero_point, self.acts_min_q_val, self.acts_max_q_val)
        # 前向传播数目+1
        self.num_forwards += 1
        return out_f

    def _prepare_input(self, idx, input):
        """默认实现是量化输入向量，这可以用于除了RangeLinearFakeQuantWrapper之外的所有Wrapper类

        :param idx: 输入索引
        :param input: 输入数据
        :return: 量化后的输入数据
        """
        input.quant_metadata = self._get_input_quant_metadata(idx, input)
        # 执行量化操作
        return linear_quantize_clamp_with_metadata(input, inplace=False)

    def _get_input_quant_metadata(self, idx, input):
        if hasattr(input, 'quant_metadata'):
            if idx in self.inputs_quant_settings_overrides:
                raise RuntimeError('<{}> Input {}: CONFLICT - Tensor has embedded quantization metadata AND user '
                                   'defined input quantization settings'.format(self.distiller_name, idx))
            qmd = input.quant_metadata
        else:
            # 该层输入没有从之前层传播的嵌入量化数据
            # 1. 如果用户在该层进行了显式设定，则使用
            # 2. 如果配置了auto_fallback，则使用
            # Todo: 重点关注这些设定的含义以及优先级
            if idx not in self.inputs_quant_settings_overrides and not self.inputs_quant_auto_fallback:
                raise RuntimeError('<{}> Input {}: Expected tensor with embedded quantization metadata. Either:\n'
                                   '1. Make sure the previous operation is quantized\n'
                                   '2. Provide explicit input quantization settings\n'
                                   '3. Set inputs_quant_auto_fallback'.format(self.distiller_name, idx))
            if self.preset_act_stats:
                qmd = self.inputs_quant_metadata_fallback[idx]
            else:
                # Todo: 读取量化设置，然后计算量化参数，创建量化元数据
                if idx in self.inputs_quant_settings_overrides:
                    q_settings = self.inputs_quant_settings_overrides[idx]
                else:
                    # 运行到这个分支说明inputs_quant_auto_fallback已经设置好了
                    q_settings = deepcopy(self.output_quant_settings)
                    q_settings.clip_half_range = False
                # 从张量中获取参数
                scale, zp = _get_quant_params_from_tensor(input, q_settings.num_bits, q_settings.quant_mode,
                                                          q_settings.clip_mode, q_settings.per_channel,
                                                          q_settings.clip_n_stds, q_settings.clip_half_range,
                                                          self.scale_approx_mult_bits)
                signed = q_settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED
                restrict_qrange = q_settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
                min_q_val, max_q_val = get_quantized_range(q_settings.num_bits, signed, restrict_qrange)
                # 构造量化的元数据
                qmd = TensorQuantMetadata(scale, zp, min_q_val, max_q_val)
        # 确保scale和zp在正确的计算设备上
        qmd = TensorQuantMetadata(qmd.scale.to(input.device), qmd.zero_point.to(input.device),
                                  qmd.min_q_val, qmd.max_q_val)
        return qmd

    def quantized_forward(self, *inputs_q):
        """
        在量化的输入上执行前向传播并且返回量化后的结果
        :param: 量化后的输入: 输入值量化后的张量或张量列表
        :return: 输出值量化后的张量
        """
        raise NotImplementedError

    def get_output_quantization_params(self, accumulator):
        """
        为输出计算量化参数(scale以及zero-point)
        用来:
            1. 计算累加器-->量化输出的重新量化参数
            2. 将输出解量化到FP32

        :param accumulator: 累加器值的张量
        :return: scale和zero-point的元组
        """
        raise NotImplementedError

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        """
        为重新量化计算量化的参数
        重新量化：将中间的整数累加器转换到量化输出范围内。

        :param output_scale: 输出的放缩银子
        :param output_zero_point: 输出的zero-point
        :return: scale和zero-point的元组
        """
        raise NotImplementedError

    # 使用pytorch的量化模型
    def to_pytorch_quant(self, reduce_range):
        assert self.output_quant_settings.num_bits == 8, \
            'Conversion to PyTorch PTQ supported only for 8-bit quantization'
        assert self.preset_act_stats, 'Conversion to PyTorch PTQ supported only for PTQ wrappers with activation stats'
        return self._convert_to_pytorch_quant(reduce_range)

    def _convert_to_pytorch_quant(self, reduce_range):
        raise NotImplementedError

    # 量化参数的表示信息汇总
    def extra_repr(self):
        if self.output_quant_settings.num_bits is None:
            return 'output_quant_settings=Not_Quantized'

        tmpstr = 'output_quant_settings={0}'.format(self.output_quant_settings)
        tmpstr += '\naccum_quant_settings={0}'.format(self.accum_quant_settings)
        overrides = self.inputs_quant_settings_overrides
        tmpstr += '\n  inputs_quant_auto_fallback={}'.format(self.inputs_quant_auto_fallback)
        tmpstr += ', forced_quant_settings_for_inputs={}'.format(
            'None' if not overrides else list(overrides.keys()))
        for idx, qset in overrides.items():
            tmpstr += '\n    input_{}_settings={}'.format(idx, qset)
        tmpstr += '\nscale_approx_mult_bits={}'.format(self.scale_approx_mult_bits)
        tmpstr += '\npreset_activation_stats={0}'.format(self.preset_act_stats)
        if self.preset_act_stats:
            tmpstr += '\n  output_scale={0}, output_zero_point={1}'.format(_quant_param_to_str(
                self.output_scale), _quant_param_to_str(self.output_zero_point))
            for idx in self.inputs_quant_settings_overrides:
                qmd = self.inputs_quant_metadata_fallback[idx]
                tmpstr += '\n  input_#{0}_scale={1}, input_#{0}_zero_point={2}'.format(
                    idx, _quant_param_to_str(qmd.scale), _quant_param_to_str(qmd.zero_point))
        return tmpstr


class RangeLinearQuantParamLayerWrapper(RangeLinearQuantWrapper):
    """
    相关参数的解读：
        wrapped_module:  包装的模型
        num_bits_acts: 用于输入和输出量化的比特数目
        num_bits_params: 用于参数(权重和偏置)量化的比特数目
        num_bits_accum: 分配给中间整型结果累加器的比特数目
        mode: 量化模式
        per_channel_wts: 每个输出通道使用独立的量化参数量化权重
    """

    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32,
                 mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE, per_channel_wts=False, activation_stats=None,
                 clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None, input_overrides=None,
                 inputs_quant_auto_fallback=False, save_fp_weights=False, also_clip_weights=False):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts, num_bits_accum, mode,
                                                                clip_acts, activation_stats, clip_n_stds,
                                                                clip_half_range,
                                                                scale_approx_mult_bits,
                                                                input_overrides=input_overrides,
                                                                inputs_quant_auto_fallback=inputs_quant_auto_fallback)
        if not isinstance(wrapped_module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D, Conv3D and Linear modules')
        # 如果激活单元不需要量化，则执行fake量化，即先量化再解量化
        self.fake_quant_params = self.output_quant_settings.num_bits is None
        clip_wts_mode, clip_wts_n_stds = ClipMode.NONE, None
        if also_clip_weights:
            clip_wts_mode = self.output_quant_settings.clip_mode
            clip_wts_n_stds = self.output_quant_settings.clip_n_stds
        self.wts_quant_settings = QuantSettings(num_bits_params, self.mode.weights, clip_wts_mode, clip_wts_n_stds,
                                                False, per_channel_wts)
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(
            self.wts_quant_settings.num_bits,
            self.wts_quant_settings.quant_mode != LinearQuantMode.ASYMMETRIC_UNSIGNED,
            self.wts_quant_settings.quant_mode == LinearQuantMode.SYMMETRIC_RESTRICTED
        )
        self.save_fp_weights = save_fp_weights
        # 保存浮点权重，以便于重新量化
        if save_fp_weights:
            wrapped_module.register_buffer('float_weight', wrapped_module.weight.clone().detach())
        # 量化权重
        w_scale, w_zero_point = _get_quant_params_from_tensor(wrapped_module.weight,
                                                              self.wts_quant_settings.num_bits,
                                                              self.wts_quant_settings.quant_mode,
                                                              clip=self.wts_quant_settings.clip_mode,
                                                              per_channel=self.wts_quant_settings.per_channel,
                                                              num_stds=self.wts_quant_settings.clip_n_stds)
        w_scale = w_scale if isinstance(w_scale, torch.Tensor) else torch.tensor(w_scale)
        w_zero_point = w_zero_point if isinstance(w_zero_point, torch.Tensor) else torch.tensor(w_zero_point)
        # 注册权重量化缓存
        self.register_buffer('w_scale', w_scale)
        self.register_buffer('w_zero_point', w_zero_point)
        # 对权重进行量化
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.w_zero_point, self.params_min_q_val,
                              self.params_max_q_val, inplace=True)
        # 对偏置进行量化
        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        if self.has_bias and (self.fake_quant_params or not self.preset_act_stats):
            # Todo: 对偏置的量化使用的是累加器的量化配置
            b_scale, b_zero_point = _get_quant_params_from_tensor(wrapped_module.bias,
                                                                  self.accum_quant_settings.num_bits,
                                                                  self.accum_quant_settings.quant_mode)
            self.register_buffer('b_scale', b_scale)
            self.register_buffer('b_zero_point', b_zero_point)
            base_b_q = linear_quantize_clamp(wrapped_module.bias.data, self.b_scale, self.b_zero_point,
                                             self.accum_min_q_val, self.accum_max_q_val)
            if not self.preset_act_stats:
                # 动态范围，存储在的附属的buffer中
                # 每次都基于动态输入放缩因子进行重新量化
                self.register_buffer('base_b_q', base_b_q)
        # 允许重新量化偏置
        if self.has_bias and self.preset_act_stats:
            self.register_buffer('fp_bias', self.wrapped_module.bias.data.clone().detach())
        # 如果不需要量化激活单元，则对参数进行解量化然后返回？
        # Todo: 为什么不直接返回呢？
        if self.fake_quant_params:
            linear_dequantize(wrapped_module.weight.data, self.w_scale, self.w_zero_point, inplace=True)
            if self.has_bias:
                wrapped_module.bias = torch.nn.Parameter(linear_dequantize(base_b_q, self.b_scale, self.b_zero_point))
            return
        # 需要量化激活单元，设置累加器的量化参数
        device = self.w_scale.device
        if self.preset_act_stats:
            t = torch.empty_like(self.w_scale)
            if self.wts_quant_settings.per_channel:
                t = t.squeeze(dim=-1)
            self.register_buffer('accum_scale', t)
        else:
            self.accum_scale = torch.ones(1).to(device)
        self.register_buffer('is_simulated_quant_weight_shifted', torch.tensor(False, device=device))

    def named_linear_quant_params(self, filter=False):
        # 返回权重的一些信息
        if self.save_fp_weights:
            yield 'w_scale', self.w_scale
            # 如果是非对称量化，还需要返回zero_point
            if not filter or is_linear_quant_mode_asymmetric(self.mode.weights):
                yield 'w_zero_point', self.w_zero_point
        yield from super(RangeLinearQuantParamLayerWrapper, self).named_linear_quant_params(filter=filter)

    def set_linear_quant_param(self, name, val):
        if name in ['w_scale', 'w_zero_point']:
            if self.save_fp_weights:
                super().set_linear_quant_param(name, val)
                # Todo: 设置参数的函数体内为什么会出现量化和解量化的操作
                self.wrapped_module.weight.data.copy_(self.wrapped_module.float_weight.data)
                linear_quantize_clamp(self.wrapped_module.weight.data, self.w_scale, self.w_zero_point,
                                      self.params_min_q_val,
                                      self.params_max_q_val, inplace=True)
                # 如果是假量化，则进行解量化
                if self.fake_quant_params:
                    linear_dequantize(self.wrapped_module.weight.data, self.w_scale, self.w_zero_point, inplace=True)
            else:
                raise UnsatisfiedRequirements('Cannot re-quantize the weights. Please specify \'save_fp_weights\' in '
                                              'the %s constructor to enable re-quantizing the weights.' %
                                              self.__class__.__name__)
        else:
            super().set_linear_quant_param(name, val)

    def _check_requirements_weights_clipping(self, setter=False):
        if not self.wts_quant_settings.num_bits:
            raise UnsatisfiedRequirements('Cannot retrieve clipping values because the weights aren\'t quantized.')
        if setter and not self.save_fp_weights:
            warnings.warn('Without saving fp32 version of weights, re-quantization is disabled. To enable, '
                          'please set \'save_fp_weights\' while constructing the wrapper.')

    @property
    def weight_clipping(self):
        self._check_requirements_weights_clipping(setter=False)
        bits, mode = self.wts_quant_settings.num_bits, self.wts_quant_settings.quant_mode
        scale, zp = self.w_scale, self.w_zero_point
        return _get_clipping_value(scale, zp, bits, mode)

    @weight_clipping.setter
    def weight_clipping(self, val):
        self._check_requirements_weights_clipping(setter=True)
        bits = self.wts_quant_settings.num_bits
        val_min, val_max = _check_clipping_val(val, self.wts_quant_settings.quant_mode, False)

        # Todo: __init__函数中不是已经计算好了权重量化的参数了吗？为什么这里还要再计算一次呢？
        if is_linear_quant_mode_symmetric(self.wts_quant_settings.quant_mode):
            scale, zp = symmetric_linear_quantization_params(bits, abs(max(val_max, val_min)))
        else:
            signed = self.wts_quant_settings.quant_mode == LinearQuantMode.ASYMMETRIC_SIGNED
            scale, zp = asymmetric_linear_quantization_params(bits, val_min, val_max, signed=signed)
        # 对权重量化的参数进行设置
        self.set_linear_quant_param('w_scale', scale)
        self.set_linear_quant_param('w_zero_point', zp)

    def named_clipping(self, filter=False):
        try:
            yield from super().named_clipping(filter=filter)
        except UnsatisfiedRequirements as ex:
            warnings.warn(str(ex))
        # 调用计算属性
        val = self.weight_clipping
        if filter and is_linear_quant_mode_symmetric(self.mode.weights):
            val = val[1]
        yield 'weight_clipping', val

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if not self.fake_quant_params and self.is_simulated_quant_weight_shifted:
            # 想要给它们的整数表示返回权重
            self.wrapped_module.weight.data -= self.w_zero_point
            self.is_simulated_quant_weight_shifted.fill_(False)
        return super(RangeLinearQuantParamLayerWrapper, self).state_dict(destination, prefix, keep_vars)

    def quantized_forward(self, input_q):
        def get_accum_scale(input):
            # 累加器的缩放因子=输入的缩放因子乘权重的缩放因子
            accum_scale = input.quant_metadata.scale * self.w_scale
            if self.wts_quant_settings.per_channel:
                accum_scale = accum_scale.squeeze(dim=-1)
            if self.scale_approx_mult_bits:
                accum_scale = approx_scale_as_mult_and_shift(accum_scale, self.scale_approx_mult_bits)
            return accum_scale

        # 有预设的激活单元统计数据
        if self.preset_act_stats:
            if self.num_forwards == 0 or self.force_readjust:
                self.accum_scale.copy_(get_accum_scale(input_q))
                if self.has_bias:
                    # 使用累加器的缩放因子对偏置重新量化
                    self.wrapped_module.bias.data.copy_(
                        linear_quantize_clamp(self.fp_bias, self.accum_scale.squeeze(), 0,
                                              self.accum_min_q_val, self.accum_max_q_val)
                    )
                self.force_readjust.fill_(False)
        else:
            self.accum_scale = get_accum_scale(input_q)
            if self.has_bias:
                # Todo: 相关计算看distiller文档
                # 对偏置进行重新量化，使其匹配x*w的放缩因子：b_q' = (in_scale * w_scale / b_scale) * (b_q + b_zero_point)
                bias_requant_scale = self.accum_scale.squeeze() / self.b_scale
                if self.scale_approx_mult_bits is not None:
                    bias_requant_scale = approx_scale_as_mult_and_shift(bias_requant_scale, self.scale_approx_mult_bits)
                self.wrapped_module.bias.data = linear_quantize_clamp(self.base_b_q + self.b_zero_point,
                                                                      bias_requant_scale, 0,
                                                                      self.accum_min_q_val, self.accum_max_q_val)
        if is_linear_quant_mode_asymmetric(self.wts_quant_settings.quant_mode) and \
                not self.is_simulated_quant_weight_shifted:
            # 将w_zero_point存储进包装模块的权重中来改进推理时的性能
            self.wrapped_module.weight.data += self.w_zero_point
            self.is_simulated_quant_weight_shifted.fill_(True)
        input_q += input_q.quant_metadata.zero_point
        # Todo: 在调用该方法之前，权重已经量化了，主要是对bias进行量化。
        accum = self.wrapped_module(input_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)
        return accum

    def get_output_quantization_params(self, accumulator):
        # 如果有预设的数据，则在__init__中已经计算好output的scale和zp了
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point
        # 获得累加器即y的浮点数据
        y_f = accumulator / self.accum_scale
        q_set = self.output_quant_settings
        # 从输出浮点向量中计算量化参数
        return _get_quant_params_from_tensor(y_f, q_set.num_bits, q_set.quant_mode,
                                             clip=q_set.clip_mode, num_stds=q_set.clip_n_stds,
                                             half_range=q_set.clip_half_range,
                                             scale_approx_mult_bits=self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        requant_scale = output_scale / self.accum_scale
        if self.scale_approx_mult_bits is not None:
            requant_scale = approx_scale_as_mult_and_shift(requant_scale, self.scale_approx_mult_bits)
        return requant_scale, output_zero_point

    # Todo: 转换到pytorch的量化版本，先Pass

    def _convert_to_pytorch_quant(self, reduce_range):
        pass

    #     wrapped = self.wrapped_module
    #     supported = (nn.Conv2d, nn.Linear)
    #     # Tuple of module type and flag for relu fusing
    #     mapping = {
    #         (nn.Linear, False): nnq.Linear,
    #         (nn.Linear, True): nniq.LinearReLU,
    #         (nn.Conv2d, False): nnq.Conv2d,
    #         (nn.Conv2d, True): nniq.ConvReLU2d
    #     }
    #     if nn.Conv3d in torch.quantization.DEFAULT_MODULE_MAPPING:
    #         # Conv3D supported only from PyTorch 1.4
    #         supported += nn.Conv3d,
    #         mapping.update({
    #             (nn.Conv3d, False): nnq.Conv3d,
    #             (nn.Conv3d, True): nniq.ConvReLU3d,
    #         })
    #     assert isinstance(wrapped, supported), \
    #         'Conversion to PyTorch PTQ supported only for {}'.format(','.join(supported))
    #     assert self.wts_quant_settings.num_bits == 8, 'Conversion to PyTorch PTQ supported only for 8-bit quantization'
    #
    #     # Convert weights - required by PyTorch to be signed 8-bit (torch.qint8)
    #     q_weight = pytqc.distiller_quantized_tensor_to_pytorch(wrapped.weight.clone().detach(),
    #                                                            self.w_scale, self.w_zero_point,
    #                                                            self.wts_quant_settings.num_bits,
    #                                                            self.wts_quant_settings.quant_mode, torch.qint8,
    #                                                            self.wts_quant_settings.per_channel, 0)
    #
    #     # PyTorch PTQ modules expect the bias in FP32, we need to dequantize if necessary
    #     # With Distiller PTQ the bias is only quantized on the first forward - we do a crude check if it has
    #     # been quantized or not
    #     fp_bias = wrapped.bias.clone().detach()
    #     if self.has_bias:
    #         bias_quantized = (fp_bias == fp_bias.int()).all()
    #         if bias_quantized:
    #             fp_bias = linear_dequantize(fp_bias, self.accum_scale.squeeze(), 0, True)
    #
    #     pytorch_cls = mapping[(type(wrapped), self.clip_half_range)]
    #     if isinstance(wrapped, nn.Linear):
    #         pytorch_module = pytorch_cls(wrapped.in_features, wrapped.out_features, wrapped.bias is not None)
    #     else:
    #         pytorch_module = pytorch_cls(wrapped.in_channels, wrapped.out_channels, wrapped.kernel_size,
    #                                      wrapped.stride, wrapped.padding, wrapped.dilation, wrapped.groups,
    #                                      wrapped.bias is not None, wrapped.padding_mode)
    #
    #     pytorch_module.set_weight_bias(q_weight, fp_bias)
    #
    #     # Convert activations qparams - required by PyTorch to be unsigned 8-bit (torch.quint8)
    #     out_scale, out_zp = pytqc.distiller_qparams_to_pytorch(self.output_scale, self.output_zero_point,
    #                                                            self.output_quant_settings.num_bits,
    #                                                            self.output_quant_settings.quant_mode, torch.quint8,
    #                                                            reduce_range)
    #     pytorch_module.scale = float(out_scale)
    #     pytorch_module.zero_point = int(out_zp)
    #
    #     return pytorch_module

    def extra_repr(self):
        tmpstr = 'weights_quant_settings={0}\n'.format(self.wts_quant_settings)
        tmpstr += super(RangeLinearQuantParamLayerWrapper, self).extra_repr()
        tmpstr += '\nweights_scale={0}, weights_zero_point={1}'.format(_quant_param_to_str(self.w_scale),
                                                                       _quant_param_to_str(self.w_zero_point))
        if not self.preset_act_stats and self.has_bias:
            tmpstr += '\nbase_bias_scale={0}, base_bias_zero_point={1}'.format(_quant_param_to_str(self.b_scale),
                                                                               _quant_param_to_str(self.b_zero_point))
        return tmpstr


class NoStatsError(Exception):
    pass


class RangeLinearQuantConcatWrapper:
    pass


class RangeLinearQuantEltwiseAddWrapper:
    pass


class RangeLinearQuantEltwiseMultWrapper:
    pass


class RangeLinearQuantMatmulWrapper:
    pass


class FPWrapper:
    def __init__(self, module, fpq_module):
        ...


class RangeLinearFakeQuantWrapper(RangeLinearQuantWrapper):
    def __init__(self, wrapped_module, num_bits_acts, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 activation_stats=None, clip_n_stds=None, clip_half_range=False, scale_approx_mult_bits=None,
                 fpq_module=None, input_overrides=None, inputs_quant_auto_fallback=False, quantize_inputs=False):
        if isinstance(wrapped_module, (nn.ReLU, nn.ReLU6)):
            # In case of ReLU + Gauss/Laplace clipping, need to clip according to stats before ReLU is applied
            clip_half_range = True
            if clip_acts in (ClipMode.GAUSS, ClipMode.LAPLACE):
                activation_stats['output']['mean'] = activation_stats['inputs'][0]['mean']
                activation_stats['output']['std'] = activation_stats['inputs'][0]['std']
                activation_stats['output']['b'] = activation_stats['inputs'][0]['b']
        super(RangeLinearFakeQuantWrapper, self).__init__(wrapped_module, num_bits_acts, mode=mode,
                                                          clip_acts=clip_acts, activation_stats=activation_stats,
                                                          clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                          scale_approx_mult_bits=scale_approx_mult_bits,
                                                          input_overrides=input_overrides,
                                                          inputs_quant_auto_fallback=inputs_quant_auto_fallback)
        self.fpq_module = str(fpq_module) if fpq_module else None
        self.dtype = torch.float
        self.quantize_inputs = quantize_inputs
        if self.fpq_module:
            self.dtype = {'16': torch.half, '32': torch.float, '64': torch.double}[self.fpq_module]
            self.wrapped_module.to(self.dtype)

    def _prepare_input(self, idx, input):
        if not self.quantize_inputs:
            return input

        previously_quantized = hasattr(input, 'quant_metadata')
        input.quant_metadata = self._get_input_quant_metadata(idx, input)
        if previously_quantized:
            return input

        # "Fresh" tensor, so need to quantize and the de-quantize (because this is the fake-quant wrapper)
        input_q = linear_quantize_clamp_with_metadata(input, inplace=False)
        return linear_dequantize_with_metadata(input_q, inplace=True)

    def quantized_forward(self, *inputs_q):
        inputs_q = utils.convert_tensors_recursively_to(inputs_q, dtype=self.dtype)
        outputs = self.wrapped_module(*inputs_q)
        return utils.convert_tensors_recursively_to(outputs, dtype=self.dtype)

    def get_output_quantization_params(self, accumulator):
        if self.preset_act_stats:
            return self.output_scale, self.output_zero_point
        else:
            q_set = self.output_quant_settings
            return _get_quant_params_from_tensor(accumulator, q_set.num_bits, q_set.quant_mode, q_set.clip_mode,
                                                 q_set.per_channel, q_set.clip_n_stds, q_set.clip_half_range,
                                                 self.scale_approx_mult_bits)

    def get_accum_to_output_re_quantization_params(self, output_scale, output_zero_point):
        return output_scale, output_zero_point

    def extra_repr(self):
        tmpstr = super(RangeLinearFakeQuantWrapper, self).extra_repr()
        if self.dtype:
            tmpstr += '\nwrapped_module_float_dtype={}.'.format(self.dtype)
        return tmpstr


# noinspection PyShadowingNames,PyArgumentList
class PostTrainLinearQuantizer(Quantizer):
    """"
    对一个模型应用基于范围的线性量化策略。该类量化器仅可在对预训练的模型评估阶段使用

    下列模型/操作有考虑量化的特定的实现:
      * torch.nn.Conv2d/Conv3d
      * torch.nn.Linear
      * torch.nn.Embedding
      * distiller.modules.Concat
      * distiller.modules.EltwiseAdd
      * distiller.modules.EltwiseMult
      * distiller.modules.Matmul
      * distiller.modules.BatchMatmul
    在创建量化器之前，一个现存的模型需要进行修改以使用distiller.modules.*模块.

    任何不在上述列表中的叶子模型需要被fake-quantized，即执行浮点模块（浮点版本FP64/32/16可以通过fpq_module参数指定），然后量化其输出.

    为了完全禁用对模块的量化，可以使用overrides机制指定模块的bits_activations以及bits_weights为NONE
    # Todo: 对以下参数的再理解
    参数：
        model: 需要量化的模型
        its_activations/parameters/accum: 用来量化每个类型张量的比特数目
        overrides: 重写层的量化设置
        mode: 量化模式
        clip_acts: 激活单元使用的clipping模式
        per_channel_wts: 每个输出通道启动独立的权重量化参数
        model_activation_stats: 激活层的统计数据，用来量化参数的静态计算，如果为None，参数将动态计算
        fp16: 设为True时，将模型转为半精度
        scale_approx_mult_bits: 如果不是None，则scale会使用整型乘法，后接比特派偏移进行估计。即使用整型计算代替了浮点类型的放缩因子
        inputs_quant_auto_fallback: 见<distiller_root>/examples/post_train_quant/resnet18_imagenet_post_train_input_overrides.yaml
        fpq_module: 在浮点模式下使用模块，并且仅量化它们的输出，取(16,32,64)，在FakeQuantWrapper下使用
        save_fp_weights: 是否保存浮点权重的一个副本，允许之后对权重进行重新量化
    注意：
        如果设置了fpq_module,则除了overrides重写的所有层都将设置为浮点版本，而不考虑激活单元/参数/累加器
    """

    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, clip_acts=ClipMode.NONE,
                 per_channel_wts=False, model_activation_stats=None, fp16=False,
                 clip_n_stds=None, clip_half_range=False,
                 scale_approx_mult_bits=None, inputs_quant_auto_fallback=True,
                 fpq_module=None, save_fp_weights=False, also_clip_weights=False):
        overrides_bkp = deepcopy(overrides)
        super(PostTrainLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters, bits_bias=bits_accum,
                                                       overrides=overrides, train_with_fp_copy=False)
        if fp16 and str(fpq_module) not in ('16', 'None'):
            raise ValueError('Conflict - fp16 set to true and fpq_module set to other than 16.')
        mode = verify_quant_mode(mode)
        clip_acts = verify_clip_mode(clip_acts)
        if clip_acts == ClipMode.N_STD and clip_n_stds is None:
            raise ValueError('clip_n_stds must not be None when clip_acts set to N_STD')
        # 对激活单元的统计数据进行加载
        if model_activation_stats is not None:
            # 如果配置的是yaml文件
            if isinstance(model_activation_stats, str):
                if not os.path.isfile(model_activation_stats):
                    raise ValueError("Model activation stats file not found at: " + model_activation_stats)
                quant_logger.info('Loading activation stats from: ' + model_activation_stats)
                with open(model_activation_stats, 'r') as stream:
                    model_activation_stats = utils.yaml_ordered_load(stream)
            elif not isinstance(model_activation_stats, (dict, OrderedDict)):
                raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')
        mode_dict = {'activations': _enum_to_str(mode.activations), 'weights': _enum_to_str(mode.weights)}
        self.model.quantizer_metadata = {
            'type': type(self),
            'params': {'bits_activations': bits_activations,
                       'bits_parameters': bits_parameters,
                       'bits_accum': bits_accum,
                       'mode': mode_dict,
                       'clip_acts': _enum_to_str(clip_acts),
                       'clip_n_stds': clip_n_stds,
                       'clip_half_range': clip_half_range,
                       'per_channel_wts': per_channel_wts,
                       'scale_approx_mult_bits': scale_approx_mult_bits,
                       'inputs_quant_auto_fallback': inputs_quant_auto_fallback,
                       'fpq_module': fpq_module,
                       'model_activation_stats': model_activation_stats,
                       'overrides': overrides_bkp}
        }
        self.clip_acts = clip_acts
        self.clip_n_stds = clip_n_stds
        self.model_activation_stats = model_activation_stats or {}
        self.bits_accum = bits_accum
        self.mode = mode
        self.save_fp_weights = save_fp_weights
        self.also_clip_weights = also_clip_weights

        def _check_fp16_arg(fp16, fpq_module):
            if fp16:
                warnings.warn("Argument 'fp16' is deprecated. Please use 'fpq_module'(=16/32/64) argument.",
                              DeprecationWarning)
                fpq_module = fpq_module or 16
            return fpq_module

        # 替换参数层
        def replace_param_layer(module, name, qbits_map, per_channel_wts=per_channel_wts,
                                mode=mode, fp16=fp16, scale_approx_mult_bits=scale_approx_mult_bits,
                                clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                input_overrides=None, fpq_module=fpq_module, fake=False):
            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if fpq_module and not fake:
                return FPWrapper(module, fpq_module)

            norm_name = utils.normalize_module_name(name)
            activation_stats = self.model_activation_stats.get(norm_name, None)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            qbits = qbits_map[name]

            if qbits.acts is not None and qbits.wts is None:
                # Quantizing only activations equals fake-quantization
                fake = True

            if fake:
                return RangeLinearFakeQuantWrapper(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                                   activation_stats=activation_stats,
                                                   clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                   scale_approx_mult_bits=scale_approx_mult_bits,
                                                   fpq_module=fpq_module, input_overrides=input_overrides,
                                                   inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                   quantize_inputs=False)

            return RangeLinearQuantParamLayerWrapper(module, qbits.acts, qbits.wts,
                                                     num_bits_accum=self.bits_accum, mode=mode, clip_acts=clip_acts,
                                                     per_channel_wts=per_channel_wts,
                                                     activation_stats=activation_stats,
                                                     clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                     scale_approx_mult_bits=scale_approx_mult_bits,
                                                     input_overrides=input_overrides,
                                                     inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                     save_fp_weights=self.save_fp_weights,
                                                     also_clip_weights=self.also_clip_weights)

        # 替换非参数层
        def replace_non_param_layer(wrapper_type, module, name, qbits_map, fp16=fp16,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                    input_overrides=None, inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                    fpq_module=fpq_module, fake=False):
            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if fpq_module and not fake:
                return FPWrapper(module, fpq_module)
            norm_name = utils.normalize_module_name(name)
            activation_stats = self.model_activation_stats.get(norm_name, None)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            qbits = qbits_map[name]

            if fake:
                return RangeLinearFakeQuantWrapper(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                                   activation_stats=activation_stats,
                                                   clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                                   scale_approx_mult_bits=scale_approx_mult_bits,
                                                   fpq_module=fpq_module, input_overrides=input_overrides,
                                                   inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                                   quantize_inputs=False)
            try:
                return wrapper_type(module, qbits.acts, mode=mode, clip_acts=clip_acts,
                                    activation_stats=activation_stats,
                                    clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                    scale_approx_mult_bits=scale_approx_mult_bits,
                                    input_overrides=input_overrides,
                                    inputs_quant_auto_fallback=inputs_quant_auto_fallback)
            except NoStatsError:
                warnings.warn('WARNING: {0} - quantization of {1} without stats not supported. '
                              'Keeping the original FP32 module'.format(name, module.__class__.__name__), UserWarning)
                return module

        def replace_fake_quant(module, name, qbits_map, fp16=fp16,
                               clip_acts=None, clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                               scale_approx_mult_bits=scale_approx_mult_bits, input_overrides=None,
                               inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                               fpq_module=fpq_module, fake=True, make_identity=False, quantize_inputs=True):
            if isinstance(module, (nn.ReLU, nn.ReLU6)) and make_identity:
                named_modules = OrderedDict(self.model.named_modules())
                pred = self.adjacency_map[name].predecessors[0].name
                if isinstance(named_modules[pred], RangeLinearQuantWrapper):
                    return nn.Identity()

            if utils.has_children(module):
                return module

            fpq_module = _check_fp16_arg(fp16, fpq_module)
            if not fake:
                return FPWrapper(module, fpq_module)

            norm_name = utils.normalize_module_name(name)
            clip_acts = verify_clip_mode(clip_acts or self.clip_acts)
            return RangeLinearFakeQuantWrapper(module, qbits_map[name].acts, mode=mode, clip_acts=clip_acts,
                                               activation_stats=self.model_activation_stats.get(norm_name, None),
                                               clip_n_stds=clip_n_stds, clip_half_range=clip_half_range,
                                               scale_approx_mult_bits=scale_approx_mult_bits,
                                               fpq_module=fpq_module, input_overrides=input_overrides,
                                               inputs_quant_auto_fallback=inputs_quant_auto_fallback,
                                               quantize_inputs=quantize_inputs)

        # 对替换工厂进行赋值
        self.replacement_factory[nn.Conv2d] = replace_param_layer
        self.replacement_factory[nn.Conv3d] = replace_param_layer
        self.replacement_factory[nn.Linear] = replace_param_layer

        factory_concat = partial(
            replace_non_param_layer, RangeLinearQuantConcatWrapper)
        factory_eltwiseadd = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseAddWrapper)
        factory_eltwisemult = partial(
            replace_non_param_layer, RangeLinearQuantEltwiseMultWrapper)
        factory_matmul = partial(
            replace_non_param_layer, RangeLinearQuantMatmulWrapper)

        update_wrapper(factory_concat, replace_non_param_layer)
        update_wrapper(factory_eltwiseadd, replace_non_param_layer)
        update_wrapper(factory_eltwisemult, replace_non_param_layer)
        update_wrapper(factory_matmul, replace_non_param_layer)

        self.replacement_factory[modules.Concat] = factory_concat
        self.replacement_factory[modules.EltwiseAdd] = factory_eltwiseadd
        self.replacement_factory[modules.EltwiseMult] = factory_eltwisemult
        self.replacement_factory[modules.Matmul] = factory_matmul
        self.replacement_factory[modules.BatchMatmul] = factory_matmul

        # 设定默认的替换函数
        self.default_replacement_fn = replace_fake_quant
        self.replacement_blacklist.append(nn.Dropout)

        # 在.prepare_model()中设置该属性
        self.linear_quant_params = None

    def named_linear_quant_params(self, yield_clipping_params=False, filter=False):
        """带名字返回线性量化的参数，包括输出的scale、zp以及weight的scale和zp

        :param yield_clipping_params:
        :param filter:
        :return:
        """
        # 如果要求返回clipping_params
        if yield_clipping_params:
            yield from self.named_clipping(filter=filter)
            return
        for module_name, module in self.model.named_modules():
            if is_post_train_quant_wrapper(module, include_fpwrapper=False):
                for buff_name, buff in module.named_linear_quant_params(filter=filter):
                    full_buff_name = "%s.%s" % (module_name, buff_name)
                    yield full_buff_name, buff

    def named_clipping(self, filter=False):
        """
        获取模型的所有clipping参数
        :param filter: Todo: 这是什么标识呢？
        :return: tuple[str,tuple[torch.Tensor,torch.Tensor]]
        """
        for module_name, module in self.model.named_modules():
            if not is_post_train_quant_wrapper(module, include_fpwrapper=False):
                continue
            # 调用重写模块的named_clipping方法
            for clip_name, clip_val in module.named_clipping(filter=filter):
                yield '%s.%s' % (module_name, clip_name), clip_val

    def set_clipping(self, name, val):
        """通过名字设定一个clipping参数

        :param name: clipping参数的名称
        :param val: clipping参数的值
        :return:
        """
        module_name = utils.param_name_2_module_name(name)
        clip_name = name.split('.')[-1]
        module = dict(self.model.named_modules())[module_name]
        # 对module以及clip_name的有效性进行验证
        if not is_post_train_quant_wrapper(module, include_fpwrapper=False):
            raise ValueError('\'%s\' isn\'t a wrapper and has no clipping parameters.' % module_name)
        if clip_name not in dict(module.named_clipping()):
            raise ValueError('\'%s\' is not a clipping parameter.' % clip_name)
        setattr(module, clip_name, val)

    def update_clipping_parameters(self, clipping_config):
        for name, val in clipping_config.items():
            self.set_clipping(name, val)

    def _is_clipping_parameter(self, name):
        module_name = utils.param_name_2_module_name(name)
        clip_name = name.split('.')[-1]
        module = dict(self.model.named_modules())[module_name]
        # Todo: 与set_clipping()方法的有效性验证方法类似
        return is_post_train_quant_wrapper(module, False) and clip_name in dict(module.named_clipping())

    # 强制重新调整wrappers
    # Todo: 为什么重新调整？重新调整的应用场景是什么？
    def force_readjust_wrappers(self):
        def _force_readjust(module):
            if isinstance(module, RangeLinearQuantWrapper):
                module.force_readjust.fill_(True)

        self.model.apply(_force_readjust)

    # 设置线性量化参数
    def set_linear_quant_param(self, name, val):
        """
        通过module_name.quant_param_name的形式设置量化参数，也可以设置clipping值
        :param name: 量化参数的名称[module_name].[quant_param_name]
        :param val: 新的值
        :return:
        """
        if self._is_clipping_parameter(name):
            self.set_clipping(name, val)
        else:
            self.linear_quant_params[name].data.fill_(val)
        # 强制对wrappers进行重新调整
        self.force_readjust_wrappers()

    def update_linear_quant_params(self, new_config):
        for k, v in new_config.items():
            self.set_linear_quant_param(k, v)

    def save_per_layer_parameters(self, save_dir):
        pass

    def prepare_model(self, dummy_input=None):
        if not self.model_activation_stats:
            quant_logger.warning("无统计数据文件，将使用动态量化")
        if dummy_input is None:
            raise UnsatisfiedRequirements('PostTrainLinearQuantizer需要dummy_input来执行特定的优化')

        super(PostTrainLinearQuantizer, self).prepare_model(dummy_input)
        save_dir = quant_logger.logdir if hasattr(quant_logger, 'logdir') else '.'
        self.save_per_layer_parameters(save_dir)

    def _pre_prepare_model(self, dummy_input):
        # 执行一些必要的操作
        self._apply_activation_stats_fusions()
        self._apply_fuse_relu()
        save_dir = quant_logger.logdir if hasattr(quant_logger, 'logdir') else '.'
        save_path = os.path.join(save_dir, 'quant_stats_after_prepare_model.yaml')
        utils.yaml_ordered_save(save_path, self.model_activation_stats)
        quant_logger.info('Updated stats saved to ' + save_path)

    def _apply_activation_stats_fusions(self):
        """
        现在寻找层和激活函数的特定的混合fusion
            对统计数据进行修改确保只量化和激活函数相关的范围
            这样做可以减小量化的误差
        :return:
        """
        if not self.model_activation_stats:
            quant_logger.info("No activation stats - skipping optimizations for modules followed by Relu/Tanh/Sigmoid")
            return

        quant_logger.info('Optimizing output statistics for modules followed by ReLU/Tanh/Sigmoid')

        named_modules = OrderedDict(self.model.named_modules())
        model_stats = self.model_activation_stats
        for name, module in named_modules.items():
            qbits = self.module_qbits_map.get(name, QBits(None, None, None))
            if qbits.acts is None:
                continue
            if utils.has_children(module) or name not in self.adjacency_map \
                    or len(self.adjacency_map[name].successors) != 1:
                continue
            successor = self.adjacency_map[name].successors[0]
            name = utils.normalize_module_name(name)
            m_stats = model_stats[name]
            succ_type = successor.type
            succ_stats = model_stats.get(utils.normalize_module_name(successor.name), None)
            if succ_type == 'Relu':
                # ReLU将所有负值都置为0，因此，没有需要去量化它们
                min_val = 0.
                max_val = m_stats['output']['max']
            elif succ_type == 'Sigmoid' or succ_type == 'Tanh':
                max_val = 4. if succ_type == 'Tanh' else 6.
                min_val = -max_val
            # 如果是nn.ReLU6的实例
            elif isinstance(named_modules.get(successor.name, None), nn.ReLU6):
                succ_type = 'ReLU6'
                min_val = 0.
                max_val = min(m_stats['output']['max'], 6)
            else:
                # 如果非特殊类型，则不需要进行特殊的处理
                continue
            quant_logger.debug('  Module {} followed by {}, updating stats'.format(name, succ_type))
            # 对统计数据进行处理，比如clip
            self._clip_stats(m_stats['output'], min_val, max_val)
            # Todo: succ_stats用来做什么？
            if succ_stats is not None:
                succ_stats['inputs'][0] = deepcopy(m_stats['output'])

    def _apply_fuse_relu(self):
        """融合ReLU层到线性层

        :return:
        """
        model_overrides = self.module_overrides_map
        named_modules = dict(self.model.named_modules())
        for name, module in named_modules.items():
            qbits = self.module_qbits_map.get(name, QBits(None, None, None))
            if qbits.acts is None:
                continue
            if utils.has_children(module) or name not in self.adjacency_map \
                    or len(self.adjacency_map[name].successors) != 1:
                continue
            successor = self.adjacency_map[name].successors[0]
            successor_module = named_modules.get(successor.name, None)
            m_override = model_overrides.get(name, OrderedDict())
            model_overrides[name] = m_override
            # 向module overrides中添加half range clipping
            if successor.name in named_modules and isinstance(successor_module, (nn.ReLU, nn.ReLU6)):
                m_override['clip_half_range'] = True
                m_override = model_overrides.get(successor.name, OrderedDict())
                m_override['make_identity'] = True
                model_overrides[successor.name] = m_override

    # noinspection PyMethodMayBeStatic
    def _clip_stats(self, entry, min_val, max_val):
        if entry['max'] < min_val:
            entry['min'] = entry['avg_min'] = entry['max'] = entry['avg_max'] = min_val
        elif entry['min'] > max_val:
            entry['min'] = entry['avg_min'] = entry['max'] = entry['avg_max'] = max_val
        else:
            entry['min'] = max(min_val, entry['min'])
            entry['avg_min'] = max(min_val, entry['avg_min'])
            entry['max'] = min(max_val, entry['max'])
            entry['avg_max'] = min(max_val, entry['avg_max'])

    def _post_prepare_model(self):
        model = self.model
        device = utils.model_device(model)
        for param in model.parameters():
            param.data = param.data.to(device)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)
        self.linear_quant_params = OrderedDict(self.named_linear_quant_params())

    # noinspection PyTypeChecker
    @classmethod
    def from_args(cls, model, args):
        """
        基于设置的命令行参数返回一个PostTrainLinearQuantizer实例
        """
        # mock一个args的实例

        if args.qe_bits_acts == 0:
            args.qe_bits_acts = None
        if args.qe_bits_wts == 0:
            args.qe_bits_wts = None
        overrides = OrderedDict(  # 对于不量化的层，将其在overrides中的配置项置为空
            [
                (layer, OrderedDict([('bits_activations', None), ('bits_weights', None)]))
                for layer in args.qe_no_quant_layers
            ]
        )
        overrides.update(OrderedDict(  # 更新clip的配置信息
            [(layer, OrderedDict([('clip_acts', 'NONE')]))
             for layer in args.qe_no_clip_layers if layer not in args.qe_no_quant_layers]
        ))
        # 直接使用统一的配置
        mode_acts = args.qe_mode
        mode_wts = args.qe_mode
        mode = ModuleQuantMode(mode_acts, mode_wts)
        return cls(model,
                   bits_activations=args.qe_bits_acts,
                   bits_parameters=args.qe_bits_wts,
                   bits_accum=args.qe_bits_accum,
                   mode=mode,
                   clip_acts=args.qe_clip_acts,
                   model_activation_stats=args.qe_stats_file,
                   overrides=overrides,
                   inputs_quant_auto_fallback=True)


_ptq_wrappers_int_only = (RangeLinearQuantWrapper,)
_ptq_wrappers_all = _ptq_wrappers_int_only + (FPWrapper,)


def is_post_train_quant_wrapper(module, include_fpwrapper=True):
    types = _ptq_wrappers_all if include_fpwrapper else _ptq_wrappers_int_only
    return isinstance(module, types)


class FakeLinearQuantization(nn.Module):
    def __init__(self, num_bits=8, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True, inplace=False):
        super(FakeLinearQuantization, self).__init__()

        self.num_bits = num_bits
        self.mode = mode
        self.dequantize = dequantize
        self.inplace = inplace

        # Todo: 在指数滑动平均(EMA)上执行偏差纠正，因此，同时保存有偏值和无偏值
        self.register_buffer('ema_decay', torch.tensor(ema_decay))  # 指数滑动平均的衰减因子
        self.register_buffer('tracked_min_biased', torch.zeros(1))  # 有偏最小值
        self.register_buffer('tracked_min', torch.zeros(1))  # 最小值
        self.register_buffer('tracked_max_biased', torch.zeros(1))  # 有偏最大值
        self.register_buffer('tracked_max', torch.zeros(1))  # 最大值
        self.register_buffer('iter_count', torch.zeros(1))  # 迭代次数
        self.register_buffer('scale', torch.ones(1))  # 放缩因子scale
        self.register_buffer('zero_point', torch.zeros(1))  # 零点

    def forward(self, input):
        # 仅在训练阶段更新观察到的统计数据
        if self.training:
            with torch.no_grad():
                # 获取输入的最小值和最大值
                current_min, current_max = get_tensor_min_max(input)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_decay,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_decay,
                                                                             self.iter_count)
        # 使用无偏的激活单元数据来计算量化参数
        if is_linear_quant_mode_symmetric(self.mode):
            max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
            actual_min, actual_max = -max_abs, max_abs
            if self.training:
                self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(
                    self.num_bits, max_abs, restrict_qrange=self.mode == LinearQuantMode.SYMMETRIC_RESTRICTED)
        else:
            actual_min, actual_max = self.tracked_min, self.tracked_max
            signed = self.mode == LinearQuantMode.ASYMMETRIC_SIGNED
            if self.training:
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min,
                                                                                              self.tracked_max,
                                                                                              signed=signed)
        # 使用有偏数据对输入进行钳制
        input = clamp(input, actual_min.item(), actual_max.item(), False)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, False)

        return input

    def extra_repr(self):
        mode_str = str(self.mode).split('.')[1]
        return 'mode={0}, num_bits={1}, ema_decay={2:.4f})'.format(mode_str, self.num_bits, self.ema_decay)


class FakeQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, quant_mode, ema_decay):
        super(FakeQuantizationWrapper, self).__init__()
        self.wrapped_module = wrapped_module
        # 配置滑动平均策略进行fake量化+解量化
        self.fake_q = FakeLinearQuantization(num_bits, quant_mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))

    def forward(self, *input):
        res = self.wrapped_module(*input)
        res = self.fake_q(res)
        return res


# Todo: 以下部分是Quantization-aware training
class QuantAwareTrainRangeLinearQuantizer(Quantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False,
                 quantize_inputs=True, num_bits_inputs=None):
        super(QuantAwareTrainRangeLinearQuantizer, self).__init__(model, optimizer=optimizer,
                                                                  bits_activations=bits_activations,
                                                                  bits_weights=bits_weights,
                                                                  bits_bias=bits_bias,
                                                                  overrides=overrides,
                                                                  train_with_fp_copy=True)
        mode = verify_quant_mode(mode)

        # 记录一些量化的元数据
        mode_dict = {'activations': _enum_to_str(mode.activations), 'weights': _enum_to_str(mode.weights)}
        self.model.quantizer_metadata['params']['mode'] = mode_dict
        self.model.quantizer_metadata['params']['ema_decay'] = ema_decay
        self.model.quantizer_metadata['params']['per_channel_wts'] = per_channel_wts
        self.model.quantizer_metadata['params']['quantize_inputs'] = quantize_inputs

        # 为输入量化保留一些参数
        self.quantize_inputs = quantize_inputs  # 对输入进行量化的布尔变量
        if num_bits_inputs is not None:
            self.num_bits_inputs = num_bits_inputs
        else:
            self.num_bits_inputs = bits_activations
        self.mode = mode
        self.decay = ema_decay
        self.per_channel_wts = per_channel_wts

        def linear_quantize_param(param_fp, param_meta):
            """对输入的浮点参数根据量化元数据进行线性量化

            :param param_fp: 参数的浮点版本
            :param param_meta: 量化参数的元数据
            :return:
            """
            m = param_meta.module
            # 不会量化embedding层每个通道学到的权重，因为他们被使用作为后续层的输入
            # Todo: 我们还不支持对激活函数的逐通道量化
            perch = not isinstance(m, nn.Embedding) and per_channel_wts and param_fp.dim() in [2, 4]

            with torch.no_grad():
                # 根据输入的浮点张量计算量化参数
                scale, zero_point = _get_quant_params_from_tensor(param_fp, param_meta.num_bits, mode.weights,
                                                                  per_channel=perch)
            # 为模型设置量化参数
            setattr(m, param_meta.q_attr_name + '_scale', scale)
            setattr(m, param_meta.q_attr_name + '_zero_point', zero_point)
            # 解量化=True;inplace=False
            # Todo: 量化后再反量化的目的是什么？
            out = LinearQuantizeSTE.apply(param_fp, scale, zero_point, True, False)
            # 返回量化后的输出
            return out

        def activation_replace_fn(module, name, qbits_map):
            bits_acts = qbits_map[name].acts
            if bits_acts is None:
                # 说明不需要对activation进行量化，返回原module即可
                return module
            # Todo: 对激活单元使用FakeQuantization进行包装
            # 这里返回的是假的量化？
            # Todo: 使用EMA对激活单元进行修正
            return FakeQuantizationWrapper(module, bits_acts, mode.activations, ema_decay)

        self.param_quantization_fn = linear_quantize_param
        self.activation_replace_fn = activation_replace_fn
        # Todo: 对ReLU层应用该量化；对参数层进行参数的量化
        self.replacement_factory[nn.ReLU] = self.activation_replace_fn    # Todo: 对ReLU层也进行量化

    # Todo: 结合具体实例了解该函数的作用，并进行总结
    def _post_prepare_model(self):
        # 如果选择对输入进行量化
        if self.quantize_inputs:
            if isinstance(self.model, nn.DataParallel):
                m = self.model.module
            else:
                m = self.model
            m.inputs_quant = FakeLinearQuantization(self.num_bits_inputs, self.mode.activations, self.decay,
                                                    dequantize=True, inplace=False)
            # 对原来的forward函数进行备份，应用量化后的回调函数
            m.__class__.original_forward = m.__class__.forward
            # 替换前向传播函数：对输入进行量化后，再origin_forward
            m.__class__.forward = inputs_quantize_wrapped_forward

        # # 在量化参数的模块中准备scale和zero_point缓存
        # # 计算dummy的scale和zero_point，以得到它们的维度
        # for ptq in self.params_to_quantize:
        #     m = ptq.module
        #     param_fp = getattr(m, ptq.fp_attr_name)
        #     perch = not isinstance(m, nn.Embedding) and self.per_channel_wts and param_fp.dim() in [2, 4]
        #     with torch.no_grad():
        #         scale, zero_point = _get_quant_params_from_tensor(param_fp, ptq.num_bits, self.mode.weights,
        #                                                           per_channel=perch)
        #     # 注册dummy scale以及zero_point
        #     m.register_buffer(ptq.q_attr_name + '_scale', torch.ones_like(scale))
        #     m.register_buffer(ptq.q_attr_name + '_zero_point', torch.zeros_like(zero_point))


class NCFQuantAwareTrainQuantizer(QuantAwareTrainRangeLinearQuantizer):
    def __init__(self, model, optimizer=None, bits_activations=32, bits_weights=32, bits_bias=32,
                 overrides=None, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, per_channel_wts=False):
        super(NCFQuantAwareTrainQuantizer, self).__init__(model, optimizer=optimizer,
                                                          bits_activations=bits_activations,
                                                          bits_weights=bits_weights,
                                                          bits_bias=bits_bias,
                                                          overrides=overrides,
                                                          mode=mode, ema_decay=ema_decay,
                                                          per_channel_wts=per_channel_wts,
                                                          quantize_inputs=False)

        # Remove 'quantize_inputs' from the metadata dict since this quantizer hard-codes it and doesn't
        # actually take it as an argument
        # 从元数据字典中移除量化输入，因为量化器对其进行硬编码
        self.model.quantizer_metadata['params'].pop('quantize_inputs')

        self.replacement_factory[modules.EltwiseMult] = self.activation_replace_fn
        self.replacement_factory[modules.Concat] = self.activation_replace_fn
        self.replacement_factory[nn.Linear] = self.activation_replace_fn


def update_ema(biased_ema, value, decay, step):
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # 偏差纠正
    return biased_ema, unbiased_ema


def inputs_quantize_wrapped_forward(self, input):
    input = self.inputs_quant(input)
    return self.original_forward(input)
