from enum import Enum
import torch


class LinearQuantMode(Enum):
    """
    定义线性量化的模式：对称、对称+受限、非对称+无符号、非对称+有符号
    """
    SYMMETRIC = 1
    SYMMETRIC_RESTRICTED = 2
    ASYMMETRIC_UNSIGNED = 3
    ASYMMETRIC_SIGNED = 4


def is_linear_quant_mode_symmetric(quant_mode):
    return quant_mode in (LinearQuantMode.SYMMETRIC, LinearQuantMode.SYMMETRIC_RESTRICTED)


def is_linear_quant_mode_asymmetric(quant_mode):
    return not is_linear_quant_mode_symmetric(quant_mode)


# Todo: 在什么情况下使用？为什么要转为浮点数以及扩充维度
def _prep_saturation_val_tensor(sat_val):
    """给定饱和值，对其进行float转换以及维度扩充

    :param sat_val:
    :return:
    """
    # 判断是否是标量
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    # Todo: 关注为什么要进行维度的扩充
    # Answer: 方便后续的tensor处理
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def symmetric_linear_quantization_params(num_bits, saturation_val, restrict_qrange=False):
    """给定浮点范围[-saturation_val,saturation_val]，计算量化参数；由于是对称情况，因此，返回的zero-point总是0
    如果限定范围参数置为True，则量化范围限制在N-1个箱，其中N = 2 ** (num_bits - 1)
    :param num_bits:
    :param saturation_val: 传入饱和值
    :param restrict_qrange:
    :return:
    """
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)
    # 计算量化类型下的饱和值
    if restrict_qrange:
        n = 2 ** (num_bits - 1) - 1
    else:
        n = (2 ** num_bits - 1) / 2

    # Todo: 特殊处理，需要和量化理论结合搞懂
    # 如果浮点值总为0，需要将量化的值也设置为0，因此，重写饱和值为n，使得scale变为1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    """

    :param num_bits: 量化比特数
    :param saturation_min: 最小饱和值，因为是非对称，因此需要传入最小和最大饱和值
    :param saturation_max: 最大饱和值
    :param integral_zero_point: 是否对zero_point进行四舍五入
    :param signed: 有无符号量化
    :return: scale以及zero_point
    """
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)

    is_scalar = scalar_min and scalar_max

    # Todo: 为什么要转移到device？不应该转到tensor吗？
    # 注意，在调用完_prep方法后已经转成了tensor，现在应该转移到相同的设备上
    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    n = 2 ** num_bits - 1

    # 量化的理论需要：确保0在饱和范围内
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # 这里是重载的运算符
    # noinspection PyTypeChecker
    scale = n / diff
    zero_point = scale * sat_min

    # 四舍五入
    if integral_zero_point:
        zero_point = zero_point.round()
    # 有符号情况下，对zero_point进行修正
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    # 原地修改
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """在给定scale和zero_point的情况下，对input进行线性量化

    :param input: 输入的张量
    :param scale: 放缩因子
    :param zero_point: 零点
    :param inplace: 是否进行原地操作
    :return:
    """
    if inplace:
        # 量化过程：乘以缩放因子再减去零点，最后进行四舍五入
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


# 带clamp操作的线性量化实现
def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim >= t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


# Todo: 这里的计算方法结合实际的高维数据去看
def get_tensor_avg_min_max(t, across_dim=None):
    min_per_dim, max_per_dim = get_tensor_min_max(t, per_dim=across_dim)
    # min_per_dim中包含每个维度的最小值和每个维度的最大值；max_per_dim同理
    return min_per_dim.mean(), max_per_dim.mean()


def get_tensor_max_abs(t, per_dim=None):
    min_val, max_val = get_tensor_min_max(t, per_dim=per_dim)
    return torch.max(min_val.abs_(), max_val.abs_())


def get_tensor_avg_max_abs(t, across_dim=None):
    avg_min, avg_max = get_tensor_avg_min_max(t, across_dim=across_dim)
    return torch.max(avg_min.abs_(), avg_max.abs_())


def get_tensor_mean_n_stds_min_max(t, dim=None, n_stds=1):
    if dim is not None:
        raise NotImplementedError('Setting dim != None not supported yet')
    if n_stds <= 0:
        raise ValueError('n_stds must be > 0, got {}'.format(n_stds))
    mean = t.mean()
    std = t.std()
    min_val, max_val = get_tensor_min_max(t)
    min_val = torch.max(min_val, mean - n_stds * std)
    max_val = torch.min(max_val, mean + n_stds * std)
    return min_val, max_val


def get_tensor_mean_n_stds_max_abs(t, dim=None, n_stds=1):
    min_val, max_val = get_tensor_mean_n_stds_min_max(t, dim, n_stds)
    return torch.max(min_val.abs_(), max_val.abs_())


# Todo: 以下几个指标的理论意义是什么？

def get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=False):
    # Todo: shift_bits的含义是什么？
    shift_bits = torch.log2((2 ** mult_bits - 1) / fp32_scale).floor()
    if limit:
        # min函数中对torch类型的重载
        # noinspection PyTypeChecker
        shift_bits = min(mult_bits, shift_bits)
    return shift_bits


def get_scale_approximation_mult(fp32_scale, shift_bits):
    return (fp32_scale * (2 ** shift_bits)).floor()


def get_scale_approximation_params(fp32_scale, mult_bits, limit=False):
    shift_bits = get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=limit)
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    return multiplier, shift_bits


def approx_scale_as_mult_and_shift(fp32_scale, mult_bits, limit=False):
    multiplier, shift_bits = get_scale_approximation_params(fp32_scale, mult_bits, limit=limit)
    return multiplier / (2 ** shift_bits)


def get_quantized_range(num_bits, signed=True, signed_restrict_qrange=True):
    """
    根据给定的num_bits、符号模式以及受限表示返回最小和最大的量化值
    例如num_bits = 4的情况下：
    * signed == False:
        q_min = 0, q_max = 15
    * signed == True;signed_restrict_qrange == False:
        q_min = -8, q_max = 7
    * signed == True;signed_restrict_qrange == True:
        q_min = -7, q_max = 7
    :param num_bits:
    :param signed:
    :param signed_restrict_qrange:
    :return:
    """
    if signed:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax if signed_restrict_qrange else -qmax - 1
        return qmin, qmax
    return 0, 2 ** num_bits - 1


# noinspection PyMethodOverriding
# Todo: 定义线性量化的自动求导函数
class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        # 对输入进行量化得到输出
        output = linear_quantize(input, scale, zero_point, inplace)
        # 需要反量化，对量化的输入执行反量化
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
