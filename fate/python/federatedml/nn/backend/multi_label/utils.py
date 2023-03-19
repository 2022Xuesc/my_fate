from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch import nn

import federatedml.nn.backend.distiller as distiller
import federatedml.nn.backend.distiller.apputils
import inspect


def set_model_input_shape_attr(model, dataset=None, input_shape=None):
    """Sets an attribute named 'input_shape' within the model instance, specifying the expected input shape

    Args:
          model (nn.Module): Model instance
          dataset (str): Name of dataset from which to infer input shape
          input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing
            arbitrarily complex collections of tensors. Used only if 'dataset' is None
    """
    if not hasattr(model, 'input_shape'):
        model.input_shape = _validate_input_shape(dataset, input_shape)


def _validate_input_shape(dataset, input_shape):
    if dataset:
        return tuple(distiller.apputils.classification_get_input_shape(dataset))


def filter_kwargs(dict_to_filter, function_to_call):
    """Utility to check which arguments in the passed dictionary exist in a function's signature

    The function returns two dicts, one with just the valid args from the input and one with the invalid args.
    The caller can then decide to ignore the existence of invalid args, depending on context.
    """

    sig = inspect.signature(function_to_call)
    filter_keys = [param.name for param in sig.parameters.values() if (param.kind == param.POSITIONAL_OR_KEYWORD)]
    valid_args = {}
    invalid_args = {}

    for key in dict_to_filter:
        if key in filter_keys:
            valid_args[key] = dict_to_filter[key]
        else:
            invalid_args[key] = dict_to_filter[key]
    return valid_args, invalid_args


def to_np(var):
    return var.data.cpu().numpy()


class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        super().__init__()
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def density(tensor):
    """计算一个向量的密度

    密度是张量中非零元素的占比，和稀疏水平相对应
    """
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def nnz(tensor):
    return tensor.abs().gt(0).sum()


def model_params_stats(model, param_dims=None, param_types=None):
    """Returns the model , weights count, and the count of weights in the sparse model.

    Returns:
        model_sparsity - 以百分比形式返回模型权重的稀疏水平
        params_cnt - 返回整个模型中权重的数量，包括0
        params_nnz_cnt - 整个模型非零权重的数量
    """
    if param_types is None:
        param_types = ['weight', 'bias']
    if param_dims is None:
        param_dims = [2, 4]
    params_cnt = 0
    params_nnz_cnt = 0
    for name, param in model.state_dict().items():
        if param.dim() in param_dims and any(type in name for type in param_types):
            _density = density(param)
            params_cnt += torch.numel(param)
            params_nnz_cnt += param.numel() * _density
    model_sparsity = (1 - params_nnz_cnt / params_cnt) * 100
    return model_sparsity, params_cnt, params_nnz_cnt


def assign_layer_fq_names(container, name=None):
    """Assign human-readable names to the modules (layers).

    Sometimes we need to access modules by their names, and we'd like to use
    fully-qualified names for convenience.
    """
    for name, module in container.named_modules():
        module.distiller_name = name


def find_module_by_fq_name(model, fq_model_name):
    for module in model.modules():
        if hasattr(module, 'distiller_name') and fq_model_name == module.distiller_name:
            return module
    return None


def size_to_str(torch_size):
    """将pytorch size对象转换成字符串的形式"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '(' + ', '.join(['%d' % v for v in torch_size]) + ')'


def size2str(torch_size):
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, (torch.FloatTensor, torch.cuda.FloatTensor)):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    if isinstance(torch_size, tuple) or isinstance(torch_size, list):
        return size_to_str(torch_size)
    raise TypeError


def yaml_ordered_save(fname, ordered_dict):
    def ordered_dict_representer(self, value):
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())

    yaml.add_representer(OrderedDict, ordered_dict_representer)

    with open(fname, 'w') as f:
        yaml.dump(ordered_dict, f, default_flow_style=False)


def yaml_ordered_load(stream, yaml_loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Function to load YAML file using an OrderedDict

    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """

    class OrderedLoader(yaml_loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def zero_total_num(tensor):
    total = tensor.numel()
    # 在CPU上执行统计
    nonzero = tensor.to('cpu').abs().gt(0).sum()
    zero = total - nonzero
    return zero.item(), total


def sparsity(tensor):
    return 1.0 - density(tensor)


def activation_channels_l1(activation):
    """Calculate the L1-norms of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    When the activations are computed on a distributed GPU system, different parts of the
    activation tensor might be computed by a differnt GPU. If this function is called from
    the forward-callback of some activation module in the graph, we will only witness part
    of the batch.  For example, if the batch_size is 256, and we are using 4 GPUS, instead
    of seeing activations with shape = (256, num_channels, h, w), we may see 4 calls with
    shape = (64, num_channels, h, w).

    Since we want to calculate the average of the L1-norm of each of the channels of the
    activation, we need to move the partial sums results to the CPU, where they will be
    added together.

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L! magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_norms = view_2d.norm(p=1, dim=1)  # (batch*channels) x 1
        featuremap_norms_mat = featuremap_norms.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_norms_mat = activation.norm(p=1, dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_l1: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_norms_mat.mean(dim=0).cpu()


def activation_channels_means(activation):
    """Calculate the mean of each of an activation's channels.

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its L1 magnitudes (i.e. over all of the
    activations in the mini-batch, compute the mean of the L1 magnitude of each channel).
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_means = view_2d.mean(dim=1)  # (batch*channels) x 1
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_means_mat.mean(dim=0).cpu()


def activation_channels_apoz(activation):
    """Calculate the APoZ of each of an activation's channels.

    APoZ is the Average Percentage of Zeros (or simply: average sparsity) and is defined in:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures".

    The activation usually has the shape: (batch_size, num_channels, h, w).

    "We first use global average pooling to convert the output of layer i, which is a
    c x h x w tensor, into a 1 x c vector."

    Returns - for each channel: the batch-mean of its sparsity.
    """
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (
                activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()


def sparsity_2D(tensor):
    """Create a list of sparsity levels for each channel in the tensor 't'

    For 4D weight tensors (convolution weights), we flatten each kernel (channel)
    so it becomes a row in a 3D tensor in which each channel is a filter.
    So if the original 4D weights tensor is:
        #OFMs x #IFMs x K x K
    The flattened tensor is:
        #OFMS x #IFMs x K^2

    For 2D weight tensors (fully-connected weights), the tensors is shaped as
        #IFMs x #OFMs
    so we don't need to flatten anything.

    To measure 2D sparsity, we sum the absolute values of the elements in each row,
    and then count the number of rows having sum(abs(row values)) == 0.
    """
    if tensor.dim() == 4:
        # For 4D weights, 2D structures are channels (filter kernels)
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        # For 2D weights, 2D structures are either columns or rows.
        # At the moment, we only support row structures
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs / num_structs


def norm_filters(weights, p=1):
    return distiller.norms.filters_lp_norm(weights, p)


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


def log_activation_statistics(epoch, phase, loggers, collector):
    """Log information about the sparsity of the activations"""
    if collector is None:
        return
    if loggers is None:
        return
    for logger in loggers:
        logger.log_activation_statistic(phase, collector.stat_name, collector.value(), epoch)


def model_device(model):
    if isinstance(model, nn.DataParallel):
        return model.src_device_obj
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        # 模型没有参数
        pass
    return 'cpu'


def is_scalar(val):
    result = isinstance(val, torch.Tensor) and val.dim() == 0
    result |= np.isscalar(val)
    return result


def normalize_module_name(layer_name):
    """Normalize a module's name.

    PyTorch let's you parallelize the computation of a model, by wrapping a model with a
    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,
    even though the actual functionality of the module doesn't change.
    Many time, when we search for modules by name, we are indifferent to the DataParallel
    module and want to use the same module name whether the module is parallel or not.
    We call this module name normalization, and this is implemented here.
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)


def param_name_2_module_name(param_name):
    return '.'.join(param_name.split('.')[:-1])


def get_dummy_input(device=None, input_shape=None):
    """生成一个随机的dummy input
    :param device: dummy_input所在的计算设备
    :param input_shape: dummy_input的形状
    :return:
    """

    def create_single(shape):
        t = torch.randn(shape)
        if device:
            t = t.to(device)
        return t

    def create_recurse(shape):
        if all(isinstance(x, int) for x in shape):
            return create_single(shape)
        return tuple(create_recurse(s) for s in shape)

    return create_recurse(input_shape)


def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.

    torch.nn.DataParallel instances are removed.
    """

    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model


def convert_tensors_recursively_to(val, *args, **kwargs):
    """ Applies `.to(*args, **kwargs)` to each tensor inside val tree. Other values remain the same."""
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)

    if isinstance(val, (tuple, list)):
        return type(val)(convert_tensors_recursively_to(item, *args, **kwargs) for item in val)

    return val


def denormalize_module_name(parallel_model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name  # Did not find a module with the name <normalized_name>


def non_zero_channels(tensor):
    """Returns the indices of non-zero channels.

    Non-zero channels are channels that have at least one coefficient that
    is not zero.  Counting non-zero channels involves some tensor acrobatics.
    """
    if tensor.dim() != 4:
        raise ValueError("Expecting a 4D tensor")

    norms = distiller.norms.channels_lp_norm(tensor, p=1)
    nonzero_channels = torch.nonzero(norms)
    return nonzero_channels


def sparsity_ch(tensor):
    """Channel-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0
    nonzero_channels = len(non_zero_channels(tensor))
    n_channels = tensor.size(1)
    return 1 - nonzero_channels / n_channels
