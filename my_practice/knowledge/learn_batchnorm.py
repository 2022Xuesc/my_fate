import torch
import torch.nn as nn

from federatedml.nn.backend.distiller.models import AlexNetBN

"""基本的batchnorm计算"""
# m = nn.BatchNorm2d(2)
# input = torch.randn(1, 2, 3, 4)
# output = m(input)
# # print(output)
#
# # print(input[0][0])
# first_dimension_mean = torch.Tensor.mean(input[0][0])
# first_dimension_var = torch.Tensor.var(input[0][0], False)
# print(m.weight)
# print(m.bias)
# # print(f'm.eps = {m.eps}')
# print(first_dimension_mean)
# print(first_dimension_var)
#
# batchnorm_one = ((input[0][0][0][0] - first_dimension_mean) / torch.pow(first_dimension_var, 0.5) + m.eps) \
#                 * m.weight[0] + m.bias[0]
#
# print(batchnorm_one)


alex_bn = AlexNetBN(num_classes=100)
print(alex_bn)