#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import federatedml.nn.backend.distiller as distiller
import federatedml.nn.backend.distiller.utils
import numpy as np

import logging

msglogger = logging.getLogger()


def dataset_summary(data_loader):
    """ 创建一个数据集内类成员分布的直方图

    用于检查训练、验证以及测试集，确保其是平衡的
    # Todo: 用来说明IID和NON-IID的情况
    """
    msglogger.info("开始分析数据集: ")

    all_labels = []

    # 每50个batch输出一次遍历进度
    print_frequency = 50
    for batch, (input, label_batch) in enumerate(data_loader):
        all_labels = np.append(all_labels, distiller.utils.to_np(label_batch))
        if (batch + 1) % print_frequency == 0:
            # 进度指示器
            print("已执行到批次 : %d" % batch)

    hist = np.histogram(all_labels, bins=np.arange(1000 + 1))
    nclasses = len(hist[0])
    for data_class, size in enumerate(hist[0]):
        msglogger.info("\t类 {} = {}".format(data_class, size))
    msglogger.info("数据集中包含 {} 个样本".format(len(data_loader.sampler)))
    msglogger.info("已发现 {} 类别".format(nclasses))
    msglogger.info("每个类别中平均包含: {} 样本".format(np.mean(hist[0])))
