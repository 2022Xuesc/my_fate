""" A TensorBoard backend.

Write logs to a file using a Google's TensorBoard protobuf format.
See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
"""
import os

import warnings
import tensorflow as tf

# Disable FutureWarning from TensorFlow
warnings.simplefilter(action='ignore', category=FutureWarning)


class TBBackend(object):
    def __init__(self, log_dir):
        self.writers = []
        self.log_dir = log_dir
        self.writers.append(tf.summary.create_file_writer(log_dir))

    def scalar_summary(self, tag, scalar, step):
        # 获取对应的writer
        writer = self.writers[0]
        with writer.as_default():
            # tag表示标量的名称，scalar表示标量值，step是当前迭代位置
            tf.summary.scalar(tag, scalar, step=step)

    # Todo: 对tensorflow v1版本进行升级，迁移到v2版本
    def list_summary(self, tag, data_list, step, multi_graphs):
        """记录一个标量列表

        我们需要去追踪单个图中多个标量参数的更新进展
        该列表提供了我们追踪的每个参数的一个值
        有两种方式达到这个目标，并且各有优劣
        1. 使用单个writer：所有参数使用相同的颜色，很难辨别
        2. 使用多个writer：每个参数都有独特的颜色，然而，每个writer向不同的文件中记录日志并且会创建很多文件，使得TB加载变慢
        """
        for i, scalar in enumerate(data_list):
            # 如果采用多图的方式，且writer数量不够时，
            if multi_graphs and (i + 1 > len(self.writers)):
                self.writers.append(tf.summary.create_file_writer(os.path.join(self.log_dir, str(i))))
            # 计算得到对应的writer
            writer = self.writers[0 if not multi_graphs else i]
            with writer.as_default:
                tf.summary.scalar(tag, scalar, step=step)

    def histogram_summary(self, tag, tensor, step):
        writer = self.writers[0]
        with writer.as_default:
            tf.summary.histogram(tag, tensor, step=step, buckets=200)

    def sync_to_file(self):
        for writer in self.writers:
            writer.flush()
