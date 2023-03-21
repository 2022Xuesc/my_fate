"""
    该类被用来追踪训练时表现最好的epoch
"""
import operator
import federatedml.nn.backend.distiller as distiller
import federatedml.nn.backend.distiller.utils

__all__ = ["TrainingPerformanceTracker",
           "SparsityAccuracyTracker"]


# 该类是用来比较对相同数据上相同模型的压缩任务的优劣
class TrainingPerformanceTracker(object):
    """Base class for performance trackers using Top1 and Top5 accuracy metrics"""
    def __init__(self, num_best_scores):
        self.perf_scores_history = []
        self.max_len = num_best_scores

    def reset(self):
        self.perf_scores_history = []

    # 该函数中传入model以及epoch还有其他指标；便于汇总后进行比较
    def step(self, model, epoch, **kwargs):
        """更新已经完成的最高训练分数的列表"""
        raise NotImplementedError

    def best_scores(self, how_many=1):
        """返回how_many个最好的结果"""
        if how_many < 1:
            how_many = self.max_len
        how_many = min(how_many, self.max_len)
        return self.perf_scores_history[:how_many]


class SparsityAccuracyTracker(TrainingPerformanceTracker):
    """ 主要关注非零参数的性能追踪器

    将非零参数的个数作为主键对性能历史进行排序，然后使用top1、top5以及epoch数目等指标

    top1和top5应该出现在**kwargs位置上
    """
    def step(self, model, epoch, **kwargs):
        assert all(score in kwargs.keys() for score in ('top1', 'top5'))
        # Todo: 此处调用distiller.utils的方法来计算模型的稀疏程度以及非零参数的个数。
        model_sparsity, _, params_nnz_cnt = distiller.utils.model_params_stats(model)
        self.perf_scores_history.append(distiller.utils.MutableNamedTuple({
            # Todo: 这里为什么是负数？debug
            'params_nnz_cnt': -params_nnz_cnt,
            'sparsity': model_sparsity,
            'top1': kwargs['top1'],
            'top5': kwargs['top5'],
            'epoch': epoch}))
        # 保持训练历史记录从优到劣的排序
        self.perf_scores_history.sort(
            key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'),
            reverse=True)
