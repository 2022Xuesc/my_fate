from collections import deque
import numpy as np

class Solution:
    def removeLabel(self, prob):
        # 阈值1，去除权重小于th1的边
        th1 = 0
        # 阈值2，如果预测概率大于th2，则认为标签能预测出来
        th2 = 0.5
        n = len(prob)

        inDegrees = [0] * n
        predict = [0.0] * n

        # 遍历节点对
        for i in range(n):
            for j in range(i + 1, n):
                # 遍历有向边
                con1 = prob[i][j]
                con2 = prob[j][i]

                if con1 < th1:
                    con1 = 0
                if con2 < th1:
                    con2 = 0

                if con1 < con2:
                    con1 = 0
                else:
                    con2 = 0

                # 保留了从i到j的边
                if con1 != 0:
                    inDegrees[j] += 1
                # 保留了从j到i的边
                elif con2 != 0:
                    inDegrees[i] += 1
        # 需要自主预测的标签
        save = set()
        q = deque()

        # 拓扑排序初始化
        for i in range(n):
            if inDegrees[i] == 0:
                q.append(i)
                save.add(i)
                predict[i] = 1.0

        while q:
            from_node = q.popleft()
            for to_node in range(n):
                if to_node == from_node:
                    continue

                if prob[from_node][to_node] != 0:
                    inDegrees[to_node] -= 1
                    # Todo: 使用sum还是使用max？
                    #  和图的可达性结合起来看
                    predict[to_node] = max(predict[to_node], predict[from_node] * prob[from_node][to_node])
                    # predict[to_node] += predict[from_node] * prob[from_node][to_node]

                    # 度数为0时再进行处理
                    if inDegrees[to_node] == 0:
                        # 无法辅助预测
                        if predict[to_node] < th2:
                            save.add(to_node)
                            q.append(to_node)
                            predict[to_node] = 1.0
                        # 可以辅助预测
                        else:
                            q.append(to_node)

        res = [i for i in range(n) if i not in save]
        return res


# 1. 读取概率矩阵
prob_path = "../data/prob_matrix.npy"
prob = np.load(prob_path)
vec = Solution().removeLabel(prob)
# 2. 打印移除的标签
for label in vec:
    print(label, end=" ")
