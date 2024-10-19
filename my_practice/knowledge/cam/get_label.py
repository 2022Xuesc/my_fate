from collections import deque


class Solution:
    def removeLabel(self, prob):
        th1 = 0.1
        th2 = 0.5
        n = len(prob)

        inDegrees = [0] * n
        predict = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
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

                if con1 != 0:
                    inDegrees[j] += 1
                elif con2 != 0:
                    inDegrees[i] += 1

        save = set()
        q = deque()

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
                    predict[to_node] = max(predict[to_node], predict[from_node] * prob[from_node][to_node])

                    if inDegrees[to_node] == 0 and predict[to_node] < th2:
                        save.add(to_node)
                        q.append(to_node)
                        predict[to_node] = 1.0
                    elif predict[to_node] > th2:
                        q.append(to_node)

        res = [i for i in range(n) if i not in save]
        return res


# Given input is an n * n label probability matrix
prob = [[1, 0.8, 0], [0.2, 1, 0], [0, 0, 1]]
vec = Solution().removeLabel(prob)

# Remove the labels indicated by the result vector
for label in vec:
    print(label, end=" ")
