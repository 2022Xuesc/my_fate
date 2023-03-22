class _Regularizer(object):
    # 加上正则损失的超参数
    def __init__(self, name, lamda=1):
        self.name = name
        self.lamda = lamda

    def loss(self, param, regularizer_loss):
        raise NotImplementedError
