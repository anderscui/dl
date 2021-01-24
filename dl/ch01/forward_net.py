# coding=utf-8
import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        w, b = self.params
        out = np.dot(x, w) + b
        return out


class Sigmoid:
    def __init__(self, params=None):
        if params is None:
            params = []
        self.params = params

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重与偏置
        w1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        w2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [Affine(w1, b1),
                       Sigmoid(),
                       Affine(w2, b2)]

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == '__main__':
    # predict by nn
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    scores = model.predict(x)
    print(scores.shape)
    print(scores)
