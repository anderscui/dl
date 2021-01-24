# dl - 深度学习进阶 - NLP

[书籍链接]([https://book.douban.com/subject/35225413/)

# 前言

## 主要内容

* 基于 Python 的文本处理
* DL 中的单词表示法
* 获取单词向量的 word2vec（CBOW 与 skip-gram）
* 加快大规模数据训练速度的 Negative Sampling
* 处理时序数据的 RNN、LSTM 与 GRU
* 处理时序数据的误差反向传播法
* 进行文本生成的神经网络
* seq2seq
* 关注重要信息的 Attention

## 使用的语言与库

* Python 3
* numpy
* matplotlib
* CuPy

PS:

* [100 numpy exercises](https://github.com/rougier/numpy-100)
* 

# 第一章 神经网络的复习

本书依然坚持"从零开始创建"，通过上手实践，探寻深度学习相关的乐趣。

## 1.1 数学与 Python 的复习

### 1.1.1 向量与矩阵

向量和矩阵分别可用一维数组和二维数组表示，如果推广到 N 维，则得到[张量](https://en.wikipedia.org/wiki/Tensor)。

在深度学习中，一般将向量作为**列向量**处理，本书则通常将其作为**行向量**处理。

```python
import numpy as np

x = np.array([1, 2, 3])
type(x), x.shape, x.ndim
# (numpy.ndarray, (3,), 1)

W = np.array([[1, 2, 3], [4, 5, 6]])
type(W), W.shape, W.ndim
# ((2, 3), 2)
```

### 1.1.4 向量内积和矩阵乘积

$x \cdot y = x_1y_1 + x_2y_2 + \cdots + x_ny_n$

向量内积是两个同样shape的向量的对应元素乘积之和，又称为点积、数量积，因其结果为标量。内积可直观地衡量两个向量夹角大小（除以向量大小之积后）。

矩阵乘积可视为内积的推广（左侧矩阵行向量与右侧矩阵列向量之积）。故而，两种乘积都使用 `np.dot` 函数。

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b), a.dot(b)
# (32, 32)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# == A.dot(B)
np.dot(A, B)
# array([[19, 22],
#        [43, 50]])
```

### 1.1.5 矩阵形状检查

形状分别为 $m \cdot n$ 和 $n \cdot p$ 的矩阵之积为 $m \cdot p$，矩阵乘积对形状有严格要求。在深度学习中存在大量矩阵乘积，形状检查是一个简单而有效的调试方法。

## 1.2 神经网络的推理

### 1.2.1 神经网络推理全貌

神经网络中进行的处理分为学习和推理两部分，本节介绍推理部分。

![nn layers](./images/nn_layers.png)

上图为最简单的神经网络结构，输入层有3个神经元，隐藏层有4个，输出层有2个。圆圈表示”神经元“，箭头表示神经元之间的连接。箭头上带有”权重“，权重与对应神经元相乘，乘积之和（一般是经过激活函数变换后）作为下一个神经元的输入。此外，还要加上一个不受前一层神经元影响的常数，通常称为”偏置“。因为所有相邻神经元之间都存在”连接“，因此这样的神经网络又称为”全连接网络“。

上图中的网络，因为有三层，故有的文献称之为3层网络，而本书则称之为2层的，因其仅有2层（输入和隐藏）具有权重。上图之网络涉及术语：

* 神经元（neuron）
* 层（layer）
* 权重（weight）
* 偏置（bias）
* 全连接网络（fully connected network）

PS：上图从网络中寻得，但形状与书中不同，为保持一致，现假装该图三层神经元数量分别是：2、4、3。

用 $(x_1, x_2)$ 表示输入层数据，权重分别为 $w_{11}, w_{21}$，相应偏置为 $b_1$，则隐藏层第一个元素可以如此计算：

$$h_1 = x_1w_{11} + x2w_{21} + b_1$$

重复计算隐藏层其它神经元，发现基于全连接的变换可表示为矩阵乘积。

$$h = xW + b$$

形状说明：x 是 (1, 2)，W 是 (2, 4)，b 是 (1, 4)，如此，h 也是 (1, 4)。可以看到这里的向量都是”行向量“。这个乘积计算的是”一笔“数据，即一个输入。在 mini-batch 的情形下，输入可表示为矩阵。

在这一的表示下，x 的每一行表示一个**输入**，h 的每一行是对应**输入**产生的隐藏层。代码实现如下：

```python
W1 = np.random.randn(2, 4) # 权重
b1 = np.random.randn(4) # 偏置

x = np.random.randn(10, 2) # 10笔数据
h = np.dot(x, W1) + b1
```

#### 1.2.1.1 激活函数

按上节的计算方式，矩阵相当于对**输入**作了一个**线性变换**，如果没有其它操作介入，那么全连接网络也只是一个线性变换，这显然会拟合不足，弥补这一点的“其它操作”就是**激活函数**。常见的一个函数是**sigmoid 函数**：

$$\sigma(x) = \frac{1}{1 + exp(-x)}$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = sigmoid(h)
```

应用 sigmoid 激活 h 后，接下来是隐藏层到输出层，这时需要有类似的权重和偏置，shape 分别是 (4, 3) 和 (1, 3)。完整代码是：

```python
x = np.random.randn(10, 2)

# 到隐藏层
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
h = np.dot(x, W1) + b1
a = sigmoid(h)

# 到输出层
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)
s = np.dot(a, W2) + b2
s.shape
# (10, 3)
```

使用矩阵表示后，繁琐的计算变得颇为简洁。这里的 s 即是输出层，每一行是一个“输出”，是3维向量。假设这是一个分类问题，3维对应3个分类的得分，那么得分最高的那个是分类结果。得分不是概率，通过把得分输入 Softmax 函数，可以转换为概率。

到这里，神经网络的**推理**部分实现了。

### 1.2.2 层的类化和正向传播

全连接层相当于几何学中的仿射变换（线性变换+平移，Affine transformation），因此全连接层实现为 `Affine` 层；将 sigmoid 函数的变换实现为 `Sigmoid` 层。

神经网络的推理相当于网络中的**正向传播**，而学习则对应**反向传播**。

在本书中，各种层实现为 Python 类，将它们模块化，之后则像乐高积木一样搭建网络。本书的实现有以下规范：

* 所有层都有 `forward()`、`backward()` 方法
* 所有层都有 `params`、`grads` 字段

params 保存权重与偏置，grads 保存参数对应的梯度。对于正向传播来说，仅关注以下两点：`forward()`、`params`。

下面的代码实现了两层网络的**推理**，看起来有点样子了。

```python
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
```

## 1.3 神经网络的学习

看起来，**推理**相对来说是简单的，**学习**才是困难的部分，而且先学习了才能真正进行推理。

