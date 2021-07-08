# A simple Word2vec Tutorial

references:

[A simple Word2vec Tutorial](https://medium.com/@zafaralibagh6/a-simple-word2vec-tutorial-61e64e38a6a1)

[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)

[A hands-on intuitive approach to Deep Learning Methods for Text Data — Word2Vec, GloVe and FastText](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)

# 引子

在 MBTI 或 Big Five 人格测试中，我们可以使用4或5个特征来描述人格，将特征们表示为向量，就容易了解两个人的相似度。这种方法的核心之处是：

* 我们可以将人/物表示为向量（因此便于机器处理）
* 可以容易地计算两个人/物之间的相似度

# Word Embedding

通过 GloVe 的预训练模型可以看到，在少数维的训练中，可以捕获很多微妙的词义的特征，有趣的是，虽然不清楚每个维度的具体含义，我们可以通过词之间的对比来确认它们确实捕获到了些什么。

# 同义词

Word2Vec 的一个让人印象极为深刻的例子是“相似词（Analogy）”，如 `king - man + woman ~ queen` 。

# Language Modeling


