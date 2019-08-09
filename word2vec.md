### 语言模型

在统计自然语言处理中，语言模型指的是计算一个句子的概率模型。

传统的语言模型中词的表示是原始的、面向字符串的。两个语义相似的词的字符串可能完全不同，比如“番茄”和“西红柿”。这给所有NLP任务都带来了挑战——字符串本身无法储存语义信息。该挑战突出表现在模型的平滑问题上：标注语料是有限的，而语言整体是无限的，传统模型无法借力未标注的海量语料，只能靠人工设计平滑算法，而这些算法往往效果甚微。

神经概率语言模型（Neural Probabilistic Language Model）中词的表示是向量形式、面向语义的。两个语义相似的词对应的向量也是相似的，具体反映在夹角或距离上。甚至一些语义相似的二元词组中的词语对应的向量做线性减法之后得到的向量依然是相似的。词的向量表示可以显著提高传统NLP任务的性能，例如《[基于神经网络的高性能依存句法分析器](http://www.hankcs.com/nlp/parsing/neural-network-based-dependency-parser.html)》中介绍的词、词性、依存关系的向量化对正确率的提升等。

从向量的角度来看，字符串形式的词语其实是更高维、更稀疏的向量。若词汇表大小为N，每个字符串形式的词语字典序为i，则其被表示为一个N维向量，该向量的第i维为1，其他维都为0。汉语的词汇量大约在十万这个量级，十万维的向量对计算来讲绝对是个维度灾难。而word2vec得到的词的向量形式（下文简称“词向量”，更学术化的翻译是“词嵌入”）则可以自由控制维度，一般是100左右。

### word2vec

word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习**某个语言模型**而产生的中间结果。具体来说，“某个语言模型”指的是“CBOW”和“Skip-gram”。具体学习过程会用到两个降低复杂度的近似方法——Hierarchical Softmax或Negative Sampling。两个模型乘以两种方法，一共有四种实现。这些内容就是本文理论部分要详细阐明的全部了。

## Hierarchical Softmax

### 模型共同点

无论是哪种模型，其基本网络结构都是在下图的基础上，省略掉hidden layer：

![屏幕快照 2016-07-14 下午7.30.23.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5to0uwydsj216i0ikdi3.jpg)

为什么要去掉这一层呢？据说是因为word2vec的作者嫌从hidden layer到output layer的矩阵运算太多了。于是两种模型的网络结构是：

![屏幕快照 2016-07-14 下午7.27.35.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5to6e5d9lj216c0qkwhk.jpg)

其中w(t)代表当前词语位于句子的位置t，同理定义其他记号。在窗口内（上图为窗口大小为5），除了当前词语之外的其他词语共同构成上下文。

### CBOW

#### 原理

CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。其图示如上图左。

CBOW是已知上下文，估算当前词语的语言模型。其学习目标是最大化对数似然函数：

![屏幕快照 2016-07-14 下午7.45.11.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5tocs4lykj20fa03aaa8.jpg)

其中，w表示语料库C中任意一个词。从上图可以看出，对于CBOW，

**输入层**是上下文的词语的词向量（什么！我们不是在训练词向量吗？不不不，我们是在训练CBOW模型，词向量只是个副产品，确切来说，是CBOW模型的一个参数。训练开始的时候，词向量是个随机值，随着训练的进行不断被更新）。

**投影层**对其求和，所谓求和，就是简单的向量加法。

**输出层**输出最可能的w。由于语料库中词汇量是固定的|C|个，所以上述过程其实可以看做一个多分类问题。给定特征，从|C|个分类中挑一个。

对于神经网络模型多分类，最朴素的做法是softmax回归：

![神经网络依存句法分析29.png](http://ww2.sinaimg.cn/large/6cbb8645gw1exxdsuugv1j20cl033jrj.jpg)

softmax回归需要对语料库中每个词语（类）都计算一遍输出概率并进行归一化，在几十万词汇量的语料上无疑是令人头疼的。

不用softmax怎么样？比如SVM中的多分类，我们都知道其多分类是由二分类组合而来的：

![svm_多分类.gif](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wmvf9tbrg20bf08mq30.gif)



这是一种二叉树结构，应用到word2vec中被作者称为Hierarchical Softmax：

![屏幕快照 2016-07-17 上午9.13.40.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5wmy4jdnwj214w12a42v.jpg)

上图输出层的树形结构即为Hierarchical Softmax。

非叶子节点相当于一个神经元（感知机，我认为逻辑斯谛回归就是感知机的输出代入f(x)=1/(1+e^x)），二分类决策输出1或0，分别代表向下左转或向下右转；每个叶子节点代表语料库中的一个词语，于是每个词语都可以被01唯一地编码，并且其编码序列对应一个事件序列，于是我们可以计算条件概率![屏幕快照 2016-07-17 上午10.05.33.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wofvax07j208u0283yg.jpg)。

在开始计算之前，还是得引入一些符号：

1. ![屏幕快照 2016-07-17 上午9.59.45.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wo9rldvej201o01kt8h.jpg)从根结点出发到达w对应叶子结点的路径.

2. ![屏幕快照 2016-07-17 上午10.00.06.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5woa6hrx9j201c01ojr5.jpg)路径中包含结点的个数

3. ![屏幕快照 2016-07-17 上午10.01.17.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wobigm5sj207m01wmx3.jpg)路径![屏幕快照 2016-07-17 上午9.59.45.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wo9rldvej201o01kt8h.jpg)中的各个节点

4. ![屏幕快照 2016-07-17 上午10.02.33.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5woco1brlj20c001ywei.jpg)词w的编码，![屏幕快照 2016-07-17 上午10.03.27.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wodgv934j201q020742.jpg)表示路径![屏幕快照 2016-07-17 上午9.59.45.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wo9rldvej201o01kt8h.jpg)第j个节点对应的编码（根节点无编码）

5. ![屏幕快照 2016-07-17 上午10.04.18.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5woefv8p6j20bo01udfv.jpg)路径![屏幕快照 2016-07-17 上午9.59.45.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wo9rldvej201o01kt8h.jpg)中非叶节点对应的参数向量

   于是可以给出w的条件概率：

   

![屏幕快照 2016-07-17 上午10.07.18.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wohk3pi7j20ka042t91.jpg)

这是个简单明了的式子，从根节点到叶节点经过了![屏幕快照 2016-07-17 上午10.00.06.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5woa6hrx9j201c01ojr5.jpg)-1个节点，编码从下标2开始（根节点无编码），对应的参数向量下标从1开始（根节点为1）。

其中，每一项是一个逻辑斯谛回归：

![屏幕快照 2016-07-17 上午10.15.37.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5woq5oeojj20pm05yt9a.jpg)

考虑到d只有0和1两种取值，我们可以用指数形式方便地将其写到一起：

![屏幕快照 2016-07-17 上午10.21.31.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wowc2oi2j20qm02mjrs.jpg)

我们的目标函数取对数似然：

![屏幕快照 2016-07-17 上午10.23.25.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5woyikyp0j20fc03kgls.jpg)

将![屏幕快照 2016-07-17 上午10.05.33.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wofvax07j208u0283yg.jpg)代入上式，有

![屏幕快照 2016-07-17 上午10.25.37.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wp0mws5gj20zi086gn9.jpg)

这也很直白，连乘的对数换成求和。不过还是有点长，我们把每一项简记为：

![屏幕快照 2016-07-17 上午10.27.15.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wp2ahoblj20ug02adge.jpg)

怎么最大化对数似然函数呢？分别最大化每一项即可（这应该是一种近似，最大化某一项不一定使整体增大，具体收敛的证明还不清楚）。怎么最大化每一项呢？先求函数对每个变量的偏导数，对每一个样本，代入偏导数表达式得到函数在该维度的增长梯度，然后让对应参数加上这个梯度，函数在这个维度上就增长了。这种白话描述的算法在学术上叫随机梯度上升法，详见[更规范的描述](http://www.hankcs.com/ml/the-logistic-regression-and-the-maximum-entropy-model.html#h3-6)。

每一项有两个参数，一个是每个节点的参数向量![屏幕快照 2016-07-17 上午10.58.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wpyg370hj202i022t8i.jpg)，另一个是输出层的输入![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)，我们分别对其求偏导数：

![屏幕快照 2016-07-17 上午10.52.59.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wpt4xg8tj212603q756.jpg)

因为sigmoid函数的导数有个很棒的形式：

![屏幕快照 2016-07-17 上午10.54.30.png](http://ww3.sinaimg.cn/large/6cbb8645gw1f5wpus333qj20bi02gmx5.jpg)

于是代入上上式得到：

![屏幕快照 2016-07-17 上午10.56.15.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wpwixaknj20ne02kt93.jpg)

合并同类项得到：

![屏幕快照 2016-07-17 上午10.57.17.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wpxj61wjj20co02k0ss.jpg)

于是![屏幕快照 2016-07-17 上午10.58.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wpyg370hj202i022t8i.jpg)的更新表达式就得到了：

![屏幕快照 2016-07-17 上午10.59.08.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wpzj0m0zj20l2026t90.jpg)

其中，![屏幕快照 2016-07-17 上午10.59.48.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wq02oflgj201e01k741.jpg)是机器学习的老相好——学习率，通常取0-1之间的一个值。学习率越大训练速度越快，但目标函数容易在局部区域来回抖动。

再来![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)的偏导数，注意到![屏幕快照 2016-07-17 上午10.27.15.png](http://ww1.sinaimg.cn/large/6cbb8645gw1f5wp2ahoblj20ug02adge.jpg)中![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)和![屏幕快照 2016-07-17 上午10.58.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wpyg370hj202i022t8i.jpg)是对称的，所有直接将![屏幕快照 2016-07-17 上午10.58.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wpyg370hj202i022t8i.jpg)的偏导数中的![屏幕快照 2016-07-17 上午10.58.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wpyg370hj202i022t8i.jpg)替换为![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)，得到关于![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)的偏导数：

![屏幕快照 2016-07-17 上午11.04.49.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wq5cfqknj20je03qt92.jpg)

于是![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)的更新表达式也得到了。

不过![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)是上下文的词向量的和，不是上下文单个词的词向量。怎么把这个更新量应用到单个词的词向量上去呢？word2vec采取的是直接将![屏幕快照 2016-07-17 上午10.52.10.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wps67xq7j201m01kmwx.jpg)的更新量整个应用到每个单词的词向量上去：

![屏幕快照 2016-07-17 上午11.11.33.png](http://ww4.sinaimg.cn/large/6cbb8645gw1f5wqcddww7j20q6042wf4.jpg)

其中，![屏幕快照 2016-07-17 上午11.11.46.png](http://ww2.sinaimg.cn/large/6cbb8645gw1f5wqcuc7mvj202q01yjr7.jpg)代表上下文中某一个单词的词向量。我认为应该也可以将其平均后更新到每个词向量上去，无非是学习率的不同，欢迎指正。

转自http://www.hankcs.com/nlp/word2vec.html