
# Mac tensorflow mnist实例入门


前期主要需要安装好tensorflow的环境，Mac 如果只涉及到CPU的版本，推荐使用pip3，傻瓜式安装，一行命令！代码使用python3。

在此附上个人git完整代码地址：https://github.com/Liuyubao/Tensorflow_mnist

```
sudo pip3 install tensorflow
```

开堂测试
----
下面是一些会涉及到的概念，可以参考[谷歌机器学习术语表](https://developers.google.cn/machine-learning/crash-course/glossary)。

训练集
测试集
特征
损失函数
激活函数
准确率
偏差
梯度下降


数据集
---

当我们开始学习编程的时候，第一件事往往是学习打印"Hello World"。正如编程入门有Hello World，机器学习入门有MNIST。

MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/MNIST.png)


```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
```

训练集 测试集 验证集

![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/MNIST-Matrix.png)

每一张图片包含28X28个像素点。我们可以用一个数字数组来表示这张图片：
我们把这个数组展开成一个向量，长度是 28x28 = 784。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在784维向量空间里面的点, 并且拥有比较复杂的结构 (提醒: 此类数据的可视化是计算密集型的)。

训练数据的特征：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist-train-xs.png)

训练数据的label：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist-train-ys.png)
相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。为了用于这个教程，我们使标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。所以在此教程中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。因此， mnist.train.labels 是一个 [60000, 10] 的数字矩阵。


Softmax回归介绍
-----------

我们知道MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。

这是一个使用softmax回归（softmax regression）模型的经典案例。softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率。

softmax回归（softmax regression）分两步：第一步

为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。

下面的图片显示了一个模型学习到的图片上每个像素对于特定数字类的权值。红色代表负数权值，蓝色代表正数权值。
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/softmax-weights.png)


我们也需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist1.png)

代表权重，
代表数字 i 类的偏置量，
j 代表给定图片 x 的像素索引用于像素求和。然后用softmax函数可以把这些证据转换成概率 y：

![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist4.png)

![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist5.png)
展开等式右边的子式，可以得到：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist6.png)

用图来看流程：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/softmax-regression-scalargraph.png)

如果把它写成一个等式，我们可以得到：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/softmax-regression-scalarequation.png)
我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加。这有助于提高计算效率。（也是一种更有效的思考方式）
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/softmax-regression-vectorequation.png)

更进一步，可以写成更加紧凑的方式：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist7.png)



实现回归模型
------

为了用python实现高效的数值计算，我们通常会使用函数库，比如NumPy，会把类似矩阵乘法这样的复杂运算使用其他外部语言实现。不幸的是，从外部计算切换回Python的每一个操作，仍然是一个很大的开销。如果你用GPU来进行外部计算，这样的开销会更大。用分布式的计算方式，也会花费更多的资源用来传输数据。

TensorFlow也把复杂的计算放在python之外完成，但是为了避免前面说的那些开销，它做了进一步完善。Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。）

使用TensorFlow之前，首先导入它：

```
import tensorflow as tf
sess = tf.InteractiveSession()
```

我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：

```
x = tf.placeholder(tf.float32, [None, 784])
```

x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）

我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：Variable 。 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

我们赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。

注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。

现在，我们可以实现我们的模型啦。只需要一行代码！

```
y = tf.nn.softmax(tf.matmul(x,W) + b)
```

训练模型
----

为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是，这两种方式是相同的。

一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist10.png)

y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。更详细的关于交叉熵的解释超出本教程的范畴，但是你很有必要好好理解它。

为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正y_ = tf.placeholder("float", [None,10])们可以用
![这里写图片描述](http://wiki.jikexueyuan.com/project/tensorflow-zh/images/mnist9.png)

计算交叉熵:

```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))
```

首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。

现在我们知道我们需要我们的模型做什么啦，用TensorFlow来训练它是非常容易的。因为TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。

```
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
```

在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。

TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。






然后开始训练模型，这里我们让模型循环训练1000次！

```
tf.global_variables_initializer().run()
for i in range(1000):
...     batch_xs, batch_ys = mnist.train.next_batch(100)
...     train_step.run({x: batch_xs, y_: batch_ys})
```
该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。


评估我们的模型
-------

那么我们的模型性能如何呢？

首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.

```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```

最后，我们计算所学习到的模型在测试数据集上面的正确率。

```
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
```

![这里写图片描述](//img-blog.csdn.net/20180321200059899?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L1l1YmFvTG91aXNMaXU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

0.9101，这个结果其实比较一般。这是因为我们仅仅使用了一个非常简单的模型。如果加上一些简单的改进，比如卷积神经网络等，能够轻松0.99+。博主也会继续深入讲解tensorflow，欢迎大家关注讨论。

Reference
---------
1、代码主要参考《Tensorflow实战》黄文坚 唐源；
2、图片文字材料主要参考极客学院深度学习模块；

github代码
----------

如果本博客对您有帮助，希望可以得到您的赞赏！
完整代码附上：https://github.com/Liuyubao/Tensorflow_mnist
