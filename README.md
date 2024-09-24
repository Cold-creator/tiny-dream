# tiny-dream

This repository was created for an AISummerCamp of 2024, it was named after tiny-dream since we regard this fantastic experience as a tiny dream. Who knows? Maybe we can really make sth in the end.

# 使用版本条件

根据组内调试情况，由于许多函数方法都随着库更新改掉了，部分库需要不同版本，
这里以能够运行的版本（队长邓楚枫的环境配置)为例子，要求情感分析系列代码所需环境（仅为例子，其他版本应该也可以运行）：
pip install gensim==3.8.1
pip install numpy==1.19.5 
pip install tensorflow==2.3.3 
pip install protobuf==3.20.1
pip install torch==1.13.1
pip install torchvision==0.14.1 
pip install torchtext==0.14.1 

# 说明

我们通过两种方案实现了文本情感预测，第一种的原理是：
将微博文本库提取，比如“1 我非常开心”，“1”代表这句话的情绪，后面是这句话本身。
先将data文件切分成词向量，如“1 我 非常 开心”这样，然后转成诸如“-0.9212，0.9451，0.84646”的向量形式（仅为例子），
之后进行学习，训练模型，最终进行预测。（基于word2vec和SVM，7：3的训练集和测试集）

它能够区分多种情感（"表达开心","表达伤心","表达恶心","表达生气","表达害怕","表达惊喜"），而且制作了能保留历史记录的检索表。

第二种则运用了LTSM算法，不过较单一，只能判断是active还是negative，但是准确率更高。
