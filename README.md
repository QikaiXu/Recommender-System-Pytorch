# Recommender System Pytorch
基于 Pytorch 实现推荐系统相关的算法。

想着看过书和论文后，能自己实现一下加深理解。



- **模型在 notebook 文件中都有实现效果；**
- 其中关于 Embedding 部分的思路及代码参考自 [pytorch-fm](https://github.com/rixwew/pytorch-fm)；



## Datasets

- MovieLens：ml-latest-small 中的 ratings.csv，共 1m 条记录；
- Criteo：截取头部 100k 条；
- Amazon Books：已经处理好的数据来源于 [DIEN-pipeline](https://github.com/kupuSs/DIEN-pipline)，截取头部 100k 条；

## Data Processing

数据处理方法参考自 [Recommender-System-with-TF2.0](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0)：

- 连续型数据：分箱后进行 One-hot 编码。
- 类别型数据：One-hot 编码。



## Available Models

| Model                                               | Paper                                                        |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Logistic Regression, LR                             |                                                              |
| Mixed Logistic Regression, MLR                      | [Kun Gai, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction, 2017.](https://arxiv.org/abs/1704.05194) |
| GBDT + LR                                           |                                                              |
| Factorization Machine, FM                           | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Field-aware Factorization Machine, FFM              | [ Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| Deep Crossing                                       | [Ying Shan, et al.Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features, 2016.](http://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) |
| Product-based Neural Network, PNN                   | [Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.](https://arxiv.org/abs/1611.00144) |
| Wide & Deep                                         | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Deep & Cross Network, DCN                           | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |
| Factorization Machine supported Neural Network, FNN | [W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.](https://arxiv.org/abs/1601.02376) |
| DeepFM                                              | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |
| Neural Factorization Machine, NFM                   | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| Attentional Factorization Machine, AFM              | [J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.](https://arxiv.org/abs/1708.04617) |
| Deep Interest Network, DIN                          | [Guorui Zhou, et al. Deep Interest Network for Click-Through Rate Prediction, 2017.](https://arxiv.org/abs/1706.06978) |
| Deep Interest Evolution Network, DIEN               | [Guorui Zhou, et al. Deep Interest Evolution Network for Click-Through Rate Prediction, 2018.](https://arxiv.org/abs/1809.03672) |
| Latent Factor Model, LFM                            |                                                              |
| Neural Collaborative Filtering, NeuralCF            | [X He, et al. Neural Collaborative Filtering, 2017.](https://arxiv.org/abs/1708.05031) |
|                                                     |                                                              |
|                                                     |                                                              |

