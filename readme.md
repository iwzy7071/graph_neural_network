### Week 1
#### 阅读GNNPapers https://github.com/thunlp/GNNPapers
- 先从survey看起
- 不用复现很多的basic model
#### 阅读论文原文
- GCN https://arxiv.org/abs/1609.02907
- GraphSAGE https://arxiv.org/abs/1706.02216
- SGC https://arxiv.org/abs/1902.07153
- GAT https://arxiv.org/abs/1710.10903
#### Mistake Analysis
- 不同的model，比如gcn，gat这些，在同样的setting下train和test，把test中这些不同model犯的错的点找出来 判断
- 每个model犯错的点的特征（自己的label，feature，邻居的label feature 等等）
- 不同model犯错的点之间的重合情况