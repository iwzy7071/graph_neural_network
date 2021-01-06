from torch import nn
from torch_geometric.nn import GCNConv


# TODO: https://blog.csdn.net/weixin_40239306/article/details/108819105

class UDAGCN(nn.Module):
    def __init__(self, dim_input, dim_embeddings, dim_target):
        super().__init__()
        pass
