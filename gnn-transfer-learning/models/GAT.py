from torch import nn
from torch_geometric.nn import GATConv, global_max_pool
import torch
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, dim_features, dim_target):
        super().__init__()
        num_layers = 4
        dim_embedding = 1024
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding
            conv = GATConv(dim_input, dim_embedding)
            self.layers.append(conv)
        self.fc = nn.Linear(dim_embedding, dim_target)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = torch.relu(x)
        return self.fc(x)
