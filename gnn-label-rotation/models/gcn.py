import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, dim_features, dim_target):
        super().__init__()
        num_layers = 4
        dim_embedding = 64
        self.fc_max = nn.Linear(dim_embedding, dim_embedding)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding
            conv = GCNConv(dim_input, dim_embedding)
            self.layers.append(conv)

        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
