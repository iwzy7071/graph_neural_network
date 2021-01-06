import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import CitationFull


class LSABN(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(LSABN, self).__init__()
        self.node_convs = nn.ModuleList([])
        for index in range(num_layers):
            if index == 0:
                conv = GCNConv(in_channels=in_channels, out_channels=256)
            else:
                conv = GCNConv(in_channels=256, out_channels=256)
            self.node_convs.append(conv)
        self.node_squential = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                            nn.Linear(256, 256, bias=False))
        self.graph_sequential = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                              nn.Linear(256, 256, bias=False))

    def forward(self, data):
        x, edge_index = data.x.cuda(), data.edge_index.cuda()
        graph_embeddings = []
        for conv in self.node_convs:
            x = conv(x, edge_index)
            node_embedding = self.node_squential(x)
            node_alpha = nn.Softmax(dim=0)(node_embedding)
            node_embedding = nn.ReLU()(node_alpha * node_embedding)
            graph_embedding = node_embedding.sum(dim=0).unsqueeze(dim=0)
            graph_embeddings.append(graph_embedding)
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = self.graph_sequential(graph_embeddings)
        graph_alpha = nn.Softmax(dim=0)(graph_embeddings)
        graph_embedding = nn.ReLU()(graph_alpha * graph_embeddings).sum(dim=0)
        return graph_embedding


if __name__ == '__main__':
    dataset_s = CitationFull(root="../dataset", name="dblp")
    net = LSABN(in_channels=dataset_s.num_node_features, num_layers=3)
    net = net.cuda()
    graph_embedding = net(dataset_s.data)
