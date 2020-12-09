import torch
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', type=str)
args = parser.parse_args()

# build up dataset and model
dataset = Planetoid(root='./dataset/', name=args.dataset)
model_names = ['GCNConv', 'GATConv', 'SAGEConv', 'SGConv']
models = [eval(name)(in_channels=dataset.num_node_features, out_channels=dataset.num_classes) for name in
          model_names]
[model.load_state_dict(torch.load('./save/{}_{}.pt'.format(name, args.dataset))) for model, name in
 zip(models, model_names)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = [model.to(device) for model in models]
data = dataset.data.to(device)

with torch.no_grad():
    for model, name in zip(models, model_names):
        model.eval()
        out = model(x=data.x, edge_index=data.edge_index)
        _, pred = model(x=data.x, edge_index=data.edge_index).max(dim=1)
        ne_index = pred.ne(data.y).cpu().numpy().tolist()
        json.dump(ne_index, open('./bad_case/{}_{}'.format(name, args.dataset), 'w'))
