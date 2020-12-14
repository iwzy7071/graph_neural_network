import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from utils import get_planetoid_dataset, random_planetoid_splits, run, get_summary_writer
from utils import visualize
import os.path as osp
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SAGE-Gaussian')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=args.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
# net = Net(dataset.num_node_features, hidden_channels=64, num_layers=2)
# save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model_save')
# net.load_state_dict(torch.load(osp.join(save_path, '../model_save/{}_{}.pt'.format(args.model, args.dataset))))
# visualize(net, dataset[0], args)
# train and test
summary_writer = get_summary_writer(args.dataset, args.model)
run(dataset, Net(dataset.num_node_features, hidden_channels=64, num_layers=2), args, permute_masks, summary_writer)
