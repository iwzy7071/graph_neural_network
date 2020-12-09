import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import get_planetoid_dataset, random_planetoid_splits, run, get_summary_writer
from utils import visualize
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
net = Net(dataset)
save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model_save')
net.load_state_dict(torch.load(osp.join(save_path, '../model_save/{}_{}.pt'.format(args.model, args.dataset))))
visualize(net, dataset[0], args)

# summary_writer = get_summary_writer(args.dataset, args.model)
# run(dataset, Net(dataset), args, permute_masks, summary_writer)
