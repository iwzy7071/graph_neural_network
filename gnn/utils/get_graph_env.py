import os.path as osp
import numpy as np
import random
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import torch
from models import GNN
from utils.utils import get_rotate_dataset

seed = 999
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def get_large_graph_env(args):
    root_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'dataset')
    dataset = PygGraphPropPredDataset(name=args.dataset, root=root_path)
    if args.rotate != 'none':
        dataset = get_rotate_dataset(dataset, args.rotate)
    net = GNN(gnn_type=args.model, num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
              drop_ratio=args.drop_ratio, virtual_node=False, graph_pooling='sum')
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, net
