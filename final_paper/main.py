import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import os.path as osp

root_path = '/home/wzy/graph_neural_network/dataset'
dataset = PygGraphPropPredDataset(name='ogbn-products', root=root_path)
split_idx = dataset.get_idx_split()
