import os.path as osp
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch
import numpy as np
import random
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import torch

seed = 999
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'dataset')
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_summary_writer(dataset, model, rotate):
    path = '/home/wzy/graph_neural_network/log'
    try:
        shutil.rmtree(osp.join(path, '{}_{}_sum_{}'.format(dataset, model, rotate)))
    except Exception:
        pass
    writer = SummaryWriter(log_dir=osp.join(path, '{}_{}_sum_{}'.format(dataset, model, rotate)))
    return writer


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data


def get_rotate_dataset(dataset, rotate):
    if rotate == 'random':
        dataset.data.y = torch.randint_like(dataset.data.y, low=0, high=2)
    elif rotate == 'permutation':
        rand_index = torch.randperm(dataset.data.y.size(0))
        dataset.data.y = dataset.data.y[rand_index]

    return dataset
