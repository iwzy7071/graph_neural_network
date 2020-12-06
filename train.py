import torch
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCNConv', type=str)
parser.add_argument('--dataset', default='cora', type=str)
args = parser.parse_args()
shutil.rmtree(join('./log', args.model))
writer = SummaryWriter(log_dir=join('./log', args.model))

# build up dataset and model
dataset = Planetoid(root='./dataset/', name=args.dataset)
model = eval(args.model)(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = dataset.data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_acc = 0
for epoch in tqdm(range(200)):
    model.train()
    optimizer.zero_grad()
    out = model(x=data.x, edge_index=data.edge_index)
    _, pred = model(x=data.x, edge_index=data.edge_index).max(dim=1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    acc = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()) / int(data.train_mask.sum())
    writer.add_scalar('train_loss', loss.item(), epoch)
    writer.add_scalar('train_acc', acc, epoch)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        writer.add_scalar('test_loss', loss.item(), epoch)
        writer.add_scalar('test_acc', acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), join('.', 'save', '{}.pt'.format(args.model)))
