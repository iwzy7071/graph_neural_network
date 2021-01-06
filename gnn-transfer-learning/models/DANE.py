from torch import nn
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.datasets import CitationFull
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import NeighborSampler
from torch_cluster import random_walk
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class NegativeNeighborSampler(NeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]
        neg_batch = torch.randint(0, self.adj_t.size(0), (batch.numel(),), dtype=torch.long)
        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NegativeNeighborSampler, self).sample(batch)


class Generator(nn.Module):
    def __init__(self, dim_input, dim_target):
        super(Generator, self).__init__()
        self.gcns = nn.ModuleList([SAGEConv(dim_input, 128), SAGEConv(128, 64)])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.gcns[i]((x, x_target), edge_index)
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.relu()
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.gcns):
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features_to_real_number = nn.Sequential(nn.Linear(64, 32), nn.Dropout(p=0.3), nn.Linear(32, 2))

    def forward(self, x):
        return self.features_to_real_number(x)


def calculate_gcn_loss(x, dataloader):
    total_loss = 0
    for batch_size, n_id, adjs in dataloader:
        adjs = [adj.to('cuda') for adj in adjs]
        out = generator(x[n_id].cuda(), adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
    return total_loss


# SET UP TRAINING ENVIRONMENT
dataset_s = CitationFull(root="../dataset", name="dblp")
dataset_t = CitationFull(root='../dataset', name='dblp')
data_s, data_t = dataset_s.data, dataset_t.data
x_s, x_t = data_s.x.cuda(), data_t.x.cuda()
edge_s, edge_t = data_s.edge_index.cuda(), data_t.edge_index.cuda()
dataloader_s = NegativeNeighborSampler(data_s.edge_index, sizes=[25, 10], shuffle=True, batch_size=20000)
dataloader_t = NegativeNeighborSampler(data_t.edge_index, sizes=[25, 10], shuffle=True, batch_size=20000)
generator, discriminator = Generator(dataset_s.num_features, dataset_s.num_classes), Discriminator()
generator, discriminator = generator.cuda(), discriminator.cuda()
gcn_loss = calculate_gcn_loss(x_s, dataloader_s) + calculate_gcn_loss(x_t, dataloader_t)
cls_criterion = nn.CrossEntropyLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=1.5e-3, weight_decay=5e-5)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1.5e-3)
scheduler_g = StepLR(optimizer_g, step_size=100, gamma=0.8)
scheduler_d = StepLR(optimizer_d, step_size=100, gamma=0.8)

# START TRAINING
for epoch in range(1000):
    if epoch > 500:
        scheduler_g.step()
        scheduler_d.step()
    d_loss_all = []

    # for _ in range(10):
    #     H_s, H_t = generator.full_forward(x_s, edge_s), generator.full_forward(x_t, edge_t)
    #     optimizer_d.zero_grad()
    #     D_s, D_t = discriminator(H_s), discriminator(H_t)
    #     y_s = torch.ones(D_s.size()[0], dtype=torch.long).cuda()
    #     y_t = torch.zeros(D_t.size()[0], dtype=torch.long).cuda()
    #     D = torch.cat((D_s, D_t), dim=0)
    #     y = torch.cat((y_s, y_t), dim=0)
    #     loss = cls_criterion(D, y)
    #     loss.backward()
    #     d_loss_all.append(loss)
    #     optimizer_d.step()

    optimizer_g.zero_grad()
    # max_d_loss = torch.max(torch.tensor(d_loss_all), dim=0)[0]
    calculate_gcn_loss(x_s, dataloader_s) + calculate_gcn_loss(x_t, dataloader_t)
    # loss = calculate_gcn_loss(x_s, dataloader_s) + calculate_gcn_loss(x_t, dataloader_t) + max_d_loss
    # loss.backward()
    optimizer_g.step()

    pred_t = generator.full_forward(x_t, edge_t)
    pred_t = pred_t.cpu()
    pred_t = pred_t.max(1)[1]
    acc = pred_t.eq(data_t.y).sum().item() / data_t.y.size(0)
    print(f"epoch: {epoch} acc: {acc:.3f} loss: {0:.3f}")
