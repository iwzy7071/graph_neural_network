from torch.optim import Adam
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models.GAT import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset = Planetoid(root='/home/wzy/graph_neural_network/gnn-transfer-learning/dataset', name='Cora')


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(4096, 2048)
        self.lin2 = Linear(2048, 1024)
        self.lin3 = Linear(1024, 6)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


# data = dataset.data
data = torch.load("./dataset/transfer/paper_before_2002_clean.pt")
x, edge_index, y = data.x, data.edge_index, data.y
print(edge_index)
exit()
num_nodes = data.x.size()[0]
mask = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[mask[:-len(mask) // 5]] = 1
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[mask[-len(mask) // 5:]] = 1

model = GAT(4096, 6)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model, x, edge_index, y = model.cuda(), x.cuda(), edge_index.cuda(), y.cuda()
train_mask, test_mask = train_mask.cuda(), test_mask.cuda()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    train_loss = criterion(out[train_mask], y[train_mask])
    train_loss.backward()
    optimizer.step()
    return train_loss


def test():
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[test_mask] == y[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    return test_acc


if __name__ == '__main__':
    for epoch in range(1, 201):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d},Loss: {loss:.4f}, acc: {acc:.4f}')
