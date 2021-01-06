from torch import nn
from torch_geometric.nn import GCNConv
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch_geometric.datasets import CitationFull
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Generator(nn.Module):
    def __init__(self, dim_input, dim_target):
        super(Generator, self).__init__()
        # FIXME: GCNCONV HAS BEEN RECTIFIED
        self.gcns = nn.ModuleList([GCNConv(dim_input, 1000), GCNConv(1000, 100), GCNConv(100, 16)])
        self.linear = nn.Sequential(nn.Linear(16, dim_target), nn.Sigmoid())
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data_s, data_t):
        x_s, edge_index_s = data_s.x.cuda(), data_s.edge_index.cuda()
        x_t, edge_index_t = data_t.x.cuda(), data_t.edge_index.cuda()
        for gcn_layer in self.gcns:
            x_s = gcn_layer(x_s, edge_index_s)
            x_t = gcn_layer(x_t, edge_index_t)
            x_s, x_t = self.dropout(x_s), self.dropout(x_t)

        pred_s = self.linear(x_s)
        pred_t = self.linear(x_s)
        return x_s, x_t, pred_s, pred_t


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features_to_real_number = nn.Linear(16, 1)

    def forward(self, x):
        return self.features_to_real_number(x)


def calculate_gradient_penalty(source_data, target_data):
    batch_size = min(source_data.size()[0], target_data.size()[0])
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(source_data)
    alpha = alpha.cuda()
    interpolated = alpha * source_data.data + (1 - alpha) * target_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()
    prob_interpolated = discriminator(interpolated)
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return 10 * ((gradients_norm - 1) ** 2).mean()


# SET UP TRAINING ENVIRONMENT
dataset_s = CitationFull(root="../dataset", name="dblp")
dataset_t = CitationFull(root='../dataset', name='dblp')
cls_criterion = nn.CrossEntropyLoss()
generator, discriminator = Generator(dataset_s.num_features, dataset_s.num_classes), Discriminator()
generator, discriminator = generator.cuda(), discriminator.cuda()
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
    # DOMAIN CRITIC
    for _ in range(10):
        H_s, H_t, _, _ = generator(dataset_s.data, dataset_t.data)
        optimizer_d.zero_grad()
        D_s, D_t = discriminator(H_s), discriminator(H_t)
        d_loss = D_s.mean() - D_t.mean() + calculate_gradient_penalty(H_s, H_t)
        d_loss_all.append(d_loss)
        d_loss.backward()
        optimizer_d.step()

    optimizer_g.zero_grad()
    max_d_loss = torch.max(torch.tensor(d_loss_all), dim=0)[0]
    _, _, pred_s, pred_t = generator(dataset_s.data, dataset_t.data)
    print(pred_s.size(), dataset_s.data.y.size())
    exit()
    loss = cls_criterion(pred_s, dataset_s.data.y.cuda()) + max_d_loss
    loss.backward()
    optimizer_g.step()

    pred_t = pred_t.cpu()
    pred_t = pred_t.max(1)[1]
    acc = pred_t.eq(dataset_t.data.y).sum().item() / dataset_t.data.y.size(0)
    print(f"epoch: {epoch} acc: {acc:.3f} loss: {loss.item():.3f}")
