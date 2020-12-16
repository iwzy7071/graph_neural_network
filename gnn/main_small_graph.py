from utils import get_summary_writer, get_graph_env
import torch
import argparse
from tqdm import tqdm
from ogb.graphproppred import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gin')
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=300)
parser.add_argument('--input_features', type=int, default=32)
parser.add_argument('--drop_ratio', type=float, default=0.5)
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--rotate', type=str, default='permutation')
args = parser.parse_args()

train_loader, val_loader, test_loader, net = get_graph_env(args)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
evaluator = Evaluator('ogbg-ppa')
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
writer = get_summary_writer(args.dataset, args.model, args.rotate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)


def train(epoch):
    net.train()
    y_true, y_pred = [], []
    for index, batch in enumerate(tqdm(train_loader, desc=f'{args.model}_{args.dataset}_{args.rotate}')):
        batch = batch.to(device)
        pred = net(batch)

        optimizer.zero_grad()
        is_labeled = batch.y == batch.y

        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        if index == len(train_loader) - 1 or (index + 1) % 50 == 0:
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            for key, value in evaluator.eval(input_dict).items():
                writer.add_scalar(f'{key}', value, epoch * len(train_loader) + index)
            y_true, y_pred = [], []


def eval(epoch, dataloader, mode):
    net.eval()
    y_true, y_pred = [], []
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = net(batch.x, batch.edge_index, batch.batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    print(output)


for epoch in range(args.epochs):
    train(epoch)
