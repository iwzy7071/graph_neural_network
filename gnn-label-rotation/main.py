from utils.utils import get_summary_writer, get_rotate_dataset
from utils.get_graph_env import get_graph_env
from torch_geometric.data import DataLoader
import torch
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GraphSAGE')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--rotate', type=str, default='random')
args = parser.parse_args()
cls_criterion = torch.nn.CrossEntropyLoss()


def train(epoch, dataloader, optimizer, writer):
    net.train()
    total_acc = []
    for index, batch in enumerate(dataloader):
        batch = batch.to('cuda')
        pred = net(batch)
        optimizer.zero_grad()
        loss = cls_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        pred = pred.max(1)[1]
        total_acc.append(pred.eq(batch.y).sum().item() / batch.y.size(0))
    writer.add_scalar('train_acc', sum(total_acc) / len(total_acc), epoch)


if __name__ == '__main__':
    for index in range(10):
        dataset, net = get_graph_env(args)
        dataset = dataset.shuffle()[:int(len(dataset) * 0.8)]
        dataset = get_rotate_dataset(dataset, rotate=args.rotate)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=750)
        net.to('cuda')
        writer = get_summary_writer(args.dataset, args.model, args.rotate, index)
        for epoch in tqdm(range(args.epochs), desc=f'{args.model}_{args.dataset}_{args.rotate}_{index}'):
            train(epoch, dataloader, optimizer, writer)
            scheduler.step()
