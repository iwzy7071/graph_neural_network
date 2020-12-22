import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from utils import get_summary_writer
import os.path as osp


def get_split_Dataloader(dataset):
    split_index = len(dataset) // 5
    train_dataset, test_dataset, val_dataset = dataset[:split_index * 3], \
                                               dataset[split_index * 3:split_index * 4], dataset[split_index * 4:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, val_loader


def run(model, dataset, args):
    train_loader, test_loader, val_loader = get_split_Dataloader(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    writer = get_summary_writer(args.dataset, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train(epoch, model, train_loader, device, criterion, optimizer, writer)
        model.eval()
        test_eval(epoch, model, test_loader, device, criterion, writer, 'test')
        test_eval(epoch, model, val_loader, device, criterion, writer, 'val')

    save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model_save')
    torch.save(model.state_dict(), osp.join(save_path, '{}_{}.pt'.format(args.model, args.dataset)))


def train(epoch, model, dataloader, device, criterion, optimizer, writer):
    total_acc, total_loss = [], []
    for index, data in enumerate(dataloader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = output.max(1)[1]
        total_acc.append(pred.eq(data.y).sum().item() / data.y.size(0))
    writer.add_scalar('train_loss', sum(total_loss) / len(total_loss), epoch)
    writer.add_scalar('train_acc', sum(total_acc) / len(total_acc), epoch)


def test_eval(epoch, model, dataloader, device, criterion, writer, mode):
    total_acc, total_loss = [], []
    for index, data in enumerate(dataloader):
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        total_loss.append(loss.item())

        pred = output.max(1)[1]
        total_acc.append(pred.eq(data.y).sum().item() / data.y.size(0))
    writer.add_scalar('{}_loss'.format(mode), sum(total_loss) / len(total_loss), epoch)
    writer.add_scalar('{}_acc'.format(mode), sum(total_acc) / len(total_acc), epoch)
