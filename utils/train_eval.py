import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os.path as osp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(dataset, model, args, permute_masks=None, writer=None):
    for _ in range(args.runs):
        data = dataset[0]
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        for epoch in tqdm(range(1, args.epochs + 1)):
            train(model, optimizer, data)
            evaluate(model, data, epoch, writer)

        save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'model_save')
        torch.save(model.state_dict(), osp.join(save_path, '{}_{}.pt'.format(args.model, args.dataset)))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data, epoch, writer):
    model.eval()

    with torch.no_grad():
        output = model(data)
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            loss = F.nll_loss(output[mask], data.y[mask]).item()
            pred = output[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            writer.add_scalar('{}_loss'.format(key), loss, epoch)
            writer.add_scalar('{}_acc'.format(key), acc, epoch)
