import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import shutil
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(dataset, model, rotate):
    path = '/home/wzy/graph_neural_network/log'
    try:
        shutil.rmtree(osp.join(path, '{}_{}_{}'.format(dataset, model, rotate)))
    except Exception:
        pass
    writer = SummaryWriter(log_dir=osp.join(path, '{}_{}_{}'.format(dataset, model, rotate)))
    return writer


def get_rotate_dataset(dataset, rotate):
    if rotate == 'random':
        dataset.data.y = torch.randint_like(dataset.data.y, low=0, high=2)
    elif rotate == 'permutation':
        rand_index = torch.randperm(dataset.data.y.size(0))
        dataset.data.y = dataset.data.y[rand_index]

    return dataset


writer = get_summary_writer('ogbg-molhiv', 'DeeperGCN', 'permutation')
evaluator = Evaluator('ogbg-molhiv')


def train(model, device, loader, optimizer, task_type, epoch):
    model.train()
    y_true, y_pred = [], []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        is_labeled = batch.y == batch.y
        if "classification" in task_type:
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

        loss.backward()
        optimizer.step()
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        if step == len(loader) - 1 or (step + 1) % 50 == 0:
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            for key, value in evaluator.eval(input_dict).items():
                writer.add_scalar(f'{key}', value, epoch * len(loader) + step)
            y_true, y_pred = [], []


def main():
    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygGraphPropPredDataset(name=args.dataset, root='/home/wzy/graph_neural_network/dataset')
    dataset = get_rotate_dataset(dataset, 'permutation')
    args.num_tasks = dataset.num_tasks

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    model = DeeperGCN(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(0, 50):
        train(model, device, train_loader, optimizer, dataset.task_type, epoch)


if __name__ == "__main__":
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()
    main()
