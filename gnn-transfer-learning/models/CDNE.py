from torch import nn
from torch_geometric.nn import GCNConv
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch_geometric.datasets import CitationFull
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

nn.Conv1d()