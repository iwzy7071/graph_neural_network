from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
from tqdm import tqdm
import shutil
import os.path as osp
import os

np.set_printoptions(precision=2)


def visualize(model, data, args):
    model.eval()
    pca = PCA(n_components=3)
    node_embeddings = pca.fit_transform(data.x.numpy())
    node_labels = data.y.numpy()
    original_graph = to_networkx(data=data, to_undirected=True)
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'bad_case',
                         '{}_{}'.format(args.model, args.dataset))
    try:
        shutil.rmtree(save_path)
    except Exception:
        pass
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        pred = model(data).max(1)[1]
        pred = pred.eq(data.y).cpu().numpy().tolist()

    for node_idx, label in tqdm(enumerate(pred)):
        if label is False:
            current_graph = nx.Graph()
            current_graph.add_node(node_idx, color=node_labels[node_idx],
                                   label='ERROR' + str(node_embeddings[node_idx]))
            for node_idx_1 in original_graph.adj[node_idx]:
                current_graph.add_node(node_idx_1, color=node_labels[node_idx_1],
                                       label=str(node_embeddings[node_idx_1]))
                current_graph.add_edge(node_idx, node_idx_1)
                for node_idx_2 in original_graph.adj[node_idx_1]:
                    if node_idx_2 == node_idx:
                        continue
                    current_graph.add_node(node_idx_2, color=node_labels[node_idx_2],
                                           label=str(node_embeddings[node_idx_2]))
                    current_graph.add_edge(node_idx_1, node_idx_2)
            labels = nx.get_node_attributes(current_graph, 'label')
            colors = list(nx.get_node_attributes(current_graph, 'color').values())
            nx.draw(current_graph, node_size=50, width=1, labels=labels, node_color=colors, font_size=6, alpha=0.5,
                    with_labels=True)
            plt.savefig(osp.join(save_path, '{}.png'.format(node_idx)))
            plt.clf()
            current_graph.clear()
