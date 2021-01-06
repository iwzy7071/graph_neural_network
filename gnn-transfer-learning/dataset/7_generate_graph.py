import json
import networkx as nx
from torch_geometric.utils import from_networkx
import torch


# Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
def generate_graph_embedding(name):
    print(name)
    G = nx.Graph()
    paper_title_embedding = json.load(open(f"./transfer/{name}.json"))
    paper_list = json.load(open(f"./transfer/{name}.txt"))
    paper_index_list = [paper['id'] for paper in paper_list]
    for paper, title_embedding in zip(paper_list, paper_title_embedding):
        index, references, label = paper['id'], paper['reference'], paper['label']
        G.add_node(index, x=title_embedding, y=label)
        for ref in references:
            if ref in paper_index_list:
                G.add_edge(index, ref)
    G = from_networkx(G)
    torch.save(G, f"./transfer/{name}.pt")


generate_graph_embedding("paper_before_2002_clean")
generate_graph_embedding("paper_betweeen_02and05_clean")
generate_graph_embedding("paper_betweeen_05and08_clean")
generate_graph_embedding("paper_betweeen_08and11_clean")
generate_graph_embedding("paper_betweeen_11and14_clean")
generate_graph_embedding("paper_betweeen_14and17_clean")
generate_graph_embedding("paper_betweeen_17and20_clean")
