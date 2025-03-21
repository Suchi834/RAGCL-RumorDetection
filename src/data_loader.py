import networkx as nx
import dgl

def load_graph(dataset_path):
    G = nx.read_gpickle(dataset_path)  # Load pre-processed graph
    dgl_graph = dgl.from_networkx(G)
    return dgl_graph

