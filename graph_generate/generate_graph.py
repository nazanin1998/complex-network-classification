import sys
from random import randint, random

import networkx as nx


class GraphGenerator:
    def __init__(self, graph_count):
        print("GENERATE GRAPHS ==> " + str(graph_count))
        self.graph_count = graph_count
        self.graph_matrix = []
        self.labels = []

    def generate_graphs(self):
        for i in range(self.graph_count):
            graph_type = randint(1, 4)
            if graph_type == 1:
                self.graph_matrix.append(_generate_random_barabasi_albert_graph())
                self.labels.append('BA')
            elif graph_type == 2:
                self.graph_matrix.append(_generate_random_erdos_renyi_graph())
                self.labels.append('ER')
            elif graph_type == 3:
                self.graph_matrix.append(_generate_random_watts_strogatz_graph())
                self.labels.append('WS')
            elif graph_type == 4:
                self.graph_matrix.append(_generate_random_scale_free_graph())
                self.labels.append('SF')


def _generate_random_barabasi_albert_graph():
    node_count = randint(100, 200)
    edge_count = randint(20, 99)
    return nx.to_pandas_edgelist(nx.barabasi_albert_graph(node_count, edge_count, seed=None))


def _generate_random_erdos_renyi_graph():
    node_count = randint(100, 200)
    edge_probability = random()
    return nx.to_pandas_edgelist(nx.erdos_renyi_graph(node_count, edge_probability))


def _generate_random_watts_strogatz_graph():
    node_count = randint(100, 200)
    neighbour_count = randint(2, 99)
    edge_probability = random()
    return nx.to_pandas_edgelist(nx.watts_strogatz_graph(node_count, neighbour_count, edge_probability))


def _generate_random_scale_free_graph():
    node_count = randint(100, 200)
    return nx.to_pandas_edgelist(nx.scale_free_graph(node_count))
