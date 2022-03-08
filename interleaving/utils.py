import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import pairwise_distances


def create_graph(df_sushi, plot=True, score_par=1, approximate=None):

    score_par = 1 / np.median(pairwise_distances(df_sushi))
    metric = lambda x: np.exp(-score_par * pairwise_distances(x, metric="euclidean"))
    similarities = metric(df_sushi.values)

    if not approximate is None:
        prova = np.triu(similarities)
        prova = prova[prova != 0]
        true = prova[prova != 1]
        th = np.quantile(true, approximate)
    else:
        th = 0

    G = nx.Graph()
    for i, sushi in enumerate(df_sushi.index):
        G.add_node(i, name=i)

    for i in range(len(df_sushi)):
        for j in range(len(df_sushi)):
            if i == j or similarities[i, j] < th:
                continue
            attr = {"weight": similarities[i, j]}
            # attr = {"weight" : 1-similarities[i,j]}
            G.add_edge(i, j, **attr)

    if not plot:
        return G, similarities

    print(similarities.round(2))

    return G, similarities
