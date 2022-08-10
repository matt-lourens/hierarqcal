import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_qcnn_graphs(graphs, conv_color="#0096ff", pool_color="#ff7e79", **kwargs):
    # Get the first graph to derive some relevant information from the structure
    Qc_0 = list(graphs.values())[0][0][0]
    # The number of nodes of first graph
    n_qbits = len(Qc_0)
    # Change order around a circle, this way you start at x=0 then move left around
    theta_0 = 2 / n_qbits
    theta_step = 1 / n_qbits
    pos = {
        ind
        + 1: np.array(
            [
                np.cos(2 * np.pi * (theta_0 + ind * theta_step)),
                np.sin(2 * np.pi * (theta_0 + ind * theta_step)),
            ]
        )
        for ind in range(n_qbits)
    }
    figs = []
    for layer, ((Qc_l, Ec_l), (Qp_l, Ep_l)) in graphs.items():

        nx_c_graph = nx.DiGraph()
        nx_p_graph = nx.DiGraph()

        nx_c_graph.add_nodes_from(Qc_0)
        nx_c_graph.add_edges_from(Ec_l)
        node_c_sizes = [1000 if (q in Qc_l) else 200 for q in Qc_0]

        nx_p_graph.add_nodes_from(Qc_0)
        nx_p_graph.add_edges_from(Ep_l)
        node_p_sizes = [1000 if (q in [j for (i, j) in Ep_l]) else 200 for q in Qc_0]

        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw(
            nx_c_graph,
            pos,
            with_labels=True,
            node_size=node_c_sizes,
            edge_color="#000000",
            edgecolors="#000000",
            node_color=conv_color,
            width=1.5,
            **kwargs,
        )
        figs = figs + [fig]
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw(
            nx_p_graph,
            pos,
            with_labels=True,
            node_size=node_p_sizes,
            edge_color="#000000",
            edgecolors="#000000",
            node_color=pool_color,
            width=1.5,
            **kwargs,
        )
        figs = figs + [fig]
    return figs