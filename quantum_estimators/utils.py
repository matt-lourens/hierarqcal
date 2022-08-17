import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(
    graph, conv_color="#0096ff", pool_color="#ff7e79", figsize=(7, 7), **kwargs
):
    """plot single graph

    Args:
        graph (DiGraph): _description_
        conv_color (str, optional): _description_. Defaults to "#0096ff".
        pool_color (str, optional): _description_. Defaults to "#ff7e79".
    """
    n_qbits = len(graph.Q)
    # Change order around a circle, this way you start at x=0 then move left around
    theta_0 = 2 / 8 # specify vector(0,1) as start
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
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(graph.Q)
    nx_graph.add_edges_from(graph.E)
    if graph.type == "convolution":
        node_sizes = [1000 for q in graph.Q]
        node_colour = conv_color
    elif graph.type == "pooling":
        node_sizes = [
            200 if (q in [i for (i, j) in graph.E]) else 1000 for q in graph.Q
        ]
        node_colour = pool_color
    else:
        raise NotImplementedError(f"No plot specified for {graph.type} graph type")
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        edge_color="#000000",
        edgecolors="#000000",
        node_color=node_colour,
        width=1.5,
        **kwargs,
    )
# %%
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
