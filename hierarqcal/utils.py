"""
Utility functions for the hierarqcal package. These are mostly plotting functions.

Usage:

.. code-block:: python

    qcnn = Qfree(4) + Qcycle(2) + Qmask(filter="inside") + Qfree(7) + Qpermute() + Qmask(filter="1000001")
    # Single motif
    fig, ax = plot_motif(qcnn.tail)
    # Full QCNN
    figs = plot_motifs(m, all_motifs=True, figsize=(4,4))
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .core import Qcycle, Qmask, Qpermute, Qfree


def plot_motif(
    motif,
    conv_color="#0096ff",
    pool_color="#ff7e79",
    dense_colour="#a9449d",
    qfree_colour="#92a9bd",
    node_large=400,
    node_small=150,
    edge_width=1.5,
    figsize=(4, 4),
    font_size=15,
    **kwargs,
):
    """
    Plot a motif in its directed graph representation

    Args:
        motif (Qmotif): The motif to plot, such as Qcycle, Qmask or Qpermute.
        conv_color (str, optional): The colour of nodes for convolution motifs. Defaults to "#0096ff".
        pool_color (str, optional): The colour of nodes for pooling motifs. Defaults to "#ff7e79".
        dense_colour (str, optional): The colour of nodes for dense motifs. Defaults to "#a9449d".
        qfree_colour (str, optional): The colour of nodes for free motifs. Defaults to "#92a9bd".
        node_large (int, optional): The size of the nodes for non pooled qubits. Defaults to 400.
        node_small (int, optional): The size of the nodes for the pooled qubits. Defaults to 150.
        edge_width (float, optional): The width of the edges. Defaults to 1.5.
        figsize (tuple, optional): The size of the figure. Defaults to (4, 4).
        **kwargs: Additional keyword arguments to pass to the networkx draw function.

    Returns:
        (tuple): A tuple containing:
            * fig (matplotlib.figure.Figure): The figure object.
            * ax (matplotlib.axes._subplots.AxesSubplot): The axes object.
    """
    n_qbits = len(motif.Q)
    # Change order around a circle, this way you start at x=0 then move left around
    theta_0 = 2 / 8  # specify vector(0,1) as start
    theta_step = 1 / n_qbits
    pos = {
        label: np.array(
            [
                np.cos(2 * np.pi * (theta_0 + ind * theta_step)),
                np.sin(2 * np.pi * (theta_0 + ind * theta_step)),
            ]
        )
        for label, ind in zip(motif.Q, range(n_qbits))
    }
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(motif.Q)
    nx_graph.add_edges_from(motif.E)
    if isinstance(motif, Qcycle):
        node_sizes = [node_large for q in motif.Q]
        node_colour = conv_color
    elif isinstance(motif, Qmask):
        node_sizes = [
            node_small if (q in [i for (i, j) in motif.E]) else node_large
            for q in motif.Q
        ]
        node_colour = pool_color
    elif isinstance(motif, Qpermute):
        node_sizes = [node_large for q in motif.Q]
        node_colour = dense_colour
    elif isinstance(motif, Qfree):
        node_sizes = [node_large for q in motif.Q]
        node_colour = qfree_colour
    else:
        raise NotImplementedError(f"No plot specified for {motif} motif")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Draw nodes and edges without labels
    nx.draw(
        nx_graph,
        pos,
        with_labels=False,
        node_size=node_sizes,
        edge_color="#000000",
        edgecolors="#000000",
        node_color=node_colour,
        width=edge_width,
        **kwargs,
    )

    # Draw node labels with custom font size
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels={n: n for n in nx_graph.nodes()},
        font_size=font_size,
    )
    return fig, ax


def plot_motifs(
    qcnn,
    all_motifs=False,
    conv_color="#0096ff",
    pool_color="#ff7e79",
    dense_colour="#a9449d",
    qfree_colour="#92a9bd",
    **kwargs,
):
    """
    Plot all motifs in a Qcnn object

    Args:
        qcnn (Qcnn): The Qcnn object to plot.
        all_motifs (bool, optional): Whether to plot all motifs in the Qcnn object or just the operational ones. Defaults to False (just operations)
        conv_color (str, optional): The colour of nodes for convolution motifs. Defaults to "#0096ff".
        pool_color (str, optional): The colour of nodes for pooling motifs. Defaults to "#ff7e79".
        dense_colour (str, optional): The colour of nodes for dense motifs. Defaults to "#a9449d".
        qfree_colour (str, optional): The colour of nodes for free motifs. Defaults to "#92a9bd".
        **kwargs: Additional keyword arguments to pass to the networkx draw function.

    Returns:
        figs (list): A list of matplotlib figure objects.
    """
    oPlot = FlowLayout()
    figs = []
    if all_motifs:
        motif = qcnn.tail
        while motif is not None:
            fig, ax = plot_motif(
                motif, conv_color, pool_color, dense_colour, qfree_colour, **kwargs
            )
            motif = motif.next
            oPlot.add_plot(ax)
            figs.append(fig)
            plt.close()
    else:
        for motif in qcnn:
            fig, ax = plot_motif(
                motif, conv_color, pool_color, dense_colour, qfree_colour, **kwargs
            )
            oPlot.add_plot(ax)
            figs.append(fig)
            plt.close()
    oPlot.PassHtmlToCell()
    return figs
    


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import io
import base64


# See https://stackoverflow.com/questions/21754976/ipython-notebook-arrange-plots-horizontally
class FlowLayout(object):
    """A class / object to display plots in a horizontal / flow layout below a cell"""

    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml = """
        <style>
        .floating-box {
        display: inline-block;
        margin: 10px;
        border: 2px solid #000000;  
        }
        </style>
        """

    def add_plot(self, oAxes):
        """Saves a PNG representation of a Matplotlib Axes object"""
        Bio = io.BytesIO()  # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.canvas.print_png(Bio)  # make a png of the plot in the buffer

        # encode the bytes as string using base 64
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml += (
            '<div class="floating-box">'
            + '<img src="data:image/png;base64,{}\n">'.format(sB64Img)
            + "</div>"
        )

    def PassHtmlToCell(self):
        """Final step - display the accumulated HTML"""
        display(HTML(self.sHtml))


# === Testing ===
# for stride in [1,3,5,7]:
#     m = Qfree(8) + Qcycle(stride)
#     fig, ax = plot_motif(m.head, font_size=15, node_large=700, edge_width=1.8)
#     fig.savefig(f"/home/matt/Downloads/stride_{stride}.svg", format="svg")
# print("Apastionat")
# # %%
# import hypernetx as hnx
# import itertools as it
# # 1,3,0
# # 1,3,2
# # 9q 3,1,0

# stride=3
# step=1
# offset=0
# boundary="open"
# m = Qfree(9) + Qcycle(stride, step, offset, qpu=3, boundary="open")
# # plot_motif(m.head)
# motif = m.head
# n_qbits = len(motif.Q)
# # Change order around a circle, this way you start at x=0 then move left around
# theta_0 = 1 / 4  # specify vector(0,1) as start
# theta_step = -1 / n_qbits
# pos = {
#     label: np.array(
#         [
#             np.cos(2 * np.pi * (theta_0 + ind * theta_step)),
#             np.sin(2 * np.pi * (theta_0 + ind * theta_step)),
#         ]
#     )
#     for label, ind in zip(motif.Q, range(n_qbits))
# }
# H = hnx.Hypergraph(m.head.E)
# H._add_nodes_from(motif.Q)
# fig, ax = plt.subplots()
# hnx.drawing.draw(
#     H,
#     pos=pos,
#     with_edge_labels=False,
#     node_radius=3.5,
#     node_labels_kwargs={"ha": "center", "fontsize": 15},
#     nodes_kwargs={"facecolors": "#0096ff", "edgecolors": "black", "linewidths": 1.4},
#     edges_kwargs={"edgecolors":  plt.cm.tab10(7), "linewidths": 2, "dr":0.03},
# )
# # Set title of plot
# ax.set_title(f"Stride: {stride}, Step: {step}, Offset: {offset}")
# # Save fig svg
# fig.savefig(f"/home/matt/Downloads/stride_{stride}_step_{step}_offset_{offset}.svg", format="svg")

# # # %%
# # # "edgecolors": "#FFA500",

