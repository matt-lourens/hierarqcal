"""
Utility functions for the hierarqcal package. These are mostly plotting functions.

Usage:

.. code-block:: python

    hierq = Qinit(4) + Qcycle(2) + Qmask(pattern="inside") + Qinit(7) + Qpermute() + Qmask(filter="1000001")
    # Single motif
    fig, ax = plot_motif(hierq.tail)
    # Plot all motif
    figs = plot_motifs(hierq, all_motifs=True, figsize=(4,4))
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from .core import Qcycle, Qpermute, Qinit, Qmask, Qunmask


def plot_motif(
    primitive,
    cycle_color="#0096ff",
    mask_color="#ff7e79",
    permute_colour="#a9449d",
    init_colour="#92a9bd",
    node_large=0.12,  # radius of node
    node_small=0.08,  # radius of node
    edge_width=1.5,
    figsize=(4, 4),
    font_size=15,
    start_angle=0,  # given in 2 pi radians. 1/4 is 90 degrees, 1/2 is 180 degrees, 1 is 360 degrees
    **kwargs,
):
    """
    Plot a primitive in its directed graph representation

    Args:
        primitive (Qmotif): The primitive to plot, such as Qcycle, Qmask or Qpermute.
        cycle_color (str, optional): The colour of nodes for cycle motifs. Defaults to "#0096ff".
        mask_color (str, optional): The colour of nodes for masking motifs. Defaults to "#ff7e79".
        permute_colour (str, optional): The colour of nodes for permute motifs. Defaults to "#a9449d".
        init_colour (str, optional): The colour of nodes for init motifs. Defaults to "#92a9bd".
        node_large (int, optional): The size of the nodes for non masked qubits. Defaults to 400.
        node_small (int, optional): The size of the nodes for the masked qubits. Defaults to 150.
        edge_width (float, optional): The width of the edges. Defaults to 1.5.
        figsize (tuple, optional): The size of the figure. Defaults to (4, 4).
        **kwargs: Additional keyword arguments to pass to the networkx draw function.

    Returns:
        (tuple): A tuple containing:
            * fig (matplotlib.figure.Figure): The figure object.
            * ax (matplotlib.axes._subplots.AxesSubplot): The axes object.
    """
    n_qbits = len(primitive.Q)
    labels = primitive.Q
    if isinstance(primitive, Qcycle):
        node_radi = [node_large for q in primitive.Q]
        node_colour = cycle_color
    elif isinstance(primitive, Qmask):
        node_radi = [
            node_large if q in primitive.Q_avail else node_small for q in primitive.Q
        ]
        node_colour = mask_color
    elif isinstance(primitive, Qunmask):
        node_radi = [
            node_large for q in primitive.Q_avail
        ]        
        node_colour = mask_color
        labels = primitive.Q_avail
        n_qbits = len(primitive.Q_avail)
    elif isinstance(primitive, Qpermute):
        node_radi = [node_large for q in primitive.Q]
        node_colour = permute_colour
    elif isinstance(primitive, Qinit):
        node_radi = [node_large for q in primitive.Q]
        node_colour = init_colour
    else:
        raise NotImplementedError(f"No plot specified for {primitive} primitive")

    # Change order around a circle, this way you start at x=0 then move left around
    theta_0 = start_angle  # specify vector(1,0) as start
    theta_step = 1 / n_qbits
    pos = {
        label: np.array(
            [
                np.cos(2 * np.pi * (theta_0 + ind * theta_step)),
                np.sin(2 * np.pi * (theta_0 + ind * theta_step)),
            ]
        )
        for label, ind in zip(labels, range(n_qbits))
    }
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    eps = 0.01
    plt.xlim(-1 - node_large - eps, 1 + node_large + eps)
    plt.ylim(-1 - node_large - eps, 1 + node_large + eps)
    for (q, p), r in zip(pos.items(), node_radi):
        ax.add_patch(
            patches.Circle(
                p, radius=r, edgecolor="black", facecolor=node_colour, linewidth=1.5
            )
        )
        ax.text(
            p[0],
            p[1],
            str(q),
            color="black",
            fontsize=font_size,
            ha="center",
            va="center",
        )
    # Add edges
    for edge in primitive.E:
        # Get vector from soruce to edge of targets boundary
        if len(edge) == 1:
            # TODO implement loop
            raise NotImplementedError(
                f"Plotting {len(edge)}-ary edges are not implemented yet"
            )
        if len(edge) == 2:
            ket_0 = np.array([1, 0])
            ket_1 = np.array([0, 1])
            ket_s = np.array(pos[edge[0]])  # source vector
            ket_t = np.array(pos[edge[1]])  # target vector
            t_radius = node_radi[labels.index(edge[1])]
            ket_d = ket_t - ket_s  # source to target vector
            scale_factor = (np.sqrt(np.dot(ket_d, ket_d)) - (t_radius)) / np.sqrt(
                np.dot(ket_d, ket_d)
            )
            ket_t = ket_s + (scale_factor) * ket_d
            s_xy = np.array([np.dot(ket_0, ket_s), np.dot(ket_1, ket_s)])
            t_xy = np.array([np.dot(ket_0, ket_t), np.dot(ket_1, ket_t)])

            arrow = patches.FancyArrowPatch(
                s_xy,
                t_xy,
                mutation_scale=30,
                arrowstyle=patches.ArrowStyle(
                    "-|>", head_length= 2*t_radius, head_width= t_radius
                ),
                lw=edge_width,
                fc="black",
                zorder=-1,
            )
            ax.add_patch(arrow)
        else:
            # TODO hyper graph visual (rubber band)
            raise NotImplementedError(
                f"Plotting {len(edge)}-ary edges are not implemented yet"
            )
    ax.set_aspect("equal", "box")
    # remove the x and y axis
    ax.axis("off")

    # remove the box around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # make the plot tight
    fig.tight_layout()

    return fig, ax


# def plot_motifs(
#     hierq,
#     all_motifs=False,
#     cycle_color="#0096ff",
#     mask_color="#ff7e79",
#     permute_colour="#a9449d",
#     init_colour="#92a9bd",
#     **kwargs,
# ):
#     """
#     Plot all motifs in a Hierarchical object TODO update for new version

#     Args:
#         hierq (Hierarchical): The Hierarchical object to plot.
#         all_motifs (bool, optional): Whether to plot all motifs in the Hierarchical object or just the operational ones. Defaults to False (just operations)
#         cycle_color (str, optional): The colour of nodes for cycle motifs. Defaults to "#0096ff".
#         mask_color (str, optional): The colour of nodes for masking motifs. Defaults to "#ff7e79".
#         permute_colour (str, optional): The colour of nodes for permute motifs. Defaults to "#a9449d".
#         init_colour (str, optional): The colour of nodes for init motifs. Defaults to "#92a9bd".
#         **kwargs: Additional keyword arguments to pass to the networkx draw function.

#     Returns:
#         figs (list): A list of matplotlib figure objects.
#     """
#     figs = []
#     if all_motifs:
#         motif = hierq.tail
#         while motif is not None:
#             fig, ax = plot_motif(
#                 motif, cycle_color, mask_color, permute_colour, init_colour, **kwargs
#             )
#             motif = motif.next
#             # oPlot.add_plot(ax)
#             figs.append(fig)
#             plt.close()
#     else:
#         for motif in hierq:
#             fig, ax = plot_motif(
#                 motif, cycle_color, mask_color, permute_colour, init_colour, **kwargs
#             )
#             # oPlot.add_plot(ax)
#             figs.append(fig)
#             plt.close()
#     return figs