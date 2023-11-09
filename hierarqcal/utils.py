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
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib import cm
from matplotlib.path import Path
from hierarqcal import Qcycle, Qpermute, Qinit, Qmask, Qunmask, Qpivot, Qmotif


def plot_motif(
    primitive,
    cycle_color="#0096ff",
    mask_color="#ff7e79",
    permute_colour="#a9449d",
    init_colour="#92a9bd",
    pivot_colour="#0096ff",
    node_large=0.12,  # radius of node
    node_small=0.08,  # radius of node
    edge_width=1.5,
    figsize=(4, 4),
    font_size=15,
    color_dict={"start": "green", "during": "orange", "end": "red"},
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
        node_radi = [node_large for q in primitive.Q_avail]
        node_colour = mask_color
        labels = primitive.Q_avail
        n_qbits = len(primitive.Q_avail)
    elif isinstance(primitive, Qpermute):
        node_radi = [node_large for q in primitive.Q]
        node_colour = permute_colour
    elif isinstance(primitive, Qinit):
        node_radi = [node_large for q in primitive.Q]
        node_colour = init_colour
    elif isinstance(primitive, Qpivot):
        node_radi = [node_large for q in primitive.Q]
        node_colour = pivot_colour
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
    eps = 0.2
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
                    "-|>", head_length=2 * t_radius, head_width=t_radius
                ),
                lw=edge_width,
                fc="black",
                zorder=-1,
            )
            ax.add_patch(arrow)
        else:
            end_points = []
            for i in range(len(edge)):
                ket_s = np.array(pos[edge[i]])  # source vector
                ket_t = np.array(pos[edge[(i + 1) % len(edge)]])  # target vector

                par_vec = ket_t - ket_s

                ## Cyclic order dependency for clearer plotting
                ## clockwise direction of nodes needs plotting to start from outer side
                orth_vec = np.array([par_vec[1], -par_vec[0]]) * (
                    1 / np.linalg.norm(par_vec)
                )

                ## anticlockwise direction of nodes needs plotting to start from inner side
                if primitive.arity < primitive.step:
                    orth_vec = -orth_vec

                t_radius = node_radi[labels.index(edge[i])]

                pad_ratio = 1.5
                ket_s = ket_s + (t_radius * pad_ratio) * orth_vec
                ket_t = ket_t + (t_radius * pad_ratio) * orth_vec

                path_trace = Path([ket_s, ket_t], [Path.MOVETO, Path.LINETO])
                pathpatch_trace = PathPatch(
                    path_trace, lw=edge_width, color="black", zorder=-1
                )

                ax.add_patch(pathpatch_trace)
                end_points.append(
                    (ket_s, ket_t, pos[edge[(i + 1) % len(edge)]], t_radius * pad_ratio)
                )

            for i in range(len(end_points)):
                if i == 0:
                    color = color_dict["start"]
                elif i == len(edge) - 1:
                    color = color_dict["end"]
                else:
                    color = color_dict["during"]
                center = end_points[i][2]
                start = end_points[i][1]
                end = end_points[(i + 1) % len(end_points)][0]

                if primitive.arity < primitive.step:
                    start, end = end, start

                start_vec = start - center
                end_vec = end - center
                ref_vec = np.array([1, 0])

                norm_prod = np.linalg.norm(start_vec) * np.linalg.norm(end_vec)
                angle = np.arccos(np.dot(start_vec, end_vec) / norm_prod)

                if start_vec[1] < 0:
                    norm_prod = np.linalg.norm(start_vec) * np.linalg.norm(ref_vec)
                    angle_ref = -np.arccos(np.dot(start_vec, ref_vec) / norm_prod)
                else:
                    norm_prod = np.linalg.norm(start_vec) * np.linalg.norm(ref_vec)
                    angle_ref = np.arccos(np.dot(start_vec, ref_vec) / norm_prod)

                end_point_check = np.array(
                    [
                        center[0] + t_radius * pad_ratio * np.cos(angle_ref + angle),
                        center[1] + t_radius * pad_ratio * np.sin(angle_ref + angle),
                    ]
                )
                if np.linalg.norm(end_point_check - end) > 1e-3:
                    angle = 2 * np.pi - angle
                arc_angles = np.linspace(angle_ref, angle_ref + angle, 20)
                arc_xs = center[0] + t_radius * pad_ratio * np.cos(arc_angles)
                arc_ys = center[1] + t_radius * pad_ratio * np.sin(arc_angles)

                plt.plot(arc_xs, arc_ys, lw=edge_width, color=color)

    ax.set_aspect("equal", "box")
    # remove the x and y axis
    ax.axis("off")

    # remove the box around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # make the plot tight
    fig.tight_layout()

    return fig, ax


def plot_circuit(
    hierq,
    plot_width=20,
    cycle_color="#0096ff",
    mask_color="#ff7e79",
    permute_colour="#a9449d",
    init_colour="#92a9bd",
    dx = 0.5,
    big_r = 0.5,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, plot_width)
    ax.set_ylim(-len(hierq.tail.Q) - 1, 1)
    ax.set_aspect("equal")
    layer = hierq.tail
    x = 1
    # dx = 0.5
    small_r = 0.2
    ddx = 0
    while layer is not None:
        if isinstance(layer, Qcycle):
            node_colour = cycle_color
        elif isinstance(layer, Qmask):
            node_colour = mask_color
        elif isinstance(layer, Qunmask):
            node_colour = mask_color
        elif isinstance(layer, Qpermute):
            node_colour = cycle_color
        elif isinstance(layer, Qpivot):
            node_colour = cycle_color
        elif isinstance(layer, Qinit):
            node_colour = init_colour
        elif isinstance(layer, Qmotif):
            node_colour = cycle_color
        if isinstance(layer, Qinit):
            # plot ket tensors
            for i, label in enumerate(layer.Q):
                # Give border
                circle = plt.Circle(
                    (x, -i), big_r, facecolor=node_colour, edgecolor="black", linewidth=1
                )
                ax.add_artist(circle)
                ax.text(x, -i, label, ha="center", va="center")
                ax.hlines(-i, x, plot_width, color="gray", zorder=-2)
            ddx += dx
        elif isinstance(layer, Qmask) and len(layer.E) == 0:
            for i, label in enumerate([q for q in layer.Q if q not in layer.Q_avail]):
                ind = hierq.tail.Q.index(label)
                circle1 = plt.Circle(
                    (x + ddx, -ind), small_r, fill=True, color=node_colour
                )
                ax.add_artist(circle1)
        elif isinstance(layer, Qunmask):
            for i, label in enumerate([q for q in layer.Q_avail if q not in layer.Q]):
                ind = hierq.tail.Q.index(label)
                circle1 = plt.Circle((x + ddx, -ind), small_r, fill=True, color="green")
                ax.add_artist(circle1)
        else:
            # plot ket tensors
            for e_ind, e in enumerate(layer.E):
                q_prev = e[0]
                q_prev_ind = hierq.tail.Q.index(q_prev)
                i_order = 0
                color = get_color(i_order, len(e))
                circle1 = plt.Circle(
                    (x + ddx, -q_prev_ind), small_r, fill=True, color=color
                )
                ax.add_artist(circle1)
                if layer.edge_mapping[e_ind].name is not None:
                    # get rotation from kwargs
                    rotation = kwargs.get("rotation", 30)
                    ax.text(
                        x + ddx,
                        -q_prev_ind+.15,
                        layer.edge_mapping[e_ind].name,
                        ha="center",
                        va="bottom",
                        rotation=rotation,
                        # bbox=dict(facecolor='white', edgecolor='none', pad=0),
                    )
                i_order += 1
                for q_next in e[1:]:
                    q_next_ind = hierq.tail.Q.index(q_next)
                    ax.plot(
                        [x + ddx, x + ddx],
                        [-q_prev_ind, -q_next_ind],
                        color="black",
                        zorder=-1,
                    )
                    # arrow = FancyArrowPatch((x + ddx, -q_prev), (x + ddx,-q_next), arrowstyle='-|>', mutation_scale=10, color='black', zorder=1)
                    # ax.add_patch(arrow)
                    color = get_color(i_order, len(e))
                    circle1 = plt.Circle(
                        (x + ddx, -q_next_ind), small_r, fill=True, color=color
                    )
                    # ax.text(x + ddx, -q_next, i_order, ha="center", va="center")
                    ax.add_artist(circle1)
                    i_order += 1
                    q_prev_ind = q_next_ind
                ddx += dx
        x = x + ddx + dx
        ddx = 0
        layer = layer.next
    plt.axis("off")
    plt.show()
    return fig, ax


def get_color(i, n):
    return cm.Blues((n - i) / n)


def tensor_to_matrix_rowmajor(t0, indices):
    # Get all indices that are going to form rows
    t0_ind_rows = [ind for ind in range(len(t0.shape)) if ind not in indices]
    # Get all indices that are going to form columns
    t0_ind_cols = list(indices)
    new_ind_order = t0_ind_rows + t0_ind_cols
    # Get number of rows
    remaining_idx_ranges = [t0.shape[ind] for ind in t0_ind_rows]
    n_rows = int(np.multiply.reduce(remaining_idx_ranges))
    n_cols = int(np.multiply.reduce([t0.shape[ind] for ind in t0_ind_cols]))
    matrix = np.ascontiguousarray(t0.transpose(new_ind_order).reshape(n_rows, n_cols))
    return matrix, t0_ind_rows, remaining_idx_ranges


def tensor_to_matrix_colmajor(t0, indices):
    # Get all indices that are going to form columns
    t0_ind_cols = [ind for ind in range(len(t0.shape)) if ind not in indices]
    # Get all indices that are going to form rows
    t0_ind_rows = list(indices)
    new_ind_order = t0_ind_cols + t0_ind_rows
    # Get number of rows
    remaining_idx_ranges = [t0.shape[ind] for ind in t0_ind_cols]
    n_rows = int(np.multiply.reduce([t0.shape[ind] for ind in t0_ind_rows]))
    n_cols = int(np.multiply.reduce(remaining_idx_ranges))
    matrix = np.ascontiguousarray(t0.transpose(new_ind_order).reshape(n_rows, n_cols))
    return matrix, t0_ind_cols, remaining_idx_ranges


def contract(t0, t1=None, indices=None):
    if t1 is None:
        # assume t1 is delta and t0 is "hyper square" then trace
        t0_range = len(t0.shape)
        t1 = np.zeros(t0_range**t0_range)
        t1 = t1.reshape([t0_range for i in range(t0_range)])
        for i in range(t0_range):
            t1[(i,) * t0_range] = 1
        # indices should just be one list, so we create another identical one
        indices = [indices, indices]
    a, a_remaining_d, a_idx_ranges = tensor_to_matrix_rowmajor(t0, indices[0])
    b, b_remaining_d, b_idx_ranges = tensor_to_matrix_colmajor(t1, indices[1])
    result = a @ b
    result = result.reshape(a_idx_ranges + b_idx_ranges)
    # The matrix is currently in this order
    current_order = a_remaining_d + list(indices[0])
    # But needs to be transposed back to its original:
    new_ind_order = [current_order.index(i) for i in range(len(result.shape))]
    result = result.transpose(new_ind_order)
    return result


def get_tensor_as_f(u):
    def generic_f(bits, symbols=None, state=None, u=u):
        if len(u.shape) == 2:
            # if u is provided as a matrix, we turn it into the correct tensor
            # for quantum circuits all tensors have as many inputs as outputs
            # all inputs indices also have the same range
            n_inputs = len(bits)
            idx_range = int(
                u.shape[0] ** (1 / n_inputs)
            )  # matrix is square so we can take first or second axis
            if isinstance(u, sp.Matrix):
                # convert to numpy matrix
                f = sp.lambdify(list(u.free_symbols), u, "numpy")
                u = f(*symbols)
            u = u.reshape([idx_range for i in range(n_inputs * 2)])
        new_tensor = contract(state, u, [bits, [i for i in range(len(bits))]])
        # new_tensor = np.tensordot(u, state, axes=[[i for i in range(len(bits))], bits])
        state = new_tensor
        return state

    return generic_f
