from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motifs,
    plot_motif,
    Qunitary,
)

primitive = Qinit(8) + Qcycle()
plot_motif(primitive.head)

primitive = Qinit(15) + Qcycle(stride=1, step=3, offset=0, mapping=Qunitary(arity=3))
plot_motif(primitive.head)

# primitive = Qinit(8) + Qmask("right") + Qunmask("previous")
# plot_motif(primitive.head)

# primitive = Qinit(8) + Qcycle(1,2,0,mapping=Qunitary(None,1,3))
# plot_motif(primitive.head)

# primitive = Qinit(8) + Qpermute()
# plot_motif(primitive.head)

primitive = (
    Qinit(4) + Qcycle() + Qmask("inside") + Qcycle() + Qunmask("previous") + Qcycle()
)
plot_motif(primitive[4])
print("Hey")

# ==== Old code examples and paper plots ===
# kets_q = np.array([pos[q] for q in edge]).T
# ket_start = kets_q.T[0]
# ket_end = kets_q.T[-1]
# ket_d = ket_end - ket_start
# eps = node_large
# q_radis = np.array([node_radi[primitive.Q.index(q)] + eps for q in edge])
# ket_scale = (
#     np.ones(
#         len(edge),
#     )
#     + q_radis
# )
# kets_q_scaled = np.multiply(kets_q, ket_scale)
# # For start and end add points to go around the node
# #ket_d2 = ket_start + (1 + q_radis[-1]) * ket_d
# ket_d3 = ket_end+q_radis[-1]* (-kets_q_scaled.T[-1]+ket_d)
# ket_d4 = ket_start+q_radis[0]* (-kets_q_scaled.T[0]+(-1*ket_d))
# #ket_d5 = ket_end + -1 * (1 + q_radis[-1]) * ket_d
# # hypergraph edge
# control_points = (
#     [pos for pos in kets_q_scaled.T]
#     #+ [ket_d2]
#     + [ket_d3]
#     + [ket_d4]
#     #+ [ket_d5]
#     + [kets_q_scaled.T[0]]
# )
# codes = [Path.MOVETO] + [Path.LINETO for point in control_points[1:]]
# path = Path(control_points, codes)
# patch = patches.PathPatch(
#     path, facecolor="none", edgecolor="orange", linewidth=2
# )
# ax.add_patch(patch)

# === Testing ===
# for stride in [1,3,5,7]:
#     m = Qinit(8) + Qcycle(stride)
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
# m = Qinit(9) + Qcycle(stride, step, offset, qpu=3, boundary="open")
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
#     edges_kwargs={"edgecolors": plt.cm.tab10(7), "linewidths": 2, "dr": 0.03},
# )
# # Set title of plot
# ax.set_title(f"Stride: {stride}, Step: {step}, Offset: {offset}")
# # Save fig svg
# fig.savefig(f"/home/matt/Downloads/stride_{stride}_step_{step}_offset_{offset}.svg", format="svg")
