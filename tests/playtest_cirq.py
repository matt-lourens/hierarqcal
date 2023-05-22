"""
TODO one day these will turn into nice tests, the idea is to run them one by one in debug mode. There are a lot of plots so make sure you have break points
"""
# %%
import numpy as np
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
from hierarqcal.cirq.cirq_circuits import V2, U2
from cirq.contrib.svg import SVGCircuit
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
# %%
# Test default
hierq = Qinit(8) + Qcycle(1, 1, 0)
circuit = hierq(backend="cirq")
SVGCircuit(circuit)
# %%
# Test masking
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right"))*3
circuit = hierq(backend="cirq")
SVGCircuit(circuit)
# %%
# Test masking
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right"))*3 + (Qunmask("previous") + Qcycle(1, 1, 0))*3
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
# %%
# Test unmasking nothing
hierq = Qinit(8) + Qunmask("previous") + Qunmask("previous")
circuit= hierq(backend="cirq")
# SVGCircuit(circuit)
# %%
# Test unmasking nothing
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right"))*3 + (Qunmask("previous") + Qcycle(1, 1, 0))*4
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
# %%
# Test unmasking patterns
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right"))*3 + (Qunmask("right") + Qcycle(1, 1, 0))*4
circuit= hierq(backend="cirq")
SVGCircuit(circuit)

# %%
# # Test qcnn
u = Qunitary(U2, 1, 2)
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right", mapping=u))*3
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
# %%
# # Test sub TODO demo
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)

circuit= hierq(backend="cirq")
SVGCircuit(circuit)

# %%
# Test sub sub TODO demo
u = Qunitary(U2, 1, 2)
subsub = Qinit(3) + Qpermute(mapping=u)
sub = Qinit(12) + Qcycle(
    1, 2, 0, mapping=subsub, share_weights=True, boundary="open"
)
hierq = Qinit(24) + Qcycle(1, 11, 0, mapping=sub, share_weights=True, boundary="open")
circuit= hierq(backend="cirq")
SVGCircuit(circuit)

# %%
# # Test sub weights TODO demo
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit= hierq(backend="cirq")
SVGCircuit(circuit)

u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=True)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=True)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit= hierq(backend="cirq")
SVGCircuit(circuit)
print("Hello")
# %%
