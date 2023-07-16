import pytest
import numpy as np
import itertools as it
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpivot,
    Qpermute,
    Qmask,
    Qunmask,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    Qunitary,
)

CAP = 10


@pytest.mark.parametrize(
    "nq, motif, arity, n_symbols",
    it.product(
        [i for i in range(4, CAP, 1)],
        [Qcycle, Qpermute, Qmask],  # TODO add pivot
        [1, 2, 3, 4],
        [i for i in range(0, CAP, 1)],
    ),
)
def test_motif_get_symbols_and_n_symbols_match(nq, motif, arity, n_symbols):
    """
    Test share_weights hyperparameter for motif:
        expectation: the number of symbols for a motif must be correctly determined based on this parameter
    """
    u = Qunitary(lambda x: True, arity=arity, n_symbols=n_symbols)

    # Share weights is False, so there should be a symbol for each edge
    c_0 = motif(mapping=u)
    test_0 = Qinit(nq) + c_0
    # Test the motif
    assert test_0[1].n_symbols == len(list(test_0[1].get_symbols()))
    # Test the full hierarchy
    assert test_0.n_symbols == len(list(test_0.get_symbols()))
    # Test cross (since there is only one motif, this mus be the same)
    assert test_0.n_symbols == test_0[1].n_symbols


@pytest.mark.parametrize(
    "nq, motif, arity, n_symbols",
    it.product(
        [i for i in range(4, CAP, 1)],
        [Qcycle, Qpermute, Qmask],  # TODO add pivot
        [1, 2, 3, 4],
        [i for i in range(0, CAP, 1)],
    ),
)
def test_motif_share_weights_symbolic(nq, motif, arity, n_symbols):
    """
    Test share_weights hyperparameter for motif:
        expectation: the number of symbols for a motif must be correctly determined based on this parameter
    """
    # === dev ===
    # n_symbols = 0
    # N = 1
    # arity =1
    # motif = Qcycle
    # === dev ===
    u = Qunitary(lambda x: True, arity=arity, n_symbols=n_symbols)

    # Share weights is False, so there should be a symbol for each edge
    c_0 = motif(mapping=u, share_weights=False)
    test_0 = Qinit(nq) + c_0
    assert test_0[1].n_symbols == len(test_0[1].E) * test_0[1].mapping.n_symbols

    # Share weights is True, so there should be as many symbols as was provided
    c_1 = motif(mapping=u, share_weights=True)
    test_1 = Qinit(nq) + c_1
    assert test_1[1].n_symbols == test_1[1].mapping.n_symbols


@pytest.mark.parametrize(
    "nq, motif, arity, symbols",
    it.product(
        [i for i in range(4, CAP, 1)],
        [Qcycle, Qpermute, Qmask],  # TODO add pivot
        [1, 2, 3, 4],
        [np.random.uniform(-10, 10, i) for i in range(0, CAP, 1)],
    ),
)
def test_motif_share_weights_numeric(nq, motif, arity, symbols):
    """
    Test Numeric Symbols:
        motif: cycle of rx gates with 1 numerical symbol
        expectation: the numeric symbol gets repeated irrespective of share_weights
    """
    u = Qunitary(lambda x: True, arity=arity, symbols=symbols)
    # === dev ===
    # n_symbols = 5
    # arity =1
    # motif = Qcycle
    # symbols = np.random.uniform(-10, 10, n_symbols)
    # u = Qunitary(lambda x: True, arity=arity, symbols=symbols)
    # === dev ===

    # Share weights is False, so there should be a symbol for each edge
    c_0 = motif(mapping=u, share_weights=False)
    test_0 = Qinit(nq) + c_0
    assert test_0[1].n_symbols == len(test_0[1].E) * test_0[1].mapping.n_symbols
    # Ensure symbols were repeated correctly
    assert all(test_0[1].edge_mapping[0].symbols == symbols)

    # Share weights is True, so there should be as many symbols as was provided
    c_1 = motif(mapping=u, share_weights=True)
    test_1 = Qinit(nq) + c_1
    assert test_1[1].n_symbols == test_1[1].mapping.n_symbols
    # == For dev purposes ==
    # circuit = test_0(backend="qiskit")
    # circuit.draw("mpl")
    # ====


def test_symbol_function_share_weights():
    """
    Test symbol_fn functionality:
        expectation: the numeric symbol should be transformed by the symbol_fn based on the # edge it is.
    """
    # === dev ===
    nq = 5
    arity = 2
    motif = Qcycle
    symbols = [np.pi]
    symbol_fn = lambda x, ns, ne: x / (ns)
    u = Qunitary("crx(x)^01", symbols=symbols)
    # === dev ===

    # Share weights is False, so there should be a symbol for each edge
    c_0 = motif(mapping=u, share_weights=False, symbol_fn=symbol_fn)
    test_0 = Qinit(nq) + c_0
    assert list(test_0[1].get_symbols()) == [
        3.141592653589793,
        1.5707963267948966,
        1.0471975511965976,
        0.7853981633974483,
        0.6283185307179586,
    ]

    # Test share weights is true
    c_1 = motif(mapping=u, share_weights=True, symbol_fn=symbol_fn)
    test_1 = Qinit(nq) + c_1
    assert list(test_1[1].get_symbols()) == symbols


def test_symbol_function_by_idx():
    """
    Test symbol_fn functionality:
        expectation: the numeric symbol should be transformed by the symbol_fn based on the # edge it is.

        TODO finish test, I don't think share weights should influence the ne functions.
    """
    # === dev ===
    nq = 10
    arity = 2
    motif = Qcycle
    symbols = [np.pi, np.sqrt(2)]
    symbol_fn = lambda x, ns, ne: x / (ns)
    u = Qunitary("crx(x)^01;cry(y)^10", symbols=symbols)
    # === dev ===
    # Test across idx
    test_00 = Qinit(nq) + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn) * 2
    test_01 = (
        Qinit(nq)
        + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn)
        + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn)
    )
    test_10 = (
        Qinit(nq)
        + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn)
        + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn)
    )
    test_11 = Qinit(nq) + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn) * 2

    # Test across edge
    symbol_fn = lambda x, ns, ne: x / (ne)
    # Test across idx
    test_00 = Qinit(nq) + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn) * 2
    test_01 = (
        Qinit(nq)
        + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn)
        + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn)
    )
    test_10 = (
        Qinit(nq)
        + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn)
        + motif(mapping=u, share_weights=False, symbol_fn=symbol_fn)
    )
    test_11 = Qinit(nq) + motif(mapping=u, share_weights=True, symbol_fn=symbol_fn) * 2
    circuit = test_00(backend="qiskit")
    # circuit.draw("mpl")


def test_symbol_function_sub():
    """
    Test symbol_fn functionality:
        expectation: the numeric symbol should be transformed by the symbol_fn based on the # edge it is.

        TODO finish test, I don't think share weights should influence the ne functions.
    """
    # === dev ===
    nq1 = 3
    nq2 = 10
    nq3 = 20
    motif = Qcycle
    symbols = [-1, 1]
    symbol_fn = lambda x, ns, ne: x / (ne)
    u = Qunitary("crx(x)^01;cry(y)^10", symbols=symbols)
    # === dev ===
    # Test across idx
    test_0 = Qinit(nq1) + motif(mapping=u, share_weights=False)
    test_1 = Qinit(nq1) + motif(mapping=u, share_weights=True)

    test_00 = Qinit(nq2) + motif(
        step=3, mapping=test_0, share_weights=False, boundary="open"
    )
    test_01 = Qinit(nq2) + motif(
        step=3,
        mapping=test_1,
        share_weights=False,
    )
    test_10 = Qinit(nq2) + motif(
        step=3,
        mapping=test_0,
        share_weights=True,
    )
    test_11 = Qinit(nq2) + motif(
        step=3,
        mapping=test_1,
        share_weights=True,
    )

    test_0 = Qinit(nq1) + motif(mapping=u, share_weights=False)
    test_1 = Qinit(nq1) + motif(mapping=u, share_weights=True)

    test_00 = Qinit(nq2) + motif(
        step=3,
        mapping=test_0,
        share_weights=False,
        boundary="open",
    )
    test_01 = Qinit(nq2) + motif(
        step=3,
        mapping=test_1,
        share_weights=False,
        symbol_fn=symbol_fn,
    )
    test_10 = Qinit(nq2) + motif(
        step=3,
        mapping=test_0,
        share_weights=True,
        symbol_fn=symbol_fn,
    )
    test_11 = Qinit(nq2) + motif(
        step=3,
        mapping=test_1,
        share_weights=True,
        symbol_fn=symbol_fn,
    )

    test_000 = Qinit(nq3) + motif(
        step=nq2, mapping=test_00, share_weights=True, symbol_fn=symbol_fn
    )
    circuit = test_000(backend="qiskit", barriers=True)
    # circuit.draw("mpl")


if __name__ == "__main__":
    cap = 10  # TODO once the deppcopy bug is fixed, remove this cap
    nq = np.random.randint(2, cap)
    arity = np.random.randint(1, nq)
    n_symbols = np.random.randint(0, 2 * nq)
    symbols = np.random.uniform(-10, 10, n_symbols)
    motif = np.random.choice([Qcycle, Qpermute, Qmask])

    # test_motif_get_symbols_and_n_symbols_match(nq, motif, arity, n_symbols)
    # test_motif_share_weights_symbolic(nq, motif, arity, n_symbols)
    # test_motif_share_weights_numeric(nq, motif, arity, symbols)
    # test_symbol_function_share_weights()
    # test_symbol_function_by_idx()
    test_symbol_function_sub()
