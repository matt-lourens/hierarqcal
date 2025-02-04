from argparse import Action
import pennylane as qml
from hierarqcal import Qunitary


class ParseAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)


# set up pennylane circuit
def get_circuit(hierq, embedding="AngleEmbedding", **kwargs):
    dev = qml.device("default.qubit", wires=hierq.tail.Q)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        getattr(qml, embedding)(features=inputs, wires=hierq.tail.Q, **kwargs)
        hierq.set_symbols(weights)
        hierq(backend="pennylane")  # This executes the compute graph in order
        return qml.probs(wires=hierq.head.Q[0])

    return circuit


def penny_gate_to_function(gate):
    return lambda bits, symbols: gate(*symbols, wires=[*bits])


primitive_gates = ["CRZ", "CRX", "CRY", "RZ", "RX", "RY", "Hadamard", "CNOT", "PauliX"]
penny_gates = [getattr(qml, gate_name) for gate_name in primitive_gates]
hierq_gates = {
    primitive_gate: Qunitary(
        penny_gate_to_function(penny_gate),
        n_symbols=penny_gate.num_params,
        arity=penny_gate.num_wires,
    )
    for primitive_gate, penny_gate in zip(primitive_gates, penny_gates)
}
