# Quantum specific imports
import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding

DEVICE = qml.device("default.qubit", wires=8)

@qml.qnode(DEVICE, interface="torch") 
def quantum_node(X, classifier):
    if getattr(classifier, "numpy", False):
        # If classifier needs to be deserialized
        classifier = classifier.numpy()
    n_wires = classifier.n_wires
    # Encode data
    AngleEmbedding(X, wires=range(n_wires), rotation="Y")
    # Evaluate circuit
    classifier._evaluate()
    # Obtain probability
    result = qml.probs(wires=classifier.response_wire_)
    return result

# Circuits
def U(params, wires):
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])

def V(params, wires):
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])