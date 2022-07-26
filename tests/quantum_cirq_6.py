# %%
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

# visualization tools

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# %%

a, b = sympy.symbols("a b")
# %%
q0, q1 = cirq.GridQubit.rect(1, 2)
circuit = cirq.Circuit(
    cirq.rx(a).on(q0), cirq.ry(b).on(q1), cirq.CNOT(control=q0, target=q1)
)

SVGCircuit(circuit)
# %%
resolver = cirq.ParamResolver({a: 0.5, b: 0.5})
output_state_vector = cirq.Simulator().simulate(circuit, resolver).final_state_vector
output_state_vector
# %%
z0 = cirq.Z(q0)
qubit_map = {q0: 0, q1: 1}
z0.expectation_from_state_vector(output_state_vector, qubit_map).real
# %%
circuit_tensor = tfq.convert_to_tensor([circuit])
print(circuit_tensor.shape)
# %%
batch_vals = np.array(np.random.uniform(0, 2 * np.pi, (5, 2)), dtype=np.float32)
cirq_results = []
cirq_simulator = cirq.Simulator()

for vals in batch_vals:
    resolver = cirq.ParamResolver({a: vals[0], b: vals[1]})
    final_state_vector = cirq_simulator.simulate(circuit, resolver).final_state_vector
    cirq_results.append(
        [z0.expectation_from_state_vector(final_state_vector, {q0: 0, q1: 1}).real]
    )
print(f"cirq batch results: \n{np.array(cirq_results)}")
# %%
tfq.layers.Expectation()(
    circuit, symbol_names=[a, b], symbol_values=batch_vals, operators=z0
)
# %%
control_params = sympy.symbols("theta_1 theta_2 theta_3")
qubit = cirq.GridQubit(0, 0)
model_circuit = cirq.Circuit(
    cirq.rz(control_params[0])(qubit),
    cirq.ry(control_params[0])(qubit),
    cirq.rx(control_params[0])(qubit),
)
SVGCircuit(model_circuit)

# %%
controller = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, activation="elu"), tf.keras.layers.Dense(3)]
)

# %%
circuit_input = tf.keras.Input(shape=(), dtype=tf.string, name="circuit_input")
command_input = tf.keras.Input(
    shape=(1,), dtype=tf.dtypes.float32, name="command_input"
)

# %%
dense_2 = controller(command_input)
expectation_layer = tfq.layers.ControlledPQC(model_circuit, operators=cirq.Z(qubit))
expectation = expectation_layer([circuit_input, dense_2])
# The full Keras model is built from our layers.
model = tf.keras.Model(inputs=[circuit_input, command_input],
                       outputs=expectation)

tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)
# %%
