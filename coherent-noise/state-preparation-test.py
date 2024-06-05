from qiskit_aer.noise import NoiseModel, coherent_unitary_error

import pennylane as qml
from pennylane import numpy as np

# NOTE: Test fidelity wrt to circuit length when coherent noise occurs
# NOTE: First test with different lengths
# NOTE: Then test with different noise values
# Total experiments = 5 (qubits values) X 5 (noise values) X 3 (repetitions to average noise) = 75

num_qubits = 5
shots = 1000

# Coherent noise overrotation parameter
epsilon = ((np.pi * 2) / 360) * 10 # ~10 degrees
theta = np.pi # Equivalent to Pauli gates

# Create the coherent error per gate in the circuit (RY, RZ and CNOT)
ry_rot = qml.RY(epsilon, wires=0).matrix()
ry_coherent = coherent_unitary_error(ry_rot)

rz_rot = qml.RZ(epsilon, wires=0).matrix()
rz_coherent = coherent_unitary_error(rz_rot)

cnot_rot = qml.CRX(epsilon, wires=[0,1]).matrix()
cnot_coherent = coherent_unitary_error(cnot_rot)

# Create an empty noise model
coherent_noise = NoiseModel()

# Attach the error to the gates in the circuit (RY, RZ and CNOT)
coherent_noise.add_all_qubit_quantum_error(ry_coherent, ['ry'])
coherent_noise.add_all_qubit_quantum_error(rz_coherent, ['rz'])
coherent_noise.add_all_qubit_quantum_error(cnot_coherent, ['cx'])

def circuit_state_prep(x, num_qubits):
    '''
        x: feature to encode
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=0.)

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def circuit_mottonen(x, num_qubits):
    '''
        x: feature to encode
    '''

    # Add padding if required
    shape = qml.math.shape(x)
    dim = 2 ** num_qubits
    n_features = shape[-1]
    if n_features < dim:
        padding = [0] * (dim - n_features)
        if len(shape) > 1:
            padding = [padding] * shape[0]
        padding = qml.math.convert_like(padding, x)
        x = qml.math.hstack([x, padding])

    # Amplitude encoding with Möttönen method
    state = x / np.linalg.norm(x)
    qml.MottonenStatePreparation(state, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def average(values_list):
  values = np.array(values_list)
  avgs = np.average(values, axis=0)
  return avgs

def state_preparation_test(shots, x, num_qubits):
  # Apply test for 1, 2, 3 and 4 qubits.
  for i in range(num_qubits):
    n = i + 1
    dev = qml.device("qiskit.aer", wires=n, noise_model=coherent_noise)
    #dev = qml.device("default.qubit", wires=n)

    print("\nAll-gate coherent noise test with epsilon = "+str(epsilon))
    print("Shots: "+str(shots))

    circuit = qml.QNode(circuit_state_prep, dev)
    res = []
    for s in range(shots):
      res.append(circuit(x[:(2**n)], n))
    print(str(x[:(2**n)])+": "+str(average(res)))
    print(qml.draw(circuit, expansion_strategy="device", max_length=80)(x[:(2**n)], n))
    
    circuit = qml.QNode(circuit_mottonen, dev)
    res = []
    for s in range(shots):
      res.append(circuit(x[:(2**n)], n))
    print(str(x[:(2**n)])+": "+str(average(res)))
    print(qml.draw(circuit, expansion_strategy="device", max_length=80)(x[:(2**n)], n))

x = np.ones(2**num_qubits)
#x = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
state_preparation_test(shots, x, num_qubits)