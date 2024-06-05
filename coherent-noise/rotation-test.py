from qiskit_aer.noise import NoiseModel, coherent_unitary_error

import pennylane as qml
from pennylane import numpy as np

num_qubits = 1
shots = 100

# Coherent noise overrotation parameter
epsilon = ((np.pi * 2) / 360) * 10 # ~10 degrees
theta = np.pi # Equivalent to Pauli gates

# Create the coherent error per gate in the circuit (RY, RZ and CNOT)
ry_rot = qml.RY(epsilon, wires=0).matrix()
ry_coherent = coherent_unitary_error(ry_rot)

rz_rot = qml.RZ(epsilon, wires=0).matrix()
rz_coherent = coherent_unitary_error(rz_rot)

# Create an empty noise model
coherent_noise_ry = NoiseModel()
coherent_noise_rz = NoiseModel()

# Attach the error to the gates in the circuit (RY and RZ)
coherent_noise_ry.add_all_qubit_quantum_error(ry_coherent, ['ry'])
coherent_noise_rz.add_all_qubit_quantum_error(rz_coherent, ['rz'])

dev_ry = qml.device("qiskit.aer", wires=num_qubits, noise_model=coherent_noise_ry)
dev_rz = qml.device("qiskit.aer", wires=num_qubits, noise_model=coherent_noise_rz)
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev_ry)
def circuit_ry(x):
    '''
        x: computational basis state
    '''

    if x[0] == 0 and x[1] == 1:
        qml.PauliX(wires=0)
    qml.RY(theta, wires=0)

    return qml.expval(qml.PauliZ(0))

# NOTE: No noise incurred if dev = dev_rz, however noise is incurred with |1> if dev = dev_ry??????'
# NOTE: StatePrep is not responsible for this behavior, tried to prepare state with X gates and noise was still there
# NOTE: Maybe X gate is implemented with Y rotation gates under the hood(?), but then why is there no noise with the dev_rz device
@qml.qnode(dev_rz)
def circuit_rz(x):
    '''
        x: computational basis state
    '''
    
    if x[0] == 0 and x[1] == 1:
        qml.PauliX(wires=0)
    qml.RZ(theta, wires=0)

    return qml.expval(qml.PauliZ(0))

def average(values_list):
  values = np.array(values_list)
  avgs = np.average(values, axis=0)
  return avgs

def ry_test(x, shots=100):
    print("\nRY gate coherent noise test with epsilon = " + str(epsilon))
    print("Shots: " + str(shots))

    res = []
    for s in range(shots):
        res.append(circuit_ry(x))
    print(str(x) + ": " + str(average(res)))

    print(qml.draw(circuit_ry, expansion_strategy="device", max_length=80)(x))

def rz_test(x, shots=100):
    print("\nRZ gate coherent noise test with epsilon = " + str(epsilon))
    print("Shots: " + str(shots))

    res = []
    for s in range(shots):
        res.append(circuit_rz(x))
    print(str(x) + ": " + str(average(res)))

    print(qml.draw(circuit_rz, expansion_strategy="device", max_length=80)(x))

if __name__ == "__main__":
    ry_test([1, 0], shots)
    ry_test([0, 1], shots)

    rz_test([1, 0], shots)
    rz_test([0, 1], shots)