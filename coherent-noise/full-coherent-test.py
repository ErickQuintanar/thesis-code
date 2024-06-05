from qiskit_aer.noise import NoiseModel, coherent_unitary_error

import pennylane as qml
from pennylane import numpy as np

'''
TODO: Investigate ReadoutError for measurements
Notes:
- Which gates are used for state preparation a.k.a. encoding?
  - If MottonenStatePreparation, then RZ, RY and CNOT are used.
  intro to why we don't do state preparation this way
Questions:
- Experimentally how does coherent noise occur? Is it per qubit or per gate? ANS: Per gate
- How to encode coherent noise for CNOT?
  - Should noise be added if the control bit is not set?
  - How to deal with different control and target wires? to test
- What are values for the noise probability that make sense?
- What is the best comparison for models with and without noise?
  - Accuracy?
  - Weights differences?
  - Fidelity to check how different the quantum state is? maybe state differs but acc remains
- What are the expected results?
- Should we play around with hyperparameters to find optimal case or just do 1:1 comparison?
'''

num_qubits = 2
shots = 100

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

dev = qml.device("qiskit.aer", wires=num_qubits, noise_model=coherent_noise)
#dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit(x):
    '''
        x: computational basis state
    '''

    qml.StatePrep(state=x, wires=range(num_qubits))
    for i in range(num_qubits):
      qml.RY(theta,wires=i)
      qml.RZ(theta,wires=i)
    qml.CNOT(wires=range(num_qubits))

    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

def average(values_list):
  values = np.array(values_list)
  avgs = np.average(values, axis=0)
  return avgs

def circuit_test(shots):
  print("All-gate coherent noise test with epsilon = "+str(epsilon))
  print("Shots: "+str(shots))

  res = []
  x = [1,0,0,0]
  for s in range(shots):
    res.append(circuit(x))
  print(str(x)+": "+str(average(res)))

  res = []
  x = [0,1,0,0]
  for s in range(shots):
    res.append(circuit(x))
  print(str(x)+": "+str(average(res)))

  res = []
  x = [0,0,1,0]
  for s in range(shots):
    res.append(circuit(x))
  print(str(x)+": "+str(average(res)))

  res = []
  x = [0,0,0,1]
  for s in range(shots):
    res.append(circuit(x))
  print(str(x)+": "+str(average(res)))

  print(qml.draw(circuit, expansion_strategy="device", max_length=80)(x))

circuit_test(shots)
