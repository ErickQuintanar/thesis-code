from qiskit_aer.noise import NoiseModel, coherent_unitary_error

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt

'''
Notes:
- Which gates are used for state preparation a.k.a. encoding?
  - If MottonenStatePreparation, then RZ, RY and CNOT are used.
  intro to why we don't do state preparation this way
Questions:
- Experimentally how does coherent noise occur? Is it per qubit or per gate? ANS: Per gate
- How to encode coherent noise for CNOT?
  - Should noise be added if the control bit is not set? No
  - How to deal with different control and target wires? framework does it automatically
- What are values for the noise probability that make sense?
- What is the best comparison for models with and without noise?
  - Accuracy?
  - Weights differences?
  - Fidelity to check how different the quantum state is? maybe state differs but acc remains
- What are the expected results?
- Should we play around with hyperparameters to find optimal case or just do 1:1 comparison?
'''

num_qubits = 1000

ry_count = []
rz_count = []
cnot_count = []

# Coherent noise overrotation parameter
epsilon = ((np.pi * 2) / 360) * 10 # ~10 degrees

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

def count_gates(operations):
  ry = 0
  rz = 0
  cnot = 0
  for op in operations:
    if type(op) == qml.ops.op_math.controlled_ops.CNOT:
      cnot+=1
      continue
    elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RY:
      ry+=1
      continue
    elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RZ:
      rz+=1
      continue
  ry_count.append(ry)
  rz_count.append(rz)
  cnot_count.append(cnot)

def state_preparation_test(x, num_qubits):
  # Apply test for 1, 2, 3 and 4 qubits.
  for n in range(1, num_qubits+1):
    dev_noise = qml.device("qiskit.aer", wires=n, noise_model=coherent_noise)

    print("\nAll-gate coherent noise test with epsilon = "+str(epsilon))

    circuit = qml.QNode(circuit_state_prep, dev_noise)
    res = circuit(x[:(2**n)], n)
    print(str(x[:(2**n)])+": "+str(res))
    ops_1 = circuit.tape.expand(depth=3).operations
    
    circuit = qml.QNode(circuit_mottonen, dev_noise)
    res = circuit(x[:(2**n)], n)
    print(str(x[:(2**n)])+": "+str(res))
    ops_2 = circuit.tape.expand(depth=1).operations
    print(qml.draw(circuit, expansion_strategy="device", max_length=80)(x[:(2**n)], n))

    # Check if circuits are equivalent and count gates
    print("Are both circuits equivalent? "+str(ops_1 == ops_2))
    count_gates(ops_1)

x = np.ones(2**num_qubits)
state_preparation_test(x, num_qubits)

# ry_count = [1, 2, 3, 4, 5, 7, 11, 19, 35, 67]
#            [1, 2, 3, 4, 5, 7, 11, 21, 37, 70]
# NOTE: why is it different in different computers?

x_axis = range(1, (num_qubits+1))
x_axis_upper = np.linspace(0, num_qubits, 100)
cnot_upper = 2**(x_axis_upper + 2) - (4 * x_axis_upper) - 4
rot_upper = 2**(x_axis_upper + 2) - 5

# Graph Möttönen method bounds and observed data
font = {'weight':'bold', 'size':15}
font2 = {'weight':'bold', 'size':12}
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("CNOT gate", fontdict=font)
plt.plot(x_axis_upper, cnot_upper, label="Upper bound")
plt.plot(x_axis, cnot_count, label="CNOT count")
plt.ylabel("Gate count", fontdict=font2)
plt.xlabel("Num. of Qubits", fontdict=font2)
plt.legend(loc='best')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xticks(range(1,num_qubits+1))

plt.subplot(1, 2, 2)
plt.title("Unitary Rotational Gates", fontdict=font)
plt.plot(x_axis_upper, rot_upper, label="Upper bound")
plt.plot(x_axis, ry_count, label="RY count")
plt.plot(x_axis, rz_count, label="RZ count")
plt.ylabel("Gate count", fontdict=font2)
plt.xlabel("Num. of Qubits", fontdict=font2)
plt.legend(loc='best')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xticks(range(1,num_qubits+1))

plt.tight_layout()
plt.savefig('state-prep-gates-count.png')
plt.show()