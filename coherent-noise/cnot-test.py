from qiskit_aer.noise import NoiseModel, coherent_unitary_error

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt

num_qubits = 2
shots = 1000

# Coherent noise overrotation parameter
epsilon = ((np.pi * 2) / 360) * 10 # ~10 degrees

# Implement coherent noise for all the wires combinations in the CNOT gate. (Reversed CNOT)
cnot_rot = qml.CRX(epsilon, wires=[0,1]).matrix()
cnot_coherent = coherent_unitary_error(cnot_rot)

# Create an empty noise model
coherent_noise = NoiseModel()

# Attach the error to the gates in the circuit
coherent_noise.add_all_qubit_quantum_error(cnot_coherent, ['cx'])

dev_noise = qml.device("qiskit.aer", wires=num_qubits, noise_model=coherent_noise)
dev = qml.device("default.qubit", wires=num_qubits)

def circuit_cnot(x):
    '''
        x: computational basis state
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits))
    qml.CNOT(wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def circuit_cnot_rev(x):
    '''
        x: computational basis state
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits))
    qml.CNOT(wires=range(num_qubits)[::-1])

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

@qml.qnode(dev)
def circuit_cnot_iso(x):
    '''
        x: computational basis state
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits))
    qml.CNOT(wires=range(num_qubits))
    qml.CRX(epsilon, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

@qml.qnode(dev)
def circuit_cnot_rev_iso(x):
    '''
        x: computational basis state
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits))
    qml.CNOT(wires=range(num_qubits)[::-1])
    qml.CRX(epsilon, wires=range(num_qubits)[::-1])

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def average(values_list):
  values = np.array(values_list)
  avgs = np.average(values, axis=0)
  return avgs

def artificial_circuit(operations):
  # Add rotation gates to artificially simulate coherent noise
  for op in operations:
    qml.apply(op)
    if type(op) == qml.ops.op_math.controlled_ops.CNOT:
      qml.CRX(wires=op.wires, phi=epsilon)
      continue
    elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RY:
      qml.RY(wires=op.wires, phi=epsilon)
      continue
    elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RZ:
      qml.RY(wires=op.wires, phi=epsilon)
      continue
  return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def circ(ops):
  for op in ops:
    qml.apply(op)
  return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

def cnot_test(x, shots):
  print("\nCNOT gate coherent noise test with epsilon = "+str(epsilon))
  print("Shots: "+str(shots))
  circuit = qml.QNode(circuit_cnot, dev_noise)

  res = []
  for s in range(shots):
    res.append(circuit(x))
  print("Noisy device")
  print(str(x)+": "+str(average(res)))

  ops = circuit.tape.expand(depth=3).operations
  print(qml.draw(circ, max_length=80)(ops))
  
  circuit = qml.QNode(artificial_circuit, dev)
  print("Artificial simulation")
  print(str(x)+": "+str(circuit(ops)))
  
  print(qml.draw(circuit, expansion_strategy="device", max_length=80)(ops))

def cnot_rev_test(x, shots):
  print("\nCNOT reversed gate coherent noise test with epsilon = "+str(epsilon))
  print("Shots: "+str(shots))
  circuit = qml.QNode(circuit_cnot_rev, dev_noise)

  res = []
  for s in range(shots):
    res.append(circuit(x))
  print("Noisy device")
  print(str(x)+": "+str(average(res)))
  
  ops = circuit.tape.expand(depth=3).operations
  print(qml.draw(circ, max_length=80)(ops))
  
  circuit = qml.QNode(artificial_circuit, dev)
  print("Artificial simulation")
  print(str(x)+": "+str(circuit(ops)))
  
  print(qml.draw(circuit, expansion_strategy="device", max_length=80)(ops))

def cnot_test_iso(x, shots):
  print("\nCNOT gate coherent noise test without state prep noise and with epsilon = "+str(epsilon))
  print("Shots: "+str(shots))

  res = []
  for s in range(shots):
    res.append(circuit_cnot_iso(x))
  print(str(x)+": "+str(average(res)))

  print(qml.draw(circuit_cnot_iso, expansion_strategy="device", max_length=80)(x))

def cnot_rev_test_iso(x, shots):
  print("\nCNOT reversed gate coherent noise test without state prep noise and with epsilon = "+str(epsilon))
  print("Shots: "+str(shots))

  res = []
  for s in range(shots):
    res.append(circuit_cnot_rev_iso(x))
  print(str(x)+": "+str(average(res)))
  
  print(qml.draw(circuit_cnot_rev_iso, expansion_strategy="device", max_length=80)(x))

if __name__ == "__main__":
    cnot_test_iso([1,0,0,0], shots)
    cnot_test_iso([0,1,0,0], shots)
    cnot_test_iso([0,0,1,0], shots)
    cnot_test_iso([0,0,0,1], shots)

    cnot_rev_test_iso([1,0,0,0], shots)
    cnot_rev_test_iso([0,1,0,0], shots)
    cnot_rev_test_iso([0,0,1,0], shots)
    cnot_rev_test_iso([0,0,0,1], shots)
    
    cnot_test([1,0,0,0], shots)
    cnot_test([0,1,0,0], shots)
    cnot_test([0,0,1,0], shots)
    cnot_test([0,0,0,1], shots)
     
    cnot_rev_test([1,0,0,0], shots)
    cnot_rev_test([0,1,0,0], shots)
    cnot_rev_test([0,0,1,0], shots)
    cnot_rev_test([0,0,0,1], shots)