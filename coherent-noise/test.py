import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(device=dev)
def circuit():
    qml.RZ(phi=(np.pi), wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(device=dev)
def circuit_2():
    qml.PauliX(wires=0)
    qml.RZ(phi=(np.pi + 0.175), wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(device=dev)
def circuit_3():
    qml.RZ(phi=(np.pi), wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(device=dev)
def circuit_4():
    qml.PauliX(wires=0)
    qml.RZ(phi=(np.pi + 0.175), wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(device=dev)
def circuit_4():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    qml.RY(phi=(np.pi), wires=0)
    return qml.expval(qml.PauliZ(0))

print(circuit())
print(circuit_2())
print(circuit_3())
print(circuit_4())