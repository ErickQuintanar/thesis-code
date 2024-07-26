import pennylane as qml

from pennylane import numpy as np

import torch

# TODO: Check the right loss function for the PQC models (maybe check torch.nn.NLLLoss and how many classes from the probs vector we are taking)

def define_model(config):
    # Check which QML model to use
    dev = qml.device("default.mixed", wires=config["num_qubits"])
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    if config["qml_model"] == "pqc":
        # Check noise model to determine which device and model to use
        if config["noise_model"] == "none":
            print("Setting up noiseless PQC model.")
            return pqc(dev, config), loss
        elif config["noise_model"] == "coherent":
            # TODO: Figure out how to inject artificial coherent noise
            print("Coherent PQC model incoming")
            return pqc_coherent_noise(dev, config), loss
        else:
            print("Setting up noise PQC model.")
            return pqc_incoherent_noise(dev, config), loss
    elif config["qml_model"] == "kernel":
        print("Model type implementation coming soon!")
    else:
        print("Model type can't be found")


def pqc(dev, config):
    def strongly_entangling_layers(weights, wires):
        shape = qml.math.shape(weights)[-3:]
        n_layers = qml.math.shape(weights)[-3]
        wires = qml.wires.Wires(wires)
        ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))

        for l in range(n_layers):
            for i in range(len(wires)):
                qml.RZ(phi=weights[..., l, i, 0], wires=[wires[i]])
                qml.RY(phi=weights[..., l, i, 1], wires=[wires[i]])
                qml.RZ(phi=weights[..., l, i, 2], wires=[wires[i]])

            if len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    qml.CNOT(wires=act_on)

    @qml.qnode(dev, interface="torch")
    def pqc_circuit(parameters, x):
            '''
            parameters: (layers, qubits, 3)
            x: datapoint
            '''
            qml.AmplitudeEmbedding(features=x, wires=range(config["num_qubits"]), normalize=True, pad_with=0.)
            strongly_entangling_layers(weights=parameters, wires=range(config["num_qubits"]))

            return qml.probs(wires=range(int(np.ceil(np.log2(config["num_classes"])))))
    return pqc_circuit

def pqc_incoherent_noise(dev, config):
    p = config["probability"]

    # Adjust noise injection for differents types of incoherent noise
    if config["noise_model"] == "depolarizing":
        operation = qml.DepolarizingChannel
        print("Setting up depolarizing PQC model.")
    elif config["noise_model"] == "bit-flip":
        operation = qml.BitFlip
        print("Setting up bit-flip PQC model.")
    elif config["noise_model"] == "phase-flip":
        operation = qml.PhaseFlip
        print("Setting up phase-flip PQC model.")
    elif config["noise_model"] == "phase-damping":
        operation = qml.PhaseDamping
        print("Setting up phase-damping PQC model.")
    elif config["noise_model"] == "amplitude-damping":
        operation = qml.AmplitudeDamping
        print("Setting up amplitude-damping PQC model.")
    return qml.transforms.insert(pqc(dev, config), op=operation, op_args=p, position="all")

def pqc_coherent_noise(dev, config):
    # Build coherent noise model
    theta = config["miscalibration"]

    c_ry = qml.noise.op_eq(qml.RY)
    c_rz = qml.noise.op_eq(qml.RZ)
    c_cnot = qml.noise.op_eq(qml.CNOT)

    def n_ry(op, **metadata):
        qml.RY(((np.pi * 2) / 360) * metadata["theta"], op.wires)

    def n_rz(op, **metadata):
        qml.RZ(((np.pi * 2) / 360) * metadata["theta"], op.wires)
    
    def n_cnot(op, **metadata):
        qml.CRX(((np.pi * 2) / 360) * metadata["theta"], op.wires)

    noise_model = qml.NoiseModel({c_ry: n_ry, c_rz: n_rz, c_cnot: n_cnot}, theta=theta)

    return qml.transforms.add_noise(pqc(dev, config), noise_model=noise_model)
