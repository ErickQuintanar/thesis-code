import os.path
import torch

import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape

import pandas as pd

from sklearn.model_selection import train_test_split

from alive_progress import alive_bar

np.random.seed(0)
torch.manual_seed(0)

num_qubits = 3
num_layers = 40
learning_rate = 0.005
batch_size = 30
epochs = 10
test_size = 0.4

epsilon = ((np.pi * 2) / 360) * 1

dev = qml.device("default.qubit", wires=num_qubits)

# Strongly entangled binary classificator for diabetes dataset
def variational_classifier(parameters, x):
    '''
        parameters: (layers, qubits, 3)
        x: datapoint
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=0.)
    
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(num_qubits))

    return qml.probs(wires=[0])

# Transform for circuit to add coherent noise
def add_coherent_noise(tape: QuantumTape):
    operations = tape.operations
    new_operations = []

    for op in operations:
        new_operations.append(op)
        if type(op) == qml.ops.op_math.controlled_ops.CNOT:
            new_operations.append(qml.CRX(wires=op.wires, phi=epsilon))
            continue
        elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RY:
            new_operations.append(qml.RY(wires=op.wires, phi=epsilon))
            continue
        elif type(op) == qml.ops.qubit.parametric_ops_single_qubit.RZ:
            new_operations.append(qml.RY(wires=op.wires, phi=epsilon))
    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    return new_tape

# Noisy binary classificator for diabetes dataset
def noisy_variational_classifier(parameters, X):
    predictions = torch.empty(0, requires_grad=True)
    for x in X:
        circuit = qml.QNode(variational_classifier, dev, interface="torch")
        fn, _, _ = circuit.get_gradient_fn(dev, interface="torch")
        circuit(parameters, x)
        noisy_circuit = add_coherent_noise(circuit.tape.expand(depth=3))
        noisy_res = qml.execute([noisy_circuit], dev, interface="torch", grad_on_execution=True, gradient_fn=fn)
        predictions = torch.cat((predictions, noisy_res[0]))
    predictions = torch.reshape(predictions, (int(predictions.size(dim=0)/2), 2))
    return predictions

# Use Binary Cross Entropy Loss
def cost(weights, X, Y):
    predictions = noisy_variational_classifier(weights, X)
    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    return loss(target=Y, input=predictions)

# Choose more likely prediction from probability distribution
def threshold(prediction):
    _, indices = torch.max(prediction, dim=1)
    return indices

# Determine accuracy of predictions
def accuracy(predictions, labels):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

# Retrieve dataset and split the dataset into training and testing sets (60/40 split)
df = pd.read_csv('../../replication-datasets/diabetes_preprocessed.txt', sep='\t')
X = torch.tensor(df.iloc[:, 0:(df.shape[1]-1)].values, requires_grad=False)
Y = torch.tensor(df.iloc[:, -1].values, requires_grad=False)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

weights = torch.tensor(np.random.randn(num_layers, num_qubits, 3), requires_grad=True)

# Initialize optimizer
opt = torch.optim.Adam([weights], lr=learning_rate)

def closure():
    opt.zero_grad()
    loss = cost(weights, X_batch, Y_batch)
    loss.backward()
    return loss

# Train variational classifier
with alive_bar(epochs) as bar:
    for epoch in range(epochs):

        # Update the weights by one optimizer step, using only a limited batch of data
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0,X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            X_batch, Y_batch = X_train[indices], Y_train[indices]
            opt.step(closure)

        # Compute predictions on training set
        print("Computing predictions on training set...")
        predictions = threshold(noisy_variational_classifier(weights, X_train))

        # Compute accuracy on training set
        print("Computing accuracy...")
        acc = accuracy(predictions, Y_train)

        print("Computing current cost...")
        current_cost = cost(weights, X, Y)

        print(f"Epoch: {epoch+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
        bar()

        #TODO: Improve early stopping mechanism
        #TODO: Save weights and bias
        # Early stopping
        if (1 - acc) < 1e-5:
            print("Early stopping...")
            break

# Test variational classifier
predictions_test = threshold(noisy_variational_classifier(weights, X_test))
acc_test = accuracy(predictions_test, Y_test)

print("Accuracy on unseen data:", acc_test)
print(f"L.R.: {learning_rate:f} | Epochs: {epochs:4d} | Layers: {num_layers:4d} | Batch Size: {batch_size:4d} | Accuracy: {acc_test:0.7f}")

# Store experiment results
filename = "reports/diabetes_results.csv"
if os.path.exists(filename):
    # Append result
    with open(filename,'a') as file:
        results = str(learning_rate)+"\t"+str(epochs)+"\t"+str(num_layers)+"\t"+str(batch_size)+"\t"+str(acc_test)+"\t"+str(test_size*100)+"\n"
        file.write(results)
else:
    # Create file and store result
    with open(filename,'w') as file:
        columns = "lr"+"\t"+"epochs"+"\t"+"layers"+"\t"+"batch_size"+"\t"+"accuracy"+"\t"+"test_set_size\n"
        file.write(columns)
        results = str(learning_rate)+"\t"+str(epochs)+"\t"+str(num_layers)+"\t"+str(batch_size)+"\t"+str(acc_test)+"\t"+str(test_size*100)+"\n"
        file.write(results)