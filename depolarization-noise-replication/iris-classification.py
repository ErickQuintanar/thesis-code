import os.path

import pennylane as qml
from pennylane import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from alive_progress import alive_bar

np.random.seed(0)

num_qubits = 2
num_layers = 2
learning_rate = 0.05
batch_size = 30
epochs = 100
test_size = 0.4

dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit(parameters, x):
    '''
        parameters: (layers, qubits, 3)
        x: datapoint
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=0.)

    qml.StronglyEntanglingLayers(weights=parameters, wires=range(num_qubits))

    return qml.expval(qml.PauliZ(0))

shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
parameters = np.random.random(size=shape)

# Strongly entangled binary classificator for iris dataset
def variational_classifier(weights, bias, X):
    preds = circuit(weights, X) + bias
    # Rescale value between 0 and 1
    return (preds + 1) * 0.5

# Use Binary Cross Entropy Loss
def cost(weights, bias, X, Y):
    predictions = variational_classifier(weights, bias, X)
    loss = -(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions)).mean()
    return loss

def threshold(prediction):
    return np.where(prediction > 0.5, 1, 0)

# Determine accuracy of predictions
def accuracy(predictions, labels):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

# Retrieve dataset and split the dataset into training and testing sets (60/40 split)
df = pd.read_csv('../replication-datasets/iris_preprocessed.txt', sep='\t')
X = df.iloc[:, 0:(df.shape[1]-1)].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

weights = np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias = np.array(0.0, requires_grad=True)

# Initialize optimizer
opt = qml.AdamOptimizer(stepsize=learning_rate)

# Train variational classifier
with alive_bar(epochs) as bar:
    for epoch in range(epochs):

        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        # Compute predictions on training set
        print("Computing predictions on training set...")
        predictions = threshold(variational_classifier(weights, bias, X_train))

        # Compute accuracy on training set
        print("Computing accuracy...")
        acc = accuracy(predictions, Y_train)

        print("Computing current cost...")
        current_cost = cost(weights, bias, X, Y)

        print(f"Epoch: {epoch+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
        bar()

        #TODO: Improve early stopping mechanism
        #TODO: Save weights and bias
        # Early stopping
        if (1 - acc) < 1e-5:
            print("Early stopping...")
            break

# Test variational classifier
predictions_test = threshold(variational_classifier(weights, bias, X_test))
acc_test = accuracy(predictions_test, Y_test)

print("Accuracy on unseen data:", acc_test)
print(f"L.R.: {learning_rate:f} | Epochs: {epochs:4d} | Layers: {num_layers:4d} | Batch Size: {batch_size:4d} | Accuracy: {acc_test:0.7f}")

# Store experiment results
filename = "reports/iris_results.csv"
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