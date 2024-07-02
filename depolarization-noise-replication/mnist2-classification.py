import os.path

import pennylane as qml
from pennylane import numpy as np

import pandas as pd

from alive_progress import alive_bar

np.random.seed(0)

num_qubits = 8
num_layers = 40
learning_rate = 0.005
batch_size = 512
epochs = 30
test_size = 0.02
validation_size = 0.18

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

# Strongly entangled binary classificator for MNIST2 dataset
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

# Retrieve dataset and split the dataset into training, validation and testing sets (80/18/2 split)
df = pd.read_csv('../replication-datasets/mnist2_preprocessed.txt', sep='\t')
X = df.iloc[:, 0:(df.shape[1]-1)].values
Y = df.iloc[:, -1].values
train, validation, test = np.split(df.sample(frac=1, random_state=0), [int((1-(validation_size+test_size))*len(df)), int((1-test_size)*len(df))])
Y_train = train['target'].values
X_train = train.drop(columns=['target']).values
Y_validation = validation['target'].values
X_validation = validation.drop(columns=['target']).values
Y_test = test['target'].values
X_test = test.drop(columns=['target']).values

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

        # Compute predictions on train and validation set
        print("Computing predictions on training set...")
        predictions_train = threshold(variational_classifier(weights, bias, X_train))
        print("Computing predictions on validation set...")
        predictions_val = threshold(variational_classifier(weights, bias, X_validation))

        # Compute accuracy on train and validation set
        print("Computing accuracies...")
        acc = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_validation, predictions_val)
        
        print("Computing current cost...")
        current_cost = cost(weights, bias, X, Y)

        print(f"Epoch: {epoch+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f} | Accuracy Val.: {acc_val:0.7f}")
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
filename = "reports/mnist2_results.csv"
if os.path.exists(filename):
    # Append result
    with open(filename,'a') as file:
        results = str(learning_rate)+"\t"+str(epochs)+"\t"+str(num_layers)+"\t"+str(batch_size)+"\t"+str(acc_test)+"\t"+str(validation_size*100)+"\t"+str(test_size*100)+"\n"
        file.write(results)
else:
    # Create file and store result
    with open(filename,'w') as file:
        columns = "lr"+"\t"+"epochs"+"\t"+"layers"+"\t"+"batch_size"+"\t"+"accuracy"+"\t"+"validation_set_size"+"\t"+"test_set_size\n"
        file.write(columns)
        results = str(learning_rate)+"\t"+str(epochs)+"\t"+str(num_layers)+"\t"+str(batch_size)+"\t"+str(acc_test)+"\t"+str(validation_size*100)+"\t"+str(test_size*100)+"\n"
        file.write(results)