import os.path
import torch

import pennylane as qml
from pennylane import numpy as np

import pandas as pd

from alive_progress import alive_bar

np.random.seed(0)
torch.manual_seed(0)

num_qubits = 8
num_layers = 40
learning_rate = 0.005
batch_size = 512
epochs = 30
test_size = 0.02
validation_size = 0.18
mem_size = 4000

dev = qml.device("default.qubit", wires=num_qubits)

# Strongly entangled classificator for diabetes dataset
@qml.qnode(dev, interface="torch")
def variational_classifier(parameters, x):
    '''
        parameters: (layers, qubits, 3)
        x: datapoint
    '''

    qml.AmplitudeEmbedding(features=x, wires=range(num_qubits), normalize=True, pad_with=0.)
    
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(num_qubits))

    return qml.probs(wires=[0,1,2,3])

# Use Binary Cross Entropy Loss
def cost(weights, X, Y):
    predictions = variational_classifier(weights, X)
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

# Retrieve dataset and split the dataset into training, validation and testing sets (80/18/2 split)
df = pd.read_csv('../replication-datasets/mnist10_preprocessed.txt', sep='\t')
X = df.iloc[:, 0:(df.shape[1]-1)].values
Y = df.iloc[:, -1].values
train, validation, test = np.split(df.sample(frac=1, random_state=0), [int((1-(validation_size+test_size))*len(df)), int((1-test_size)*len(df))])
Y_train = torch.tensor(train['target'].values, requires_grad=False)
X_train = torch.tensor(train.drop(columns=['target']).values, requires_grad=False)
Y_validation = torch.tensor(validation['target'].values, requires_grad=False)
X_validation = torch.tensor(validation.drop(columns=['target']).values, requires_grad=False)
Y_test = torch.tensor(test['target'].values, requires_grad=False)
X_test = torch.tensor(test.drop(columns=['target']).values, requires_grad=False)

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

        # Compute predictions on train and validation set
        predictions_train = torch.empty(0)
        predictions_val = torch.empty(0)

        dataloader = torch.utils.data.DataLoader(X_train, batch_size=mem_size, shuffle=False)
        print("Computing predictions on training set...")
        for batch in dataloader:
            predictions_train = torch.cat((predictions_train, threshold(variational_classifier(weights, batch))))

        dataloader = torch.utils.data.DataLoader(X_validation, batch_size=mem_size, shuffle=False)
        print("Computing predictions on validation set...")
        for batch in dataloader:
            predictions_val = torch.cat((predictions_val, threshold(variational_classifier(weights, batch))))

        # Compute accuracy on train and validation set
        print("Computing accuracies...")
        acc = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_validation, predictions_val)
        
        print("Computing current cost...")
        dataloader_x = torch.utils.data.DataLoader(X, batch_size=mem_size, shuffle=False)
        dataloader_y = torch.utils.data.DataLoader(Y, batch_size=mem_size, shuffle=False)
        costs = torch.empty(0)
        for b_x, b_y in zip(dataloader_x, dataloader_y):
            costs = torch.cat((costs, torch.tensor([cost(weights, b_x, b_y)])))
        current_cost = torch.mean(costs)

        print(f"Epoch: {epoch+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f} | Accuracy Val.: {acc_val:0.7f}")
        bar()

        #TODO: Improve early stopping mechanism
        #TODO: Save weights and bias
        # Early stopping
        if (1 - acc) < 1e-5:
            print("Early stopping...")
            break

# Test variational classifier
dataloader = torch.utils.data.DataLoader(X_test, batch_size=mem_size, shuffle=False)
predictions_test = torch.empty(0)
for batch in dataloader:
    predictions_test = torch.cat((predictions_test, threshold(variational_classifier(weights, batch))))
acc_test = accuracy(predictions_test, Y_test)

print("Accuracy on unseen data:", acc_test)
print(f"L.R.: {learning_rate:f} | Epochs: {epochs:4d} | Layers: {num_layers:4d} | Batch Size: {batch_size:4d} | Accuracy: {acc_test:0.7f}")

# Store experiment results
filename = "reports/mnist10_results.csv"
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