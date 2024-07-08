import os.path
import torch
import torchvision
import cv2

import pennylane as qml
from pennylane import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

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

def data_preprocessing_mnist(class_labels, im_size=16):
    nr_classes = len(class_labels)
    if not os.path.exists(f'./data/mnist{nr_classes}'):
        os.makedirs(f'./data/mnist{nr_classes}')

    if os.path.exists(f'./data/mnist{nr_classes}/mnist_data.npz'):
        data = np.load(f'./data/mnist{nr_classes}/mnist_data.npz')
        # convert to torch tensors
        X_train = torch.tensor(data['x_train'], requires_grad=False).float()
        Y_train = torch.tensor(data['y_train'], requires_grad=False).float()
        X_test = torch.tensor(data['x_test'], requires_grad=False).float()
        Y_test = torch.tensor(data['y_test'], requires_grad=False).float()
        X_vali = torch.tensor(data['x_vali'], requires_grad=False).float()
        Y_vali = torch.tensor(data['y_vali'], requires_grad=False).float()
        X = torch.cat((X_train, X_test, X_vali))
        Y = torch.cat((Y_train, Y_test, Y_vali))
    else:
        data_set = torchvision.datasets.MNIST("./data/mnist/", train=True, download=True)
        X = []
        Y = []
        for i in range(len(data_set)):
            if data_set[i][1] in class_labels:
                x_rescaled = cv2.resize(np.asarray(data_set[i][0]), (int(im_size), int(im_size)))
                X.append(x_rescaled)
                Y.append(class_labels.index(data_set[i][1]))

        nr_classes = len(class_labels)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = np.reshape(X, (X.shape[0], int(im_size) ** 2))
        Y = np.eye(nr_classes)[Y]

        X = X / 255.0

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_vali, X_test, Y_vali, Y_test = train_test_split(X_test, Y_test, test_size=0.1, random_state=42)

        np.savez(f'./data/mnist{nr_classes}/mnist_data.npz', x_train=X_train, y_train=Y_train, x_test=X_test,
                 y_test=Y_test,
                 x_vali=X_vali, y_vali=Y_vali)
        # convert to torch tensors
        X_train = torch.tensor(X_train, requires_grad=False).float()
        Y_train = torch.tensor(Y_train, requires_grad=False).float()
        X_test = torch.tensor(X_test, requires_grad=False).float()
        Y_test = torch.tensor(Y_test, requires_grad=False).float()
        X_vali = torch.tensor(X_vali, requires_grad=False).float()
        Y_vali = torch.tensor(Y_vali, requires_grad=False).float()

    return X, Y, X_train, Y_train, X_vali, Y_vali, X_test, Y_test

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
X, Y, X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data_preprocessing_mnist(class_labels=[0,1,2,3,4,5,6,7,8,9])
Y_train = threshold(Y_train)
Y_validation = threshold(Y_validation)
Y_test = threshold(Y_test)
Y = threshold(Y)

#weights = torch.tensor(np.random.randn(num_layers, num_qubits, 3), requires_grad=True)
shape_strong = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
weights = torch.tensor((2 * torch.pi * torch.rand(size=shape_strong, requires_grad=False)), requires_grad=True)

specs = qml.specs(variational_classifier, expansion_strategy="device")(weights, X_train[0])
print(specs)

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
        #TODO: Save weights
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