from datasets.fetch_data import fetch_dataset, Dataset
from models.qml_models import define_model
from training.lightning_utils import QMLModel, TrainingModule

import torch
import pennylane as qml
from pennylane import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import uuid
import json
import os

# Fix random generator
np.random.seed(0)

def train_script(config):
    # Retrieve dataset according to the config (datasets/fetch_data.py)
    X_train, X_test, Y_train, Y_test = fetch_dataset(config["dataset"])

    # Define model based on charateristics from the config
    qnode, loss = define_model(config)

    # Create weights for training the specific model
    weights = torch.tensor(np.random.randn(config["num_layers"], config["num_qubits"], 3), requires_grad=True)
    model = QMLModel(qnode, weights, config)
    print(qml.draw(qnode, expansion_strategy="device", max_length=80)(weights, X_train[0]))
    
    # Train model with lightning module
    training_model = TrainingModule(model, loss, lambda params: torch.optim.Adam(params, lr=config["learning_rate"]))

    train_set = Dataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=os.cpu_count())
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="cpu")

    # Adjust datasets and training if dataset has validation dataset
    if config["dataset"] == "mnist2" or config["dataset"] == "mnist4" or config["dataset"] == "mnist10":
        trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="cpu", limit_train_batches=0.2,  num_sanity_val_steps=0)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0, stratify=Y_test)
        val_set = Dataset(X_val, Y_val)
        test_set = Dataset(X_test, Y_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=os.cpu_count())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=os.cpu_count())
        trainer.fit(training_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elif config["dataset"] == "plus-minus":
        trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="cpu", limit_train_batches=0.2, num_sanity_val_steps=0)
        test_set = Dataset(X_test, Y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=os.cpu_count())
        trainer.fit(training_model, train_dataloaders=train_loader)
    else:
        trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="cpu")
        test_set = Dataset(X_test, Y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=os.cpu_count())
        trainer.fit(training_model, train_dataloaders=train_loader)

    # Test model on unseen data
    trainer.test(training_model, dataloaders=[test_loader])
    print("Accuracy on test: "+str(training_model.acc))
    config["accuracy"] = training_model.acc

    # Once model's training is finished, save weights and config to done experiments
    trained_weights = model.weights.detach().numpy()
    unique_id = str(uuid.uuid4())
    if config["noise_model"] == "none":
        weights_path = "../results/weights/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+unique_id+".npy"
        config_path = "../results/configs/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+unique_id+".json"
    elif config["noise_model"] == "coherent":
        weights_path = "../results/weights/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+str(int(config["miscalibration"]))+"-"+unique_id+".npy"
        config_path = "../results/configs/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+str(int(config["miscalibration"]))+"-"+unique_id+".json"
    else:
        weights_path = "../results/weights/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+str(int(config["probability"]*100))+"-"+unique_id+".npy"
        config_path = "../results/configs/"+config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+str(int(config["probability"]*100))+"-"+unique_id+".json"
    config["id"] = unique_id

    np.save(weights_path, trained_weights)

    with open(config_path, mode="w", encoding="utf-8") as file:
        json.dump(config, file, indent = 6)

    print("Weights saved in "+weights_path)