import sys
import json

dataset = sys.argv[1]

qml_models = ['pqc', 'kernel']

noise_models = ['amplitude-damping', 'bit-flip', 'depolarizing', 'phase-damping', 'phase-flip', 'coherent']

probabilities = [0.02, 0.04, 0.06, 0.08, 0.1]

def save_config(path, config):
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(config, file, indent = 6)

# TODO: Save base configs for plus-minus model

iris_base = {
    "dataset" : "iris",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 2,
    "num_layers" : 2,
    "num_classes" : 2,
    "learning_rate" : 0.05,
    "batch_size" : 30,
    "epochs" : 10
}

breast_cancer_base = {
    "dataset" : "breast-cancer",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 4,
    "num_layers" : 40,
    "num_classes" : 2,
    "learning_rate" : 0.0005,
    "batch_size" : 16,
    "epochs" : 100
}

diabetes_base = {
    "dataset" : "diabetes",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 3,
    "num_layers" : 40,
    "num_classes" : 2,
    "learning_rate" : 0.005,
    "batch_size" : 30,
    "epochs" : 10
}

mnist2_base = {
    "dataset" : "mnist2",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 8,
    "num_layers" : 40,
    "num_classes" : 2,
    "learning_rate" : 0.005,
    "batch_size" : 512,
    "epochs" : 30
}

mnist4_base = {
    "dataset" : "mnist4",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 8,
    "num_layers" : 40,
    "num_classes" : 4,
    "learning_rate" : 0.005,
    "batch_size" : 512,
    "epochs" : 30
}

mnist10_base = {
    "dataset" : "mnist10",
    "qml_model" : "pqc",
    "noise_model" : "none",
    "num_qubits" : 8,
    "num_layers" : 40,
    "num_classes" : 10,
    "learning_rate" : 0.005,
    "batch_size" : 512,
    "epochs" : 30
}

if dataset == "iris":
    config = iris_base
elif dataset == "breast-cancer":
    config = breast_cancer_base
elif dataset == "diabetes":
    config = diabetes_base
elif dataset == "mnist2":
    config = mnist2_base
elif dataset == "mnist4":
    config = mnist4_base
elif dataset == "mnist10":
    config = mnist10_base
else:
    print("Dataset is currently not supported.")
    sys.exit(1)

for qml_model in qml_models:
    config["qml_model"] = qml_model
    # Save base config
    config_path = config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+".json"
    save_config(config_path, config)

for qml_model in qml_models:
    config["qml_model"] = qml_model
    for noise_model in noise_models:
        config["noise_model"] = noise_model
        for probability in probabilities:
            config["probability"] = probability
            config_path = config["dataset"]+"/"+config["dataset"]+"-"+config["qml_model"]+"-"+config["noise_model"]+"-"+str(int(config["probability"]*100))+".json"
            save_config(config_path, config)
