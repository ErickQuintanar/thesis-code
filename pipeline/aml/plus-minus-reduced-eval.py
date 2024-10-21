import sys
import os

sys.path.append("..")

import pandas as pd
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from datasets.fetch_data import fetch_dataset
from models.qml_models import define_model
from training.lightning_utils import QMLModel, threshold
from textwrap import wrap

dataset = "plus-minus"
qml_model = "pqc"

noise_models = ['none', 'amplitude-damping', 'bit-flip', 'depolarizing', 'phase-damping', 'phase-flip', 'coherent']

probabilities = [0.02, 0.04, 0.06, 0.08, 0.1]

miscalibrations = [2, 4, 6, 8, 10]

aml_attacks = ['fgsm', 'pgd']

epsilons = [0.04, 0.08, 0.12, 0.16, 0.20]

results = []

results_path = "analysis_results/"+dataset+"/"+qml_model
os.makedirs(results_path, exist_ok=True)

def retrieve_weights(noise_model, probability):
    # Retrieve pre-trained weights and config from results according to dataset and qml_model
    results_path = "../../results/"
    directory = results_path+"weights/"+dataset+"/"

    if not (noise_model == "none" or noise_model == "coherent"):
        probability = int(probability * 100)

    # Deal with the different types of noise models and variations
    for filename in os.listdir(directory):
        if noise_model == "none":
            if dataset in filename and qml_model in filename and noise_model in filename:
                weights = torch.tensor(np.load(directory+filename))
                with open(results_path+"configs/"+dataset+"/"+filename[:-3]+"json", "r") as file:
                    config = json.load(file)
                return weights, config
        else:
            if dataset in filename and qml_model in filename and noise_model in filename and ("-"+str(probability)+"-") in filename:
                weights = torch.tensor(np.load(directory+filename))
                with open(results_path+"configs/"+dataset+"/"+filename[:-3]+"json", "r") as file:
                    config = json.load(file)
                return weights, config
            
def model_accuracy(model, X_test, Y_test):
    y_pred = threshold(model(X_test))
    res = torch.logical_not(torch.logical_xor(y_pred, Y_test))
    acc = res.float().mean().item()
    return acc * 100
            
def analysis(noise_model, probability, X_test, Y_test):
    # Retrieve weights and model
    weights, config = retrieve_weights(noise_model, probability)

    # Modify config file such that the evaluating model doesn't have any noise
    config["noise_model"] = "none"
    qnode, _ = define_model(config)
    model = QMLModel(qnode, weights, config)

    # Calculate accuracy from model on clean test set
    acc = model_accuracy(model, X_test, Y_test)
    aml_attack = "none"
    epsilon = "0"
    results.append([qml_model, noise_model, probability, aml_attack, epsilon, acc])
    
def prepare_figure(legend):
    font2 = {'weight':'bold', 'size':12}
    plt.xlabel('Attack Strength Epsilon', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=102)
    plt.xticks(epsilons)
    plt.yticks(range(0, 101, 10))
    if legend:
        plt.legend(loc='best')
    plt.tight_layout()

def create_figures(df):
    font = {'weight':'bold', 'size':15}
    font2 = {'weight':'bold', 'size':12}
    plt.figure(figsize=(5,5))

    incoherent_noise = ['amplitude-damping', 'bit-flip', 'depolarizing', 'phase-damping', 'phase-flip']
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Retrieve baseline performance of the QML model
    baseline = df.loc[(df['noise_model'] == "none") & (df['aml_attack'] == 'none')]
    for noise, color in zip(incoherent_noise, colors):
        data = df.loc[(df['noise_model'] == noise) & (df['aml_attack'] == 'none')]
        # Add baseline noiseless probability
        data = pd.concat([baseline, data])
        data.sort_values("epsilon", inplace=True)
        print(data)
        plt.plot(data['probability|miscalibration'], data['accuracy'], color=color, label=noise, alpha=0.7)
    
    plt.xlabel('Probabilities', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    title = 'Accuracy with different probabilites for incoherent noise models'
    plt.title('\n'.join(wrap(title, 38)), fontdict=font)
    plt.legend(loc='best')
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=102)
    plt.xticks(probabilities)
    plt.yticks(range(0, 101, 10))
    plt.tight_layout()
    plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/accuracy-incoherent.png")

    plt.clf()

    data = df.loc[(df['noise_model'] == 'coherent') & (df['aml_attack'] == 'none')]
    # Add baseline noiseless probability
    data = pd.concat([baseline, data])
    data.sort_values("epsilon", inplace=True)
    print(data)
    plt.plot(data['probability|miscalibration'], data['accuracy'], color='blue', label='coherent', alpha=0.7)
    plt.xlabel('Miscalibrations', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    title = 'Accuracy with different miscalibrations for coherent noise model'
    plt.title('\n'.join(wrap(title, 38)), fontdict=font)
    plt.legend(loc='best')
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=102)
    plt.xticks(miscalibrations)
    plt.yticks(range(0, 101, 10))
    plt.tight_layout()
    plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/accuracy-coherent.png")
    plt.clf()

    noise_model = "none"
    for aml_attack in aml_attacks:
        baseline = df.loc[(df['noise_model'] == noise_model) & (df['aml_attack'] == 'none')]
        data = df.loc[(df['noise_model'] == noise_model) & (df['aml_attack'] == aml_attack)]
        # Add baseline noiseless probability
        data = pd.concat([baseline, data])
        data.sort_values("epsilon", inplace=True)
        print(data)
        plt.plot(data['epsilon'], data['accuracy'], color='blue', alpha=0.7)
        title = "Adversarial accuracy with "+(aml_attack.upper())
        plt.title('\n'.join(wrap(title, 37)), fontdict=font)
        prepare_figure(False)
        plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/"+noise_model+"-"+aml_attack+".png")
        plt.clf()

# Retrieve test set
_, X_test, _, Y_test = fetch_dataset(dataset, path="../datasets")

# For all noise models, load pre-trained weights and measure accuracy on clean test set
for noise_model in noise_models:
    if noise_model == 'coherent':
        for miscalibration in miscalibrations:
            # Retrieve weights for dataset, qml_model, noise_model, miscalibration combo
            analysis(noise_model, miscalibration, X_test, Y_test)
    elif noise_model == 'none':
        # Retrieve weights for dataset, qml_model, noise_model, combo
        analysis(noise_model, 0, X_test, Y_test)
    else:
        for probability in probabilities:
            # Retrieve weights for dataset, qml_model, noise_model, probability combo
            analysis(noise_model, probability, X_test, Y_test)

print(len(results))

# Save results from adversarial analysis
results.append(["pqc",	"none",	"0.0",	"none",	"0",	"75.000"])
results.append(["pqc",	"none",	"0.0",	"fgsm",	"0.04",	"47.778"])
results.append(["pqc",	"none",	"0.0",	"fgsm",	"0.08",	"9.444"])
results.append(["pqc",	"none",	"0.0",	"fgsm",	"0.12",	"2.778"])
results.append(["pqc",	"none",	"0.0",	"fgsm",	"0.16",	"2.500"])
results.append(["pqc",	"none",	"0.0",	"fgsm",	"0.20",	"1.667"])
results.append(["pqc",	"none",	"0.0",	"pgd",	"0.04",	"46.111"])
results.append(["pqc",	"none",	"0.0",	"pgd",	"0.08",	"8.056"])
results.append(["pqc",	"none",	"0.0",	"pgd",	"0.12",	"1.389"])
results.append(["pqc",	"none",	"0.0",	"pgd",	"0.16",	"1.389"])
results.append(["pqc",	"none",	"0.0",	"pgd",	"0.20",	"2.222"])

cols = ["qml_model", "noise_model", "probability|miscalibration", "aml_attack", "epsilon", "accuracy"]

df = pd.DataFrame(results, columns=cols)
df.to_csv(results_path+"/"+dataset+".csv", sep='\t')

os.makedirs(results_path+"/figures", exist_ok=True)
# Retrieve DF from file
df = pd.read_csv(results_path+"/"+dataset+".csv", sep='\t')
create_figures(df)