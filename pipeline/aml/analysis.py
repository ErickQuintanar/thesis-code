import sys
import os

sys.path.append("..")

from datasets.fetch_data import fetch_dataset, Dataset
from models.qml_models import define_model
from training.lightning_utils import QMLModel, threshold

import numpy as np
import pandas as pd
import torch
import json
import matplotlib.pyplot as plt

dataset = sys.argv[1]
qml_model = sys.argv[2]

noise_models = ['none', 'amplitude-damping', 'bit-flip', 'depolarizing', 'phase-damping', 'phase-flip', 'coherent']

aml_attacks = ['fgsm', 'pgd']

probabilities = [0.02, 0.04, 0.06, 0.08, 0.1]

miscalibrations = [2, 4, 6, 8, 10]

epsilons = [0.06, 0.12, 0.18, 0.24, 0.3]

results = []

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
            
def model_accuracy(model, test_set):
    correct = 0
    for x, y in test_set:
        y_pred = threshold(model(x))
        if y_pred == y:
            correct += 1
    acc = correct /  test_set.size()[0] * 100.0
    return acc
                
def adversarial_analysis(noise_model, probability):
    # Retrieve weights and model
    weights, config = retrieve_weights(noise_model, probability)
    qnode, _ = define_model(config)
    model = QMLModel(qnode, weights, config)

    # Calculate accuracy from model on clean test set
    _, X_test, _, Y_test = fetch_dataset(config["dataset"], path="../datasets")
    test_set = Dataset(X_test, Y_test)
    acc = model_accuracy(model, test_set)
    aml_attack = "none"
    epsilon = "0"
    results.append([qml_model, noise_model, probability, aml_attack, epsilon, acc])

    for aml_attack in aml_attacks:
        for epsilon in epsilons:
            # Retrieve modified test set from samples_path
            items = os.listdir(samples_path)
            for attack in items:
                if str(epsilon) in attack and aml_attack in attack:
                    df = pd.read_csv(samples_path+'/'+attack, sep='\t')
                    break
            # Calculate adversarial accuracy from model on modified test set
            X = torch.tensor(df.iloc[:, 0:(df.shape[1]-1)].values, requires_grad=False)
            Y = torch.tensor(df.iloc[:, -1].values, requires_grad=False)
            adversarial_test_set = Dataset(X, Y)
            acc = model_accuracy(model, adversarial_test_set)
            results.append([qml_model, noise_model, probability, aml_attack, epsilon, acc])

def prepare_figure():
    font2 = {'weight':'bold', 'size':12}
    plt.xlabel('Attack Strength Epsilon', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    plt.xlim(left=epsilons[0])
    plt.ylim(bottom=0)
    plt.xticks(epsilons)
    plt.yticks(range(0, 101, 10))
    plt.legend(loc='best')
    plt.tight_layout()

def create_figures(df):
    # Graph Möttönen method bounds and observed data
    font = {'weight':'bold', 'size':15}
    font2 = {'weight':'bold', 'size':12}
    plt.figure(figsize=(5,5))

    incoherent_noise = ['amplitude-damping', 'bit-flip', 'depolarizing', 'phase-damping', 'phase-flip']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    # cols = ["qml_model", "noise_model", "probability|miscalibration", "aml_attack", "epsilon", "accuracy"]
    # TODO: Extract qml_model and dataset for title and fix wrapping issues

    for noise, color in zip(incoherent_noise, colors):
        data = df.loc[(df['noise_model'] == noise) & (df['aml_attack'] == 'none')]
        plt.plot(data['probability|miscalibration'], data['accuracy'], color=color, label=noise)
    
    plt.xlabel('Probabilities', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    plt.title('Accuracy with Different Probabilites for Incoherent Noise Models', fontdict=font)
    plt.legend(loc='best')
    plt.xlim(left=probabilities[0])
    plt.ylim(bottom=0)
    plt.xticks(probabilities)
    plt.yticks(range(0, 101, 10))
    plt.tight_layout()
    plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/accuracy-incoherent.png")

    plt.clf()

    data = df.loc[(df['noise_model'] == 'coherent') & (df['aml_attack'] == 'none')]
    plt.plot(data['probability|miscalibration'], data['accuracy'], color='blue', label='coherent')
    plt.xlabel('Miscalibrations', fontdict=font2)
    plt.ylabel('Accuracy', fontdict=font2)
    plt.title('Accuracy with Different Miscalibrations for Coherent Noise Model', fontdict=font)
    plt.legend(loc='best')
    plt.xlim(left=miscalibrations[0])
    plt.ylim(bottom=0)
    plt.xticks(miscalibrations)
    plt.yticks(range(0, 101, 10))
    plt.tight_layout()
    plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/accuracy-coherent.png")
    plt.clf()

    # TODO: loop over noise_model in noise_models
    for noise_model in noise_models:
        for aml_attack in aml_attacks:
            if noise_model == "none":
                data = df.loc[(df['noise_model'] == noise_model) & (df['aml_attack'] == aml_attack)]
                plt.plot(data['epsilon'], data['accuracy'], color='blue')
                # TODO: determine appropiate title
                plt.title('asdfasdfel', fontdict=font)
                prepare_figure()
                # TODO: Fix figure naming
                plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/"+noise_model+"-"+aml_attack+".png")
                plt.clf()
            elif noise_model == "coherent":
                for miscalibration, color in zip(miscalibrations, colors):
                    data = df.loc[(df['noise_model'] == noise_model) & (df['aml_attack'] == aml_attack) & (df['probability|miscalibration'] == miscalibration)]
                    plt.plot(data['epsilon'], data['accuracy'], color=color, label=miscalibration)
                    # TODO: determine appropiate title
                plt.title('asdfasdfel', fontdict=font)
                prepare_figure()
                # TODO: Fix figure naming
                plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/"+noise_model+"-"+aml_attack+".png")
                plt.clf()
            else:
                for probability, color in zip(probabilities, colors):
                    data = df.loc[(df['noise_model'] == noise_model) & (df['aml_attack'] == aml_attack) & (df['probability|miscalibration'] == probability)]
                    plt.plot(data['epsilon'], data['accuracy'], color=color, label=probability)
                    # TODO: determine appropiate title
                plt.title('asdfasdfel', fontdict=font)
                prepare_figure()
                # TODO: Fix figure naming
                plt.savefig("analysis_results/"+dataset+"/"+qml_model+"/figures/"+noise_model+"-"+aml_attack+".png")
                plt.clf()
    # TODO: loop over aml_attack in aml_attacks
    # TODO: Check special cases for coherent and noiseless models for the labels and no loops respectively

# Check if adversarial analysis is already done
results_path = "analysis_results/"+dataset+"/"+qml_model
os.makedirs(results_path, exist_ok=True)
if os.path.isdir(results_path):
    items = os.listdir(results_path)
    if len(items) == 1 or len(items) == 2:
        print("Analysis has been performed.")
        # TODO: Check if figures have been created and exit, else create them with create_graphs() function
        os.makedirs(results_path+"/figures", exist_ok=True)
        if os.path.isdir(results_path+"/figures"):
            items = os.listdir(results_path+"/figures")
            if len(items) == 16:
                print("Figures have already been created.")
                exit()
            else:
                df = pd.read_csv(results_path+"/"+dataset+".csv", sep='\t')
                create_figures(df)
                exit()
        exit()

# Check if attacks have already been performed for dataset and qml_model combo
samples_path = "modified_samples/"+dataset+"/"+qml_model
os.makedirs(samples_path, exist_ok=True)
if os.path.isdir(samples_path):
    items = os.listdir(samples_path)
    if not (len(items) == 10):
        print("AML attacks haven't been performed.")
        exit()

'''# Retrieve preprocessed test dataset according to the arguments
_, X_test, _, Y_test = fetch_dataset(dataset, path="../datasets")
test_set = Dataset(X_test, Y_test)'''

# For all noise models, load pre-trained weights and measure accuracy for PGD and FGSM attacks for all epsilons
for noise_model in noise_models:
    if noise_model == 'coherent':
        for miscalibration in miscalibrations:
            # Retrieve weights for dataset, qml_model, noise_model, miscalibration combo
            adversarial_analysis(noise_model, miscalibration)
    elif noise_model == 'none':
        # Retrieve weights for dataset, qml_model, noise_model, combo
        adversarial_analysis(noise_model, 0)
    else:
        for probability in probabilities:
            # Retrieve weights for dataset, qml_model, noise_model, probability combo
            adversarial_analysis(noise_model, probability)

print(len(results))

# Save results from adversarial analysis
cols = ["qml_model", "noise_model", "probability|miscalibration", "aml_attack", "epsilon", "accuracy"]
df = pd.DataFrame(results, columns=cols)
df.to_csv(results_path+"/"+dataset+".csv", sep='\t')

print(df.head())

os.makedirs(results_path+"/figures", exist_ok=True)
create_figures(df)
