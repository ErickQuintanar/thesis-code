import sys
import os

sys.path.append("..")

import numpy as np
import json
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from models.qml_models import define_model
from datasets.fetch_data import fetch_dataset, Dataset
from training.lightning_utils import QMLModel, threshold

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# TODO: Check if ranges are really correct
epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
dataset = "iris"
qml_model = "pqc"

# Retrieve pre-trained weights and config from results according to dataset and qml_model
results_path = "../../results/"
directory = results_path+"weights/"+dataset+"/"
for filename in os.listdir(directory):
    if dataset in filename and qml_model in filename and "none" in filename:
        weights = torch.tensor(np.load(directory+filename))
        with open(results_path+"configs/"+dataset+"/"+filename[:-3]+"json", "r") as file:
            config = json.load(file)
        break

# Retrieve preprocessed test dataset according to the config
_, X_test, _, Y_test = fetch_dataset(config["dataset"], path="../datasets")
test_set = Dataset(X_test, Y_test)
print(test_set)

qnode, _ = define_model(config)
model = QMLModel(qnode, weights, config)

cols = ["Sepal Length (cm)", "sepalwidth", "Petal Length (cm)", "petalwidth", "attack", "epsilon", "Label"]

samples_path = "tabular_data/"+dataset+"/"+qml_model

df = pd.DataFrame(columns=cols)

x = X_test[2]
y = Y_test[2]

data = x.tolist()
data.append("none")
data.append(0)
data.append(y.item())
df = pd.concat([pd.DataFrame([data], columns=cols), df], ignore_index=True)

# Perform attacks with pgd and fgsm and different epsilons
for epsilon in epsilons:

    x_fgsm = fast_gradient_method(model, x, epsilon, np.inf)

    # Save x_fgm, x_pgd, and their prediction
    y_pred = threshold(model(x))  # model prediction on clean examples
    y_pred_fgsm = threshold(model(x_fgsm))  # model prediction on FGM adversarial examples

    data = x_fgsm.tolist()
    data.append("fgsm")
    data.append(epsilon)
    data.append(y_pred_fgsm.item())
    df = pd.concat([pd.DataFrame([data], columns=cols), df], ignore_index=True)

plot = sns.relplot(data=df, x='Sepal Length (cm)', y='Petal Length (cm)', hue='Label')
plot.figure.suptitle('Petal Length vs. Sepal Length (FGSM)')
plt.plot([3.6, 4.6], [1.4, 2.1], linewidth=2, color='r')
labels = [0] + epsilons 
labels.reverse()
for i, eps in enumerate(labels):
    plt.annotate(eps, (df.iloc[i,0], df.iloc[i,2]))
ax = plt.gca()
ax.set_ylim([None, 2.6])
plt.savefig(samples_path+'/tabular-adversarial.png')
datafile = dataset+'-'+qml_model
df.to_csv(samples_path+'/'+datafile+'.csv', sep='\t', index=False)