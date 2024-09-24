import sys
import os

sys.path.append("..")

import numpy as np
import json
import pandas as pd
import torch

from models.qml_models import define_model
from datasets.fetch_data import fetch_dataset, Dataset
from training.lightning_utils import QMLModel, threshold

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]

dataset = sys.argv[1]
qml_model = sys.argv[2]

# Check if attacks have already been performed for dataset and qml_model combo
samples_path = "modified_samples/"+dataset+"/"+qml_model
if os.path.isdir(samples_path):
    items = os.listdir(samples_path)
    if len(items) == 10:
         print("AML attacks have already been performed and can be found in "+samples_path)
         exit()

# Retrieve pre-trained weights and config from results according to dataset and qml_model
results_path = "../../results/"

directory = results_path+"weights/"+dataset+"/"
for filename in os.listdir(directory):
    if dataset in filename and qml_model in filename and "none" in filename:
        weights = torch.tensor(np.load(directory+filename))
        with open(results_path+"configs/"+dataset+"/"+filename[:-3]+"json", "r") as file:
            config = json.load(file)
        break

# Make sure directory structure is in place
os.makedirs(samples_path, exist_ok=True)

# Retrieve preprocessed test dataset according to the config
# TODO: If dataset is mnist, further reduce the test dataset to half bc of validation and test
_, X_test, _, Y_test = fetch_dataset(config["dataset"], path="../datasets")
test_set = Dataset(X_test, Y_test)

qnode, _ = define_model(config)
model = QMLModel(qnode, weights, config)

cols = list(range(X_test.size()[1]))
cols.append("pred")

# Perform attacks with pgd and fgsm and different epsilons
for epsilon in epsilons:
    df_pgd = pd.DataFrame(columns=cols)
    df_fgsm = pd.DataFrame(columns=cols)

    correct = 0
    correct_pgd = 0
    correct_fgsm = 0

    for x, y in test_set:
        x_fgsm = fast_gradient_method(model, x, epsilon, np.inf)
        x_pgd = projected_gradient_descent(model, x, epsilon, 0.01, 40, np.inf)

        # Save x_fgm, x_pgd, and their prediction
        y_pred = threshold(model(x))  # model prediction on clean examples
        y_pred_fgsm = threshold(model(x_fgsm))  # model prediction on FGM adversarial examples
        y_pred_pgd = threshold(model(x_pgd))  # model prediction on PGD adversarial examples

        data = x_pgd.tolist()
        data.append(y_pred_pgd.item())
        df_pgd = pd.concat([pd.DataFrame([data], columns=cols), df_pgd], ignore_index=True)

        data = x_fgsm.tolist()
        data.append(y_pred_fgsm.item())
        df_fgsm = pd.concat([pd.DataFrame([data], columns=cols), df_fgsm], ignore_index=True)

        if y_pred == y:
            correct += 1
                
        if y_pred_fgsm == y:
            correct_fgsm += 1
                
        if y_pred_pgd == y:
            correct_pgd += 1


    print(
        "test acc on clean examples (%): {:.3f}".format(
            correct /  X_test.size()[0] * 100.0
        )
    )
    print(
        "test acc on FGSM adversarial examples (%): {:.3f}".format(
            correct_fgsm / X_test.size()[0] * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            correct_pgd / X_test.size()[0] * 100.0
        )
    )
    
    datafile = dataset+'-'+qml_model+'-'+str(epsilon)
    df_pgd.to_csv(samples_path+'/pgd-'+datafile+'.csv', sep='\t', index=False)
    df_fgsm.to_csv(samples_path+'/fgsm-'+datafile+'.csv', sep='\t', index=False)


