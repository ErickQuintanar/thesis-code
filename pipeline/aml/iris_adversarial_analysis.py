import sys
sys.path.append("..")

import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from datasets.fetch_data import fetch_dataset

epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]

samples_path = "modified_samples/iris/pqc"

aml_attacks = ["fgsm", "pgd"]

for aml_attack in aml_attacks:
    for epsilon in epsilons:
        # Retrieve modified test set from samples_path
        items = os.listdir(samples_path)
        for attack in items:
            if str(epsilon) in attack and aml_attack in attack:
                df = pd.read_csv(samples_path+'/'+attack, sep='\t')
                break

        # Plot adversarial samples
        # Drop the col that the sign is the same for both classes
        df = df.drop('1', axis=1)
        figure, axis = plt.subplots(1, 3, figsize=(15, 5))
        if 0 in df["pred"].unique():
            colors = ["blue", "orange"]
            l = ['0', '1']
        else:
            colors = ["orange"]
            l = ['1']

        # Get features individually
        df_0 = df[df["pred"]==0]
        df_1 = df[df["pred"]==1]
        x_0 = df_0["0"]
        x_1 = df_1["0"]
        y_0 = df_0["2"]
        y_1 = df_1["2"]
        z_0 = df_0["3"]
        z_1 = df_1["3"]

        # Plot x vs y
        axis[0].scatter(x_0, y_0, c="blue", label='0')
        axis[0].scatter(x_1, y_1, c="orange", label='1')
        axis[0].set_xlabel('Sepal Length')
        axis[0].set_ylabel('Petal Length')
        axis[0].set_xlim([3.5, 9])
        axis[0].set_ylim([0, 10])
        axis[0].set_title("Sepal Length vs. Petal Length")
        axis[0].legend(loc='best')

        # Plot x vs z
        axis[1].scatter(x_0, z_0, c="blue", label='0')
        axis[1].scatter(x_1, z_1, c="orange", label='1')
        axis[1].set_xlabel('Sepal Length')
        axis[1].set_ylabel('Petal Width')
        axis[1].set_xlim([3.5, 9])
        axis[1].set_ylim([-1, 1])
        axis[1].set_title("Sepal Length vs. Petal Width")
        axis[1].legend(loc='best')

        # Plot y vs z
        axis[2].scatter(y_0, z_0, c="blue", label='0')
        axis[2].scatter(y_1, z_1, c="orange", label='1')
        axis[2].set_xlabel('Petal Length')
        axis[2].set_ylabel('Petal Width')
        axis[2].set_xlim([0, 10])
        axis[2].set_ylim([-1, 1])
        axis[2].set_title("Petal Length vs. Petal Width")
        axis[2].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('adversarial-analysis/adversarial-'+aml_attack+'-'+str(epsilon)+'.png')

_, X_test, _, Y_test = fetch_dataset("iris", path="../datasets")

print(Y_test.float().mean().item())

