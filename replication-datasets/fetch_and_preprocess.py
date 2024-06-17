from sklearn.datasets import fetch_openml
from skimage.transform import resize
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd

import os
from PIL import Image

# Fix random generator
np.random.seed(0)

# Retrieve Iris data set
df = fetch_openml(name="iris", version=1, as_frame=True).frame

# Drop rows containing at least NaN feature
df = df.dropna()

# Drop "Iris-versicolor" species
df = df[df["class"] != "Iris-versicolor"]

# Modify class labels to binary classes
df["class"] = df["class"].map({"Iris-setosa" : 0 , "Iris-virginica" : 1}).astype("int32")

# Set "petal_width" feature to 0
df.loc[:, "petallength"] = 0

# Normalize features with L2-norm per data point (maybe eliminate rn)
#df.loc[:,("sepallength","sepalwidth","petallength","petalwidth")] = normalize(df[["sepallength","sepalwidth","petallength","petalwidth"]], norm="l2")

# Save dataset
df.to_csv('iris_preprocessed.txt', sep='\t', index=False)
print("Iris dataset preprocessed and saved.")


# Retrieve Diabetes dataset
df = fetch_openml(name="diabetes", version=1, as_frame=True).frame

# Drop rows containing at least one NaN feature
df = df.dropna()

# Drop duplicate rows
df = df.drop_duplicates()

# Modify class labels to binary classes
df["class"] = df["class"].map({"tested_negative" : 0 , "tested_positive" : 1}).astype("int32")

# Drop random samples from the majority class to make equal distribution
df_pos = df[df["class"] == 1]
df_neg = df[df["class"] == 0]
remove_num = np.abs(df_pos["class"].count() - df_neg["class"].count())
drop_indices = np.random.choice(df_neg.index, remove_num, replace=False)
df_subset = df.drop(drop_indices)

# TODO: change int64 to float dtypes
print(df_subset.dtypes)
features = ("preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age")
df_subset.loc[:,features] = df_subset.astype(np.float64)
print(df_subset.dtypes)

# Standard normalized and rescale between [0;1]
sc = StandardScaler(copy=False)
df_subset.loc[:,features] = sc.fit_transform(df_subset.loc[:,features])

rescaler = MinMaxScaler(copy=False)
df_subset.loc[:,features] = rescaler.fit_transform(df_subset.loc[:,features])

# Save dataset
df_subset.to_csv('diabetes_preprocessed.txt', sep='\t', index=False)
print("Diabetes dataset preprocessed and saved.")


# Retrieve Breast Cancer dataset
df = fetch_openml(name="breast-w", version=1, as_frame=True).frame
print(df.shape)

# Drop rows containing at least one NaN feature
df = df.dropna()
print(df.shape)

# Drop duplicate rows
df = df.drop_duplicates()
print(df.shape)

# Modify class labels to binary classes
df["Class"] = df["Class"].map({"benign" : 0 , "malignant" : 1}).astype("int32")

# TODO: change int64 to float dtypes
# Standard normalized and rescale between [0;1]
features = ('Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 'Marginal_Adhesion', 'Single_Epi_Cell_Size', 'Bare_Nuclei',
       'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses')

sc = StandardScaler(copy=False)
df.loc[:,features] = sc.fit_transform(df.loc[:,features])

rescaler = MinMaxScaler(copy=False)
df.loc[:,features] = rescaler.fit_transform(df.loc[:,features])

# Drop rows with all zeros
df = df.loc[(df!=0).any(axis=1)]

# Save dataset
df.to_csv('breast_cancer_preprocessed.txt', sep='\t', index=False)
print("Breast Cancer dataset preprocessed and saved.")


# Retrieve MNIST dataset
df = fetch_openml(name="mnist_784", version=1, as_frame=True).frame

# Drop rows containing at least one NaN feature
df.dropna()

# Reshape, resize and flatten
X = df.drop(columns='class').values
X_reshaped = X.reshape(-1, 28, 28)
X_resized = np.array([resize(image, (16, 16), preserve_range=True) for image in X_reshaped])
X_resized_flat = X_resized.reshape(-1, 16*16)

# TODO: Standard normalize
# Rescale features between [0,1]
rescaler = MinMaxScaler(copy=False)
X_resized_flat_rescaled = rescaler.fit_transform(X_resized_flat)

# Collect resized, flattened, and rescaled images with target labels
df_db = pd.DataFrame(X_resized_flat_rescaled, columns=range(256))
df_db = df_db.assign(target=df["class"])

# Save dataset
df_db.to_csv('mnist10_preprocessed.txt', sep='\t', index=False)
print("MNIST10 dataset preprocessed and saved.")

df_db = pd.read_csv('mnist10_preprocessed.txt', sep='\t')

# Remove all classes but 1, 3, 7, and 9
df_db = df_db.loc[((df_db["target"] == 1) | (df_db["target"] == 3) | (df_db["target"] == 7) | (df_db["target"] == 9)),:]

# Modify class labels to binary classes
df_db["target"] = df_db["target"].map({1 : 0 , 3 : 1, 7 : 2, 9 : 3}).astype("int32")

# Save dataset
df_db.to_csv('mnist4_preprocessed.txt', sep='\t', index=False)
print("MNIST4 dataset preprocessed and saved.")

# Remove all classes but 1 and 9
df_db = df_db.loc[((df_db["target"] == 1) | (df_db["target"] == 9)),:]

# Modify class labels to binary classes
df_db["target"] = df_db["target"].map({1 : 0 , 9 : 1}).astype("int32")

# Save dataset
df_db.to_csv('mnist2_preprocessed.txt', sep='\t', index=False)
print("MNIST2 dataset preprocessed and saved.")

# Extract Plus-Minus dataset
rows = []
for root, _, files in os.walk("pm_data"):
	for file in files:
		if file.lower().endswith('.png'):
			image_path = os.path.join(root, file)

			# Retrieve flattened image information
			with Image.open(image_path) as img:
				pixel_data = list(img.getdata())
				label = image_path.split("/")[2][-1:]
				rows.append(pixel_data + [int(label)])


# Create a DataFrame from the rows
features = [f'pixel_{i}' for i in range(len(rows[0]) - 1)]
columns = features + ['class']
df = pd.DataFrame(rows, columns=columns)

# Standard normalized and rescale between [0;1]
sc = StandardScaler(copy=False)
df.loc[:,features] = sc.fit_transform(df.loc[:,features])

rescaler = MinMaxScaler(copy=False)
df.loc[:,features] = rescaler.fit_transform(df.loc[:,features])

# Save dataset
df.to_csv("plus-minus_preprocessed.txt", sep='\t', index=False)
print("Plus-Minus dataset preprocessed and saved.")