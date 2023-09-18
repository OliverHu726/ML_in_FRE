import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv('https://raw.githubusercontent.com/OliverHu726/ML_in_FRE/main/Homework_1/train.csv')

X = df[['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']].values
y = df['SalePrice'].values
X_transpose = np.transpose(X)
beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
y_hat = X.dot(beta)
mean_y = np.mean(y)
ss_tot = np.sum((y - mean_y)**2)
ss_res = np.sum((y - y_hat)**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared value: {r_squared}")