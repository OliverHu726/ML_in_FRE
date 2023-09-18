print("Question 1: Feature `2ndFlrSF` shows the largest decrease in Mean Absolute Percentage Error in the plot when it is introduced to the mode.")
print("Question 2: The answer is no.\n  1. Usually when we buid a model, the more features we introduce, the better quality the model can achive. As in this case, we are adding features one by one, so we will probably observe the drop within the first 3 models regarding what features they are. This is the main reason.")
print("  2. However, the model quality improvement also depends on the quality of the feature we added. For example, if 2 features are highly correlated, then they will probably have similar impact on the model. This is tosay, if we first introdu 1 feature to build a model and then add another which is highly correlated with the previous one, we will probably observe a small drop rather than a significant one.")
print("Therefore, the answer is no.")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/OliverHu726/ML_in_FRE/main/Homework_1/train.csv')
features = [
    "1stFlrSF",
    "2ndFlrSF",
    "TotalBsmtSF",
    "LotArea",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
]

list_features = []
r2_scores = []
mse_scores = []
mae_scores = []
mape_scores = []

def LinearPredict(X, y):
  model = LinearRegression()
  model.fit(X, y)
  y_pred = model.predict(X)
  return y_pred

for i in range(1, len(features) + 1):
    selected_features = features[:i]
    X = df[selected_features].values
    y = df['SalePrice'].values
    y_pred = LinearPredict(X,y)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    list_features.append(str(i) + '#' + ' ' + features[i-1])
    r2_scores.append(r2)
    mse_scores.append(mse)
    mae_scores.append(mae)
    mape_scores.append(mape)



plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(list_features, r2_scores, marker='o')
plt.xlabel("Features")
plt.ylabel("R-squared")
plt.xticks(list_features, rotation = 30)
plt.title("R-squared vs. Features")

plt.subplot(2, 2, 2)
plt.plot(list_features, mse_scores, marker='o', color='orange')
plt.xlabel("Features")
plt.ylabel("Mean Squared Error")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Squared Error vs. Features")

plt.subplot(2, 2, 3)
plt.plot(list_features, mae_scores, marker='o', color='green')
plt.xlabel("Features")
plt.ylabel("Mean Absolute Error")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Absolute Error vs. Features")

plt.subplot(2, 2, 4)
plt.plot(list_features, mape_scores, marker='o', color='red')
plt.xlabel("Features")
plt.ylabel("Mean Absolute Percentage Error (%)")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Absolute Percentage Error vs. Features")

plt.tight_layout()
plt.show()



features = [
    "GarageArea",
    "GarageCars",
    "GrLivArea",
    "OverallQual",
    "LotArea",
    "TotalBsmtSF",
    "2ndFlrSF",
    "1stFlrSF",
]

list_features = []
r2_scores = []
mse_scores = []
mae_scores = []
mape_scores = []

for i in range(1, len(features) + 1):
    selected_features = features[:i]
    X = df[selected_features].values
    y = df['SalePrice'].values
    y_pred = LinearPredict(X,y)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    list_features.append(str(i) + '#' + ' ' + features[i-1])
    r2_scores.append(r2)
    mse_scores.append(mse)
    mae_scores.append(mae)
    mape_scores.append(mape)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(list_features, r2_scores, marker='o')
plt.xlabel("Reordered Features")
plt.ylabel("R-squared")
plt.xticks(list_features, rotation = 30)
plt.title("R-squared vs. Reordered Features")

plt.subplot(2, 2, 2)
plt.plot(list_features, mse_scores, marker='o', color='orange')
plt.xlabel("Reordered Features")
plt.ylabel("Mean Squared Error")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Squared Error vs. Reordered Features")

plt.subplot(2, 2, 3)
plt.plot(list_features, mae_scores, marker='o', color='green')
plt.xlabel("Reordered Features")
plt.ylabel("Mean Absolute Error")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Absolute Error vs. Reordered Features")

plt.subplot(2, 2, 4)
plt.plot(list_features, mape_scores, marker='o', color='red')
plt.xlabel("Reordered Features")
plt.ylabel("Mean Absolute Percentage Error (%)")
plt.xticks(list_features, rotation = 30)
plt.title("Mean Absolute Percentage Error vs. Reordered Features")

plt.tight_layout()
plt.show()



selected_df = df[features]
correlation_matrix = selected_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.title('Heatmap for selected Features')
plt.show()
