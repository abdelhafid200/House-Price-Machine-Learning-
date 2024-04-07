# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# %%
train_df = pd.read_csv('dataset/train.csv')
test_df= pd.read_csv('dataset/test.csv')
train_df.head()

# %%
train_df= train_df.drop('Id',axis=1)

# %%
print(train_df.shape)

# %%
train_df.info()

# %%
train_df.describe()


# %%
columns = train_df.columns
columns


df = train_df
df

# %%
interested_features = ['SalePrice','1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','GrLivArea','GarageArea', 'TotalBsmtSF', 'FullBath', 'HalfBath','BsmtHalfBath','BsmtFullBath','BedroomAbvGr']


# %%
train_df = train_df[interested_features]
test_df = test_df[interested_features[1:]]


# %%
train_df.head()

# %%
test_df.head()

# %%
correlation_matrix = train_df.corr()
plt.figure(figsize=(20,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")


# %%
fig, ax = plt.subplots(4,3, figsize=(20,20))
for i, ax in enumerate(ax.flat):
    if i < len(interested_features):
        sns.histplot(train_df[interested_features[i]], ax=ax)
        ax.set_title(interested_features[i])

# %%
for col in num_columns.index:
    plt.figure()
    plt.title(col)
    sns.scatterplot(x=col, y='SalePrice', data=train_df)

# %%
train_df['totalsf'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
test_df['totalsf'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']

train_df['totalarea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalarea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']

train_df['totalbaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] +  (train_df['BsmtHalfBath'] + train_df['HalfBath']) 
test_df['totalbaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] +  (test_df['BsmtHalfBath'] + test_df['HalfBath']) 

# %%
train_df.head()

# %%
test_df.head()

# %%
train_df = train_df.drop(columns=['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','GrLivArea','TotalBsmtSF', 'FullBath', 'HalfBath','BsmtHalfBath','BsmtFullBath'])
test_df = test_df.drop(columns=['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','GrLivArea','TotalBsmtSF', 'FullBath', 'HalfBath','BsmtHalfBath','BsmtFullBath'])

# %%
train_df.head()

# %%
test_df.head()

# %%
train_df.info()

# %%
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()


model.fit(X_train, y_train)

# %%
predictions = model.predict(X_test)

# %% [markdown]
# # Calculate Mean Squared Error

# %%
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# %%
# calcul the RMSE and the R2
from sklearn.metrics import r2_score
from math import sqrt

rmse = sqrt(mse)
r2 = r2_score(y_test, predictions)

print("RMSE : ", rmse)
print("R2 : ", r2)

# %%
plt.scatter(y_test, predictions)

# %%
# Supposons que vous avez les valeurs des caractéristiques pour une nouvelle maison
new_house_features = [[1000, 3, 2000, 2500, 2]]  # Exemple de valeurs pour les caractéristiques

# Utilisez la méthode predict pour prédire le prix de la nouvelle maison
predicted_price = model.predict(new_house_features)

print("Prix prédit de la maison :", predicted_price)


# %%
import pickle

# Sauvegarder le modèle
with open('modele.pkl', 'wb') as f:
    pickle.dump(model, f)





# %%



