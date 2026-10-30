#%% Libraries et données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC , LinearSVC

df = pd.read_csv("../Data/winequality_white.csv" , sep = ";")

if __name__ == "__main__":

    print(df.head())
    print(df.describe())
    print(df.columns)
    print(df.isna().sum())


#%% Visualisation des données
    n = 0
    for i in df.columns:
        for j in df.columns[n:]: 
            if i != j and i != 'quality' and j != 'quality':
                plt.figure()
                axes = plt.axes(projection="3d")            
                
                axes.set_box_aspect(None, zoom=0.8)
                axes.scatter(df[i], df["quality"], df[j])        
                
                plt.title(f"Qualité en fonction de {i} et {j}", fontsize=14, pad=20)
                axes.set_xlabel(i, fontsize=12, labelpad=18)  
                axes.set_ylabel("Qualité", fontsize=12, labelpad=15)
                axes.set_zlabel(j, fontsize=12, labelpad=15)
                
                plt.tight_layout()
                plt.show()
        n += 1

    
#%% Préparation des données
X = df[df.columns[df.columns != "quality"]]
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15,stratify=y,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.15,stratify = y_train,random_state=42)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# %%