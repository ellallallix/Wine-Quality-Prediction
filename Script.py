#%% Libraries et données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("winequality_white.csv" , sep = ";")
print(df.head())
print(df.describe())
print(df.columns)


#%% Visualisation des données

n = 0
for i in df.columns:
    for j in df.columns[n:]: 
        if i != j and i != 'quality' and j != 'quality':
            plt.figure(figsize = (10,10))
            axes = plt.axes(projection="3d")            
            
            axes.set_box_aspect(None, zoom=0.8)
            axes.scatter(df[i], df["quality"], df[j])        
            
            plt.title(f"Qualité en fonction de {i} et {j}", fontsize=1, pad=20)
            axes.set_xlabel(i, fontsize=12, labelpad=18)  
            axes.set_ylabel("Qualité", fontsize=12, labelpad=15)
            axes.set_zlabel(j, fontsize=12, labelpad=15)
            
            plt.tight_layout()
            plt.show()
    n += 1

    
#%%
