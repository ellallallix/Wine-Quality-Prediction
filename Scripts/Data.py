"""
3 - DATA
Chargement, séparation et standardisation des données : travail de l'utilisateur.
Les lignes marquées [Claude] sont des corrections ou ajouts faits avec Claude (détail dans le README).
"""

#%% Libraries et données

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC , LinearSVC
from Outils import graphe_distribution_notes, graphe_correlations, graphe_caracteristiques

SEED = 42  # [Claude] graine commune à tous les fichiers (partitions, SGD, lots)

# [Claude] Chemin relatif à ce fichier : fonctionne quel que soit le dossier de lancement
df = pd.read_csv(Path(__file__).resolve().parent.parent / "Data" / "winequality_white.csv" , sep = ";")

# [Claude] Suppression des doublons. Le CSV contient des lignes totalement identiques. Si on les garde, une même observation peut
# se retrouver à la fois en entrainement et en test : l'évaluation serait trop favorable.
# Choix : on supprime les doublons AVANT de séparer les données.
n_doublons = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

if __name__ == "__main__":

    print(df.head())
    print(df.describe())
    print(df.columns)
    print(df.isna().sum())
    print("Lignes en double supprimées :", n_doublons, "| Lignes restantes :", len(df))


#%% Visualisation des données
    # [Claude] Cette cellule remplace les graphiques 3D de l'original par 3 graphiques (fonctions dans Outils.py)
    # 3 graphiques : ils servent à justifier le protocole (classes déséquilibrées, standardisation, choix des métriques)
    y_bin_df = (df["quality"] >= 7).astype(int)

    graphe_distribution_notes(df["quality"].to_numpy() , "Distribution des notes de qualité" , "donnees_distribution_notes")
    graphe_correlations(df , "Corrélations entre les variables" , "donnees_correlations")
    graphe_caracteristiques(df , y_bin_df , "Caractéristiques selon la classe (note < 7 ou ≥ 7)" , "donnees_caracteristiques")

    
#%% Préparation des données
X = df[df.columns[df.columns != "quality"]]
y = df["quality"]

# [Claude] Partition 70 % / 15 % / 15 % (la 2e séparation prend 15/85 du reste pour obtenir 15 % du total)
# Ces partitions sont utilisées par Régression.py ET Classification.py
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15,stratify=y,random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.15/0.85,stratify = y_train,random_state=SEED)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# [Claude] Les notes en tableaux numpy (plus simple à manipuler dans les autres fichiers)
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

# %%
