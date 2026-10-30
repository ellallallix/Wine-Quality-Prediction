"""
2 - MODÈLES DE CLASSIFICATION
Fonctions d'entrainement et entrainement des modèles : travail de l'utilisateur.
Les lignes marquées [Claude] sont des corrections ou ajouts faits avec Claude (détail dans le README).
Aucun graphique ni tableau dans ce fichier.
"""

# %%
import numpy as np
from Data import X_train, X_val, X_test, y_train, y_val, y_test, SEED  # [Claude] la séparation est faite dans Data.py (mêmes partitions que la régression)
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import recall_score, f1_score, precision_score

y_train_bin = (y_train >= 7).astype(int)
y_val_bin = (y_val >= 7).astype(int)
y_test_bin = (y_test >= 7).astype(int)

# %%
def score_logistique(X, theta):
    n, _ = np.shape(X)
    X_aug = np.column_stack((X, np.ones((n,))))
    return X_aug @ theta  

def proba(z):
    return 1 / (1 + np.exp(-z))  # [Claude] correction : l'original avait exp(z), incompatible avec le gradient

def cout_logistique(X, y, theta):
    p = proba(score_logistique(X, theta))
    p = np.clip(p, 1e-12, 1 - 1e-12)  # [Claude] évite log(0)
    J = -y * np.log(p) - (1 - y) * np.log(1 - p)  # [Claude] coût adapté à la sigmoïde corrigée
    return np.mean(J)

def gradient_logistique(X, y, theta):
    n, d = np.shape(X)
    X_aug = np.column_stack((X, np.ones((n))))
    z = score_logistique(X, theta)
    p = proba(z)
    dJ = X_aug.T @ (p - y)
    return 1 / n * dJ

def entrainer_logistique(X, y, n_iterations, alpha):
    n, d = np.shape(X)
    X_aug = np.column_stack((X, np.ones((n))))
    theta = np.zeros((d + 1,))
    couts = []

    for i in range(n_iterations):
        theta -= alpha * gradient_logistique(X, y, theta)
        couts.append(cout_logistique(X, y, theta))

    return theta, couts

def predict_logistique(X, theta, s):
    z = score_logistique(X, theta)
    p = proba(z)
    return (p >= s).astype(int)

def predict_majo(X, y):
    n, _ = np.shape(X)
    classes = (y >= 7).astype(int)
    majo = np.argmax(np.bincount(classes))
    return np.array([majo] * n)

# %%
n_iterations = 2000
alpha = 0.1

theta, couts = entrainer_logistique(X_train, y_train_bin, n_iterations, alpha)

models = {}

models["SVM Linéaire"] = LinearSVC(loss='squared_hinge', dual=False, random_state=SEED)  # [Claude] random_state ajouté

c_gaussien = [0.1, 1, 5]
gamma_values = [0.1, 1, 5]

for c_val in c_gaussien:
    for g_val in gamma_values:
        nom = f"SVM Gaussien (C = {c_val}, gamma = {g_val})"
        models[nom] = SVC(C=c_val, kernel="rbf", gamma=g_val)

for nom, model in models.items():
    model.fit(X_train, y_train_bin)

# %%
def accuracy(y_pred, y):
    return np.sum((y_pred == y).astype(int)) / len(y)

def precision(y_pred, y):
    return precision_score(y, y_pred, zero_division=0)

def recall(y_pred, y):
    return recall_score(y, y_pred, zero_division=0)

def f1(y_pred, y):
    return f1_score(y, y_pred, zero_division=0)

#%%
