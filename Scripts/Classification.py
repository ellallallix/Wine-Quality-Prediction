# %%
import numpy as np
import pandas as pd
from Data import X, y 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import recall_score, f1_score, precision_score


y_bin = (y >= 7).astype(int)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y_bin, random_state=42
)

y_train_val_bin = (y_train_val >= 7).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, stratify=y_train_val_bin, random_state=42
)

y_train_bin = (y_train >= 7).astype(int)
y_val_bin = (y_val >= 7).astype(int)
y_test_bin = (y_test >= 7).astype(int)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# %%
def score_logistique(X, theta):
    n, _ = np.shape(X)
    X_aug = np.column_stack((X, np.ones((n,))))
    return X_aug @ theta  

def proba(z):
    return 1 / (1 + np.exp(z))

def cout_logistique(X, y, theta):
    p = proba(score_logistique(X, theta))
    J = y * np.log((1 - p) / p) - np.log(1 - p)
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

models["SVM Linéaire"] = LinearSVC(loss='squared_hinge', dual=False)

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

results = []

y_pred_train_majo = predict_majo(X_train, y_train)
y_pred_val_majo = predict_majo(X_val, y_train)
results.append({
    "Modèle": "Classifieur Majoritaire",
    "Train Accuracy": accuracy(y_pred_train_majo, y_train_bin),
    "Train Precision": precision(y_pred_train_majo, y_train_bin),
    "Train Recall": recall(y_pred_train_majo, y_train_bin),
    "Train F1": f1(y_pred_train_majo, y_train_bin),
    "Val Accuracy": accuracy(y_pred_val_majo, y_val_bin),
    "Val Precision": precision(y_pred_val_majo, y_val_bin),
    "Val Recall": recall(y_pred_val_majo, y_val_bin),
    "Val F1": f1(y_pred_val_majo, y_val_bin)
})

seuils = [0.3, 0.5, 0.7]
for s in seuils:
    y_pred_train_log = predict_logistique(X_train, theta, s)
    y_pred_val_log = predict_logistique(X_val, theta, s)
    results.append({
        "Modèle": f"Régression Logistique (Seuil = {int(s * 100)}%)",
        "Train Accuracy": accuracy(y_pred_train_log, y_train_bin),
        "Train Precision": precision(y_pred_train_log, y_train_bin),
        "Train Recall": recall(y_pred_train_log, y_train_bin),
        "Train F1": f1(y_pred_train_log, y_train_bin),
        "Val Accuracy": accuracy(y_pred_val_log, y_val_bin),
        "Val Precision": precision(y_pred_val_log, y_val_bin),
        "Val Recall": recall(y_pred_val_log, y_val_bin),
        "Val F1": f1(y_pred_val_log, y_val_bin)
    })

model_lin = models["SVM Linéaire"]
y_pred_train_lin = model_lin.predict(X_train)
y_pred_val_lin = model_lin.predict(X_val)
results.append({
    "Modèle": "SVM Linéaire",
    "Train Accuracy": accuracy(y_pred_train_lin, y_train_bin),
    "Train Precision": precision(y_pred_train_lin, y_train_bin),
    "Train Recall": recall(y_pred_train_lin, y_train_bin),
    "Train F1": f1(y_pred_train_lin, y_train_bin),
    "Val Accuracy": accuracy(y_pred_val_lin, y_val_bin),
    "Val Precision": precision(y_pred_val_lin, y_val_bin),
    "Val Recall": recall(y_pred_val_lin, y_val_bin),
    "Val F1": f1(y_pred_val_lin, y_val_bin)
})

for c_val in c_gaussien:
    for g_val in gamma_values:
        nom = f"SVM Gaussien (C = {c_val}, gamma = {g_val})"
        model = models[nom]
        y_pred_train_gauss = model.predict(X_train)
        y_pred_val_gauss = model.predict(X_val)
        results.append({
            "Modèle": nom,
            "Train Accuracy": accuracy(y_pred_train_gauss, y_train_bin),
            "Train Precision": precision(y_pred_train_gauss, y_train_bin),
            "Train Recall": recall(y_pred_train_gauss, y_train_bin),
            "Train F1": f1(y_pred_train_gauss, y_train_bin),
            "Val Accuracy": accuracy(y_pred_val_gauss, y_val_bin),
            "Val Precision": precision(y_pred_val_gauss, y_val_bin),
            "Val Recall": recall(y_pred_val_gauss, y_val_bin),
            "Val F1": f1(y_pred_val_gauss, y_val_bin)
        })

df_results = pd.DataFrame(results)

print("\n" + "="*95)
print("                           TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*95)
print(df_results.to_string(index=False))
print("="*95)

# %%
model = models["SVM Gaussien (C = 5, gamma = 1)"]
y_pred_test = model.predict(X_test)

print("Modèle choisi : SVM à noyau Gaussien (C = 5 , gamma = 1)")
print("Accuracy Test  :", accuracy(y_pred_test, y_test_bin) * 100, "%")
print("Precision Test :", precision(y_pred_test, y_test_bin) * 100, "%")
print("Recall Test    :", recall(y_pred_test, y_test_bin) * 100, "%")
print("F1-score Test  :", f1(y_pred_test, y_test_bin) * 100, "%")

# %%
plt.plot(np.arange(n_iterations), couts)
plt.title("Couts Logistique")
plt.show()

#%%