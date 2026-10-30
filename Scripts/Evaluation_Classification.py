"""
6 - VISUALISATION, ÉVALUATION ET COMPARAISON DE LA CLASSIFICATION
Fichier ajouté avec Claude : tableaux, matrices de confusion, graphiques, évaluation sur le test.
La structure d'origine (dictionnaires de métriques train / validation, choix du modèle, métriques sur le test) vient
de l'utilisateur ; elle a été factorisée (ligne_resultats) et complétée (FP / FN, tableaux PNG) avec Claude.
Importer Modeles_Classification lance l'entrainement des modèles.
"""

# %%
import numpy as np
import pandas as pd
from Data import X_train, X_val, X_test, y_train, y_val, y_test
from Modeles_Classification import (
    y_train_bin, y_val_bin, y_test_bin, theta, couts, models, c_gaussien, gamma_values,
    predict_logistique, predict_majo, accuracy, precision, recall, f1,
)
from Outils import tableau_png, graphe_couts, graphe_matrice_confusion

# %%
def faux_positifs(y_pred, y):  # [Claude]
    return int(np.sum((y_pred == 1) & (y == 0)))

def faux_negatifs(y_pred, y):  # [Claude]
    return int(np.sum((y_pred == 0) & (y == 1)))

def ligne_resultats(nom, y_pred_train, y_pred_val):  # [Claude] factorise les dictionnaires de l'original
    return {
        "Modèle": nom,
        "Train Accuracy": accuracy(y_pred_train, y_train_bin),
        "Train Precision": precision(y_pred_train, y_train_bin),
        "Train Recall": recall(y_pred_train, y_train_bin),
        "Train F1": f1(y_pred_train, y_train_bin),
        "Val Accuracy": accuracy(y_pred_val, y_val_bin),
        "Val Precision": precision(y_pred_val, y_val_bin),
        "Val Recall": recall(y_pred_val, y_val_bin),
        "Val F1": f1(y_pred_val, y_val_bin),
        "Val FP": faux_positifs(y_pred_val, y_val_bin),
        "Val FN": faux_negatifs(y_pred_val, y_val_bin),
    }

results = []

y_pred_train_majo = predict_majo(X_train, y_train)
y_pred_val_majo = predict_majo(X_val, y_train)
results.append(ligne_resultats("Classifieur Majoritaire", y_pred_train_majo, y_pred_val_majo))

seuils = [0.3, 0.5, 0.7]
for s in seuils:
    y_pred_train_log = predict_logistique(X_train, theta, s)
    y_pred_val_log = predict_logistique(X_val, theta, s)
    results.append(ligne_resultats(f"Régression Logistique (Seuil = {int(s * 100)}%)", y_pred_train_log, y_pred_val_log))

model_lin = models["SVM Linéaire"]
y_pred_train_lin = model_lin.predict(X_train)
y_pred_val_lin = model_lin.predict(X_val)
results.append(ligne_resultats("SVM Linéaire", y_pred_train_lin, y_pred_val_lin))

for c_val in c_gaussien:
    for g_val in gamma_values:
        nom = f"SVM Gaussien (C = {c_val}, gamma = {g_val})"
        model = models[nom]
        y_pred_train_gauss = model.predict(X_train)
        y_pred_val_gauss = model.predict(X_val)
        results.append(ligne_resultats(nom, y_pred_train_gauss, y_pred_val_gauss))

df_results = pd.DataFrame(results)

# Modèle retenu : choisi avec les 4 critères du tableau, mais surtout Precision / Recall (donc F1),
# car la classe "note >= 7" est minoritaire : une bonne Accuracy s'obtient presque gratuitement en prédisant
# toujours "note < 7" (voir le classifieur majoritaire : Accuracy élevée mais Recall = 0).
# Pour ajuster C et gamma : modifier c_gaussien / gamma_values dans Modeles_Classification.py, puis le nom ici.
nom_choisi = "SVM Gaussien (C = 5, gamma = 1)"

print("\n" + "="*95)
print("                           TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*95)
print(df_results.to_string(index=False))
print("="*95)
print("Meilleur F1 en validation :", df_results.loc[df_results["Val F1"].idxmax(), "Modèle"])

tableau_png(df_results, "Classification - Performances (entrainement / validation) - positif = note ≥ 7", "classification_tableau_validation", decimales=2, surligner=nom_choisi)

# %%
model = models[nom_choisi]

# Validation : matrice de confusion du modèle choisi (sert à ajuster C et gamma)
y_pred_val_choisi = model.predict(X_val)
graphe_matrice_confusion(y_val_bin, y_pred_val_choisi, f"Matrice de confusion (validation) - {nom_choisi}", "classification_matrice_confusion_validation")

# Test : uniquement la référence (classifieur majoritaire) et le modèle choisi
y_pred_test = model.predict(X_test)
y_pred_test_majo = predict_majo(X_test, y_train)

print(f"Modèle choisi : {nom_choisi}")
print("Accuracy Test  :", accuracy(y_pred_test, y_test_bin) * 100, "%")
print("Precision Test :", precision(y_pred_test, y_test_bin) * 100, "%")
print("Recall Test    :", recall(y_pred_test, y_test_bin) * 100, "%")
print("F1-score Test  :", f1(y_pred_test, y_test_bin) * 100, "%")
print("Faux positifs  :", faux_positifs(y_pred_test, y_test_bin))
print("Faux négatifs  :", faux_negatifs(y_pred_test, y_test_bin))

def ligne_test(nom, y_pred):
    return {
        "Modèle": nom,
        "Accuracy": accuracy(y_pred, y_test_bin),
        "Precision": precision(y_pred, y_test_bin),
        "Recall": recall(y_pred, y_test_bin),
        "F1": f1(y_pred, y_test_bin),
        "Faux positifs": faux_positifs(y_pred, y_test_bin),
        "Faux négatifs": faux_negatifs(y_pred, y_test_bin),
    }

df_test = pd.DataFrame([ligne_test("Classifieur Majoritaire", y_pred_test_majo), ligne_test(nom_choisi, y_pred_test)])
tableau_png(df_test, "Classification - Résultats sur le jeu de test", "classification_tableau_test", surligner=nom_choisi)

graphe_matrice_confusion(y_test_bin, y_pred_test, f"Matrice de confusion (test) - {nom_choisi}", "classification_matrice_confusion_test")

# %%
graphe_couts({"Régression logistique": couts}, "Classification - Coût de la régression logistique", "classification_couts", xlabel="Itérations")

#%%
