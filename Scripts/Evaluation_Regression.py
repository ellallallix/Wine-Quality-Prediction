"""
5 - VISUALISATION, ÉVALUATION ET COMPARAISON DE LA RÉGRESSION
Fichier ajouté avec Claude : tableaux, graphiques, évaluation sur le test, choix de r.
Seule la fonction MSE (et la structure "prédire sur train / validation / test") vient de l'utilisateur.
Importer Modeles_Regression lance l'entrainement des modèles.
"""

#%% Libraries et données

import numpy as np
import pandas as pd
from Data import X_train,X_test,X_val,y_train,y_test,y_val
from Modeles_Regression import (
    predict_moyenne, predict_lineaire, predict_noyau_polynomiale, entrainer_noyau_polynomiale,
    Theta_lineaire, couts_lineaire, Beta_d2, couts_d2, Beta_d3, couts_d3, r,
    p, eps, alpha_C, alpha_S, alpha_B,
    coutsC, coutsS, coutsB, tempsC, tempsS, tempsB,
    n_iterationsC, n_iterationsS, n_iterationsB,
    elapsed_timeC, elapsed_timeS, elapsed_timeB, epochsC, epochsS, epochsB,
)
from Outils import tableau_png, graphe_reel_vs_predit, graphe_erreurs, graphe_couts, graphe_optimisation

#%% Régression - Validation

def MSE(y_pred,y):  # fonction de l'utilisateur
    e = y_pred-y
    return np.mean(e**2)

def MAE(y_pred,y):  # [Claude]
    return np.mean(np.abs(y_pred-y))

# Moyenne
y_pred_train_moyenne = predict_moyenne(y_train,X_train)
y_pred_val_moyenne = predict_moyenne(y_train,X_val)

# Linéaire
y_pred_train_lineaire = predict_lineaire(X_train,Theta_lineaire)
y_pred_val_lineaire = predict_lineaire(X_val, Theta_lineaire)

# Polynomiale
y_pred_train_polynomiale_d2 = predict_noyau_polynomiale(X_train,X_train,2,r,Beta_d2)
y_pred_val_polynomiale_d2 = predict_noyau_polynomiale(X_train,X_val,2,r,Beta_d2)

y_pred_train_polynomiale_d3 = predict_noyau_polynomiale(X_train,X_train,3,r,Beta_d3)
y_pred_val_polynomiale_d3 = predict_noyau_polynomiale(X_train,X_val,3,r,Beta_d3)

# Tableau comparatif : (prédictions train, prédictions validation, coût final)
modeles = {
    "Moyenne (référence)" : (y_pred_train_moyenne , y_pred_val_moyenne , np.nan),
    "Linéaire" : (y_pred_train_lineaire , y_pred_val_lineaire , couts_lineaire[-1]),
    "Polynomiale d=2" : (y_pred_train_polynomiale_d2 , y_pred_val_polynomiale_d2 , couts_d2[-1]),
    "Polynomiale d=3" : (y_pred_train_polynomiale_d3 , y_pred_val_polynomiale_d3 , couts_d3[-1]),
}

lignes = []
for nom , (p_train , p_val , cout_final) in modeles.items():
    lignes.append({
        "Modèle" : nom,
        "MSE Train" : MSE(p_train,y_train),
        "MSE Val" : MSE(p_val,y_val),
        "MAE Train" : MAE(p_train,y_train),
        "MAE Val" : MAE(p_val,y_val),
        "Coût final" : cout_final,
    })

df_regression = pd.DataFrame(lignes)
print(df_regression.to_string(index=False))
tableau_png(df_regression , "Régression - Comparaison des modèles (entrainement / validation)" , "regression_tableau_validation" , surligner = "Polynomiale d=3")

#%% Régression - Visualisation

graphe_couts({
        "Linéaire" : couts_lineaire,
        "Polynomiale d=2" : couts_d2,
        "Polynomiale d=3" : couts_d3,
    },
    "Régression - Évolution du coût pendant l'entrainement" , "regression_couts",
    xlabel = "Itérations (échelle log)" , log_x = True , log_y = True)

#%% Régression - Comparaison

predictions_val = {
    "Linéaire" : y_pred_val_lineaire,
    "Polynomiale d=2" : y_pred_val_polynomiale_d2,
    "Polynomiale d=3" : y_pred_val_polynomiale_d3,
}

graphe_reel_vs_predit(y_val , predictions_val , "Régression (validation) - Note réelle vs note prédite" , "regression_reel_vs_predit_validation")
graphe_erreurs(y_val , predictions_val , "Régression (validation) - Distribution des erreurs" , "regression_erreurs_validation")

# Choix du modèle : régression à noyau polynomial de degré 3.
# Le tableau de validation (MSE et MAE) permet de comparer avec la référence (moyenne), le modèle linéaire et d=2.

# %% Régression - Evaluation
# Sur le test : uniquement la référence (moyenne) et le modèle choisi

y_pred_test = predict_noyau_polynomiale(X_train,X_test,3,r,Beta_d3)
y_pred_test_moyenne = predict_moyenne(y_train,X_test)

print("-"*8)
print("Modèle choisi : Régression à noyau polynomiale de degré 3")
print("-"*8)
print("MSE Test :", MSE(y_pred_test , y_test))
print("MAE Test :", MAE(y_pred_test , y_test))
print("-"*8)

df_test = pd.DataFrame([
    {"Modèle" : "Moyenne (référence)" , "MSE Test" : MSE(y_pred_test_moyenne,y_test) , "MAE Test" : MAE(y_pred_test_moyenne,y_test)},
    {"Modèle" : "Polynomiale d=3 (choisi)" , "MSE Test" : MSE(y_pred_test,y_test) , "MAE Test" : MAE(y_pred_test,y_test)},
])
tableau_png(df_test , "Régression - Résultats sur le jeu de test" , "regression_tableau_test" , surligner = "Polynomiale d=3 (choisi)")

graphe_reel_vs_predit(y_test , {"Polynomiale d=3 (choisi)" : y_pred_test} , "Régression (test) - Note réelle vs note prédite" , "regression_reel_vs_predit_test")
graphe_erreurs(y_test , {"Polynomiale d=3 (choisi)" : y_pred_test} , "Régression (test) - Distribution des erreurs" , "regression_erreurs_test")

#%% Optimisation - Visualisation

df_optim = pd.DataFrame([
    {"Méthode" : "Descente Complète" , "Pas" : alpha_C , "Mises à jour" : n_iterationsC , "Epoques" : epochsC , "Temps (s)" : elapsed_timeC , "Coût final" : coutsC[-1] , "Objectif atteint" : "oui" if coutsC[-1] < eps else "non"},
    {"Méthode" : "Descente Stochastique" , "Pas" : alpha_S , "Mises à jour" : n_iterationsS , "Epoques" : epochsS , "Temps (s)" : elapsed_timeS , "Coût final" : coutsS[-1] , "Objectif atteint" : "oui" if coutsS[-1] < eps else "non"},
    {"Méthode" : "Descente Batch (p = " + str(p) + ")" , "Pas" : alpha_B , "Mises à jour" : n_iterationsB , "Epoques" : epochsB , "Temps (s)" : elapsed_timeB , "Coût final" : coutsB[-1] , "Objectif atteint" : "oui" if coutsB[-1] < eps else "non"},
])

print("Objectif : coût < " , eps)
print(df_optim.to_string(index=False))
tableau_png(df_optim , f"Optimisation - Comparaison des descentes de gradient (objectif : coût < {eps:.3f})" , "optimisation_tableau")

graphe_optimisation({
        "Descente Complète" : (coutsC , tempsC),
        "Descente Stochastique" : (coutsS , tempsS),
        "Descente Batch" : (coutsB , tempsB),
    },
    f"Optimisation - Régression linéaire (objectif : coût < {eps:.3f})" , "optimisation_couts")

#%% Régression - Choix de r (optionnel, assez long)
# r (noyau (x.x' + r)^d) n'est pas appris par la descente de gradient : c'est un hyperparamètre.
# On le choisit avec la VALIDATION (jamais avec le test) : même nombre d'itérations pour chaque r (noyau de degré 3),
# on garde le r qui a la MSE de validation la plus basse. Ensuite, modifier r dans Modeles_Regression.py.
r_candidats = [0, 1, 5, 10, 20]
n_iterations_r = 3000

lignes_r = []
for r_test in r_candidats:
    Beta_r , _ = entrainer_noyau_polynomiale(X_train,y_train,None,3,r_test,n_iterations_r)
    lignes_r.append({
        "r" : r_test,
        "MSE Train" : MSE(predict_noyau_polynomiale(X_train,X_train,3,r_test,Beta_r),y_train),
        "MSE Val" : MSE(predict_noyau_polynomiale(X_train,X_val,3,r_test,Beta_r),y_val),
    })

df_r = pd.DataFrame(lignes_r)
r_conseille = int(df_r.loc[df_r["MSE Val"].idxmin() , "r"])
print(df_r.to_string(index=False))
print("r conseillé :" , r_conseille , "(r utilisé actuellement :" , r , ")")
tableau_png(df_r , f"Noyau polynomial d=3 - Choix de r sur la validation (r conseillé : {r_conseille})" , "regression_choix_r")

# %%
