"""
1 - MODÈLES DE RÉGRESSION
Fonctions d'entrainement et entrainement des modèles : travail de l'utilisateur.
Les lignes marquées [Claude] sont des corrections ou ajouts faits avec Claude (détail dans le README).
Aucun graphique ni tableau dans ce fichier.
"""

#%% Libraries et données

import numpy as np
from Data import X_train,X_test,X_val,y_train,y_test,y_val,SEED
import time
from random import randint , shuffle , seed

seed(SEED)  # [Claude] graine fixe : SGD et lots reproductibles

#%% Régression - Création
"""
Régression Linéaire
"""
def score_lineaire(X_aug,Theta):
    return X_aug @ Theta

def cout_lineaire(X_aug,y,Theta):
    E = ( score_lineaire(X_aug,Theta) - y )
    return 1/2 * np.mean(E **2)

def gradient (X_aug,y,Theta):

    E = score_lineaire(X_aug,Theta) - y 
    if np.ndim(X_aug) == 1:  # [Claude] correction : l'original testait len(X_aug) == 12 (confusion avec le nb de caractéristiques)

        return E * X_aug
    n,_ = np.shape(X_aug)

    return 1/n * E @ X_aug

def entrainer_lineaire(X,y,n_iterations,alpha):
    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    couts = []
    Theta = np.zeros((d+1))

    for _ in range(n_iterations):
        Theta -= alpha * gradient(X_aug,y,Theta)
        couts.append(cout_lineaire(X_aug,y,Theta))
    return Theta , couts


def predict_lineaire(X,Theta):
    n,_ = np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    return score_lineaire(X_aug,Theta)

"""
La moyenne
"""
def predict_moyenne(y_train,X):  # [Claude] correction : l'original prenait la moyenne de X_train au lieu de y_train
    n,_ = np.shape(X)
    return np.array([np.mean(y_train)] * n ) 

"""
Régression à noyau Polynomiale
"""

def noyau_polynomiale(X,x,d,r):
    return (X @ x.T + r)**d

def cout_noyau_polynomiale(G,y,beta):

    E = G @ beta - y

    return 1/2 * np.mean(E**2)

def entrainer_noyau_polynomiale(X,y,alpha,d,r,n_iterations):
    n,_ = np.shape(X)
    G = noyau_polynomiale(X,X,d,r)
    if alpha is None:  # [Claude] ajout : pas automatique (utilisé seulement pour la recherche de r, voir Evaluation_Regression.py)
        alpha = n / np.linalg.eigvalsh(G)[-1]
    beta = np.zeros((n,))
    couts = []

    for _ in range(n_iterations):
        beta -= alpha/n * (G @ beta - y)
        couts.append(cout_noyau_polynomiale(G,y,beta))

    return beta , couts


def predict_noyau_polynomiale(X,x,d,r,beta):
    G = noyau_polynomiale(X,x,d,r)

    return G.T @ beta


#%% Régression - Entrainement
n_iterations = 5000
alpha = 0.1
r = 5  # à ajuster avec le tableau "Choix de r" de Evaluation_Regression.py ; si r change, revoir les pas ci-dessous (ou mettre None)
k= 8 # coeff sur n_iterations pour le noyau polynomiale


Theta_lineaire , couts_lineaire = entrainer_lineaire(X_train,y_train,k*n_iterations,alpha)

Beta_d2 , couts_d2 = entrainer_noyau_polynomiale(X_train,y_train,0.01,2,r,k * n_iterations)
Beta_d3 , couts_d3 = entrainer_noyau_polynomiale(X_train,y_train,1e-5,3,r,k * n_iterations)
# Beta_d5 , couts_d5 = entrainer_noyau_polynomiale(X_train,y_train,1e-8,5,r,100000)

#%% Régression - Optimisation
# Comparaison des descentes de gradient (linéaire) : complète, stochastique, par lots

p = 5 
max_epochs = {"CGD" : 20000 , "SGD" : 200 , "BGD" : 200}  # [Claude] limite de sécurité : plus de boucle infinie

eps = couts_lineaire[-1] + 0.02  # [Claude] objectif atteignable (l'original : eps = 0.3, parfois inatteignable)

alpha_C = alpha   # [Claude] un pas par méthode (l'original : alpha pour CGD et SGD, 1e-5 pour BGD)
alpha_S = 0.01
alpha_B = 0.05

def generer_Batch(max_entier, taille_sous_liste):
    tous_les_indices = list(range(max_entier ))
    shuffle(tous_les_indices)
    Batches = [
        tous_les_indices[i : i + taille_sous_liste] 
        for i in range(0, len(tous_les_indices), taille_sous_liste)
    ]
    return Batches

# [Claude] Les 3 fonctions ci-dessous gardent la structure de l'original (boucle while, mêmes noms),
# mais ont été modifiées avec Claude :
#  - le coût est calculé une fois par époque (et non à chaque mise à jour) ;
#  - seul le temps des mises à jour est mesuré ;
#  - ajout de max_epochs et d'un retour supplémentaire "temps" (temps cumulé à chaque époque) ;
#  - y indexé en numpy (y[i]) et lots tirés à chaque époque.

def entrainer_lineaire_CGD(X,y,alpha,eps,max_epochs):

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    Theta = np.zeros((d+1))
    couts = [cout_lineaire(X_aug,y,Theta)]
    temps = [0.0]
    n_iterations = 0

    while couts[-1] >= eps and n_iterations < max_epochs :
        debut = time.perf_counter()
        Theta -= alpha * gradient(X_aug,y,Theta)
        temps.append(temps[-1] + time.perf_counter() - debut)
        n_iterations += 1
        couts.append(cout_lineaire(X_aug,y,Theta))

    epochs = n_iterations # 1 mise à jour = 1 passage sur les données
    return Theta , couts , n_iterations , temps[-1] , epochs , temps



def entrainer_lineaire_SGD(X,y,alpha,eps,max_epochs):

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    Theta = np.zeros((d+1))
    couts = [cout_lineaire(X_aug,y,Theta)]
    temps = [0.0]
    n_iterations = 0

    while couts[-1] >= eps and n_iterations < max_epochs * n :
        debut = time.perf_counter()
        for _ in range(n):  # une époque = n mises à jour
            i = randint(0,n-1)
            Theta -= alpha * gradient(X_aug[i],y[i],Theta)
        temps.append(temps[-1] + time.perf_counter() - debut)
        n_iterations += n
        couts.append(cout_lineaire(X_aug,y,Theta))
    
    epochs = n_iterations / n

    return Theta , couts , n_iterations , temps[-1] , epochs , temps



def entrainer_lineaire_BGD(X,y,alpha,eps,p,max_epochs):

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    Theta = np.zeros((d+1))
    couts = [cout_lineaire(X_aug,y,Theta)]
    temps = [0.0]
    n_iterations = 0
    n_epochs = 0

    while couts[-1] >= eps and n_epochs < max_epochs :
        debut = time.perf_counter()
        for lot in generer_Batch(n,p):  # une époque = tous les lots (mélangés)
            Theta -= alpha * gradient(X_aug[lot] , y[lot] , Theta)
            n_iterations += 1
        temps.append(temps[-1] + time.perf_counter() - debut)
        n_epochs += 1
        couts.append(cout_lineaire(X_aug,y,Theta))

    epochs = n_epochs
    return Theta , couts , n_iterations , temps[-1] , epochs , temps

ThetaC , coutsC , n_iterationsC , elapsed_timeC , epochsC , tempsC = entrainer_lineaire_CGD(X_train,y_train,alpha_C,eps,max_epochs["CGD"]) 
ThetaS , coutsS , n_iterationsS , elapsed_timeS , epochsS , tempsS = entrainer_lineaire_SGD(X_train,y_train,alpha_S,eps,max_epochs["SGD"]) 
ThetaB , coutsB , n_iterationsB , elapsed_timeB , epochsB , tempsB = entrainer_lineaire_BGD(X_train,y_train,alpha_B,eps,p,max_epochs["BGD"]) 

# %%
