#%% Libraries et données

import numpy as np
import matplotlib.pyplot as plt
from Data import X_train,X_test,X_val,y_train,y_test,y_val
import time
from random import randint , shuffle

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
    if len(X_aug) ==12:

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
def predict_moyenne(X_train,X):
    n,_ = np.shape(X)
    return np.array([np.mean(X_train)] * n ) 

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
r = 5 
k= 8 # coeff sur n_iterations pour le noyau polynomiale


Theta_lineaire , couts_lineaire = entrainer_lineaire(X_train,y_train,k*n_iterations,alpha)

Beta_d2 , couts_d2 = entrainer_noyau_polynomiale(X_train,y_train,0.01,2,r,k * n_iterations)
Beta_d3 , couts_d3 = entrainer_noyau_polynomiale(X_train,y_train,1e-5,3,r,k * n_iterations)
# Beta_d5 , couts_d5 = entrainer_noyau_polynomiale(X_train,y_train,1e-8,5,r,100000)

#%% Régression - Validation

def MSE(y_pred,y):
    e = y_pred-y
    return np.mean(e**2)

# Moyenne
y_pred_train_moyenne = predict_moyenne(X_train,X_train)
y_pred_val_moyenne = predict_moyenne(X_train,X_val)

# Linéaire
y_pred_train_lineaire = predict_lineaire(X_train,Theta_lineaire)
y_pred_val_lineaire = predict_lineaire(X_val, Theta_lineaire)


# Polynomiale
y_pred_train_polynomiale_d2 = predict_noyau_polynomiale(X_train,X_train,2,r,Beta_d2)
y_pred_val_polynomiale_d2 = predict_noyau_polynomiale(X_train,X_val,2,r,Beta_d2)

y_pred_train_polynomiale_d3 = predict_noyau_polynomiale(X_train,X_train,3,r,Beta_d3)
y_pred_val_polynomiale_d3 = predict_noyau_polynomiale(X_train,X_val,3,r,Beta_d3)

"""
y_pred_train_polynomiale_d5 = predict_noyau_polynomiale(X_train,X_train,5,r,Beta_d5)
y_pred_val_polynomiale_d5 = predict_noyau_polynomiale(X_train,X_val,5,r,Beta_d5)
"""

print("Moyenne :")
print("MSE Entrainement :" , MSE(y_pred_train_moyenne,y_train))
print("MSE Validation :" , MSE(y_pred_val_moyenne,y_val))

print("-"*19)

print("Linéaire :")
print("MSE Entrainement :", MSE(y_pred_train_lineaire,y_train))
print("MSE Entrainement :", MSE(y_pred_val_lineaire,y_val))
print("cout finale", couts_lineaire[-1])

print("-"*19)

print("Polynomiale :")

print("-"*8)
print("degré : 2")
print("MSE Entrainement :", MSE(y_pred_train_polynomiale_d2 , y_train))
print("MSE Validation :", MSE(y_pred_val_polynomiale_d2 , y_val))
print("cout finale", couts_d2[-1])


print("-"*8)
print("degré : 3")
print("MSE Entrainement :", MSE(y_pred_train_polynomiale_d3 , y_train))
print("MSE Validation :", MSE(y_pred_val_polynomiale_d3 , y_val))
print("cout finale", couts_d3[-1])

"""
print("-"*8)
print("degré : 5")
print("MSE Entrainement :", MSE(y_pred_train_polynomiale_d5 , y_train))
print("MSE Validation :", MSE(y_pred_val_polynomiale_d5 , y_val))
print("cout finale", couts_d5[-1])
"""

#%% Régression - Visualisation

plt.plot(np.arange(k* n_iterations) , couts_lineaire)
plt.title("Le cout Linéaire")
plt.show()

plt.plot(np.arange(k * n_iterations) , couts_d2)
plt.title(" Le cout Polynomiale d=2 ")
plt.show()

plt.plot(np.arange(k * n_iterations) , couts_d3)
plt.title(" Le cout Polynomiale d=3 ")
plt.show()

"""
plt.plot(np.arange(100000) , couts_d5)
plt.title(" Le cout Polynomiale d=5 ")
plt.show()
"""
#%% Régression - Comparaison


n_val = len(y_val)


plt.plot(np.arange(100) , y_val[:100], label = "Données de validation")
plt.plot(np.arange(100) , y_pred_val_lineaire[:100], label = "Régression Linéaire")
plt.legend()
plt.title("Comparaison des résultats sur l'évaluation (1)")
plt.show()

plt.plot(np.arange(100) , y_val[:100], label = "Données de validation")
plt.plot(np.arange(100) , y_pred_val_polynomiale_d2[:100], label = "Régression Polynomiale (d2)")
plt.legend()
plt.title("Comparaison des résultats sur l'évaluation (2)")
plt.show()

plt.plot(np.arange(100) , y_val[:100], label = "Données de validation")
plt.plot(np.arange(100) , y_pred_val_polynomiale_d3[:100], label = "Régression Polynomiale (d3)")
plt.legend()
plt.title("Comparaison des résultats sur l'évaluation (3)")
plt.show()


# Le noyau polynomiale atteint la MSE la plus basse sur Entrainement et Validation ( meme si l'écart entre les deux est plus grand que pour les autres modèles )
# On fait le choix de retenir la régression à noyau polynomiale à d = 2 car sur l'évaluation les résultats des deux (Noyau polynomaile d=2 et d=3) sont assez proches et celui à d=2 est plus vite et consomme moins.

# %% Régression - Evaluation

y_pred_test = predict_noyau_polynomiale(X_train,X_test,2,r,Beta_d2)

print("-"*8)
print("Modèle choisi : Régression à noyau polynomiale de degré 2")
print("-"*8)
print("MSE Test :", MSE(y_pred_test , y_test))
print("-"*8)



plt.plot(np.arange(50) , y_test[:50], label = "Données de validation")
plt.plot(np.arange(50) , y_pred_test[:50], label = "Régression Polynomiale (d2)" , color = "r")
plt.legend()
plt.title("Résultats du Test")
plt.show()

#%% Régression - Optimisation

eps = 0.3
p = 5 

def generer_Batch(max_entier, taille_sous_liste):
    tous_les_indices = list(range(max_entier ))
    shuffle(tous_les_indices)
    Batches = [
        tous_les_indices[i : i + taille_sous_liste] 
        for i in range(0, len(tous_les_indices), taille_sous_liste)
    ]
    return Batches


def entrainer_lineaire_CGD(X,y,alpha,eps):
    start_time = time.perf_counter()

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    couts = []
    Theta = np.zeros((d+1))
    n_iterations = 0
    while len(couts) ==0 or couts[-1] >= eps :
        Theta -= alpha * gradient(X_aug,y,Theta)
        couts.append(cout_lineaire(X_aug,y,Theta))
        n_iterations += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    epochs = n_iterations
    return Theta , couts , n_iterations , elapsed_time , epochs 



def entrainer_lineaire_SGD(X,y,alpha,eps):
    start_time = time.perf_counter()

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    couts = []
    Theta = np.zeros((d+1))
    n_iterations = 0

    while len(couts) ==0 or couts[-1] >= eps :
        k = randint(0,n-1)
        Theta -= alpha * gradient(X_aug[k],y.iloc[k],Theta)
        couts.append(cout_lineaire(X_aug,y,Theta))
        n_iterations += 1
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    epochs = n_iterations / n

    return Theta , couts , n_iterations , elapsed_time , epochs



def entrainer_lineaire_BGD(X,y,alpha,eps,p):
    start_time = time.perf_counter()

    n ,d= np.shape(X)
    X_aug = np.column_stack((X,np.ones( (n,) ))) 
    couts = []
    Theta = np.zeros((d+1))
    n_iterations = 0
    batches = generer_Batch(n,p)
    n_batches = len(batches)

    while len(couts) ==0 or couts[-1] >= eps :
        i = n_iterations % n_batches 
        Theta -= alpha * gradient([X_aug[k] for k in batches[i]] ,[y.iloc[k] for k in batches[i]] ,Theta)
        couts.append(cout_lineaire(X_aug,y,Theta))
        n_iterations += 1


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    epochs = n_iterations * 5 / n 
    return Theta , couts , n_iterations , elapsed_time , epochs

ThetaC , coutsC , n_iterationsC , elapsed_timeC , epochsC = entrainer_lineaire_CGD(X_train,y_train,alpha,eps) 
ThetaS , coutsS , n_iterationsS , elapsed_timeS , epochsS = entrainer_lineaire_SGD(X_train,y_train,alpha,eps) 
ThetaB , coutsB , n_iterationsB , elapsed_timeB , epochsB = entrainer_lineaire_BGD(X_train,y_train,1e-5,eps,p) 

 #%% Optimisation - Visualisation




print("*" * 19)
print("Temps écoulé :")
print("Complete Gradient Descent :" , elapsed_timeC)
print("Stochastic Gradient Descent :" , elapsed_timeS)
print("Batch Gradient Descent :" , elapsed_timeB)
print("*" * 19)
print("Epochs parcourues :")
print("Complete Gradient Descent :" , epochsC)
print("Stochastic Gradient Descent :" , epochsS)
print("Batch Gradient Descent :" , epochsB)

plt.plot(np.arange(len(coutsC)),coutsC,label = "Descente Complète")
plt.plot(np.arange(len(coutsS)),coutsS,label = "Descente Stochastique")
plt.plot(np.arange(len(coutsB)),coutsB,label = "Descente Batch")

plt.legend()
plt.show()



# %%
