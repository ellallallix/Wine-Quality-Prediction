# Prédiction de la qualité du vin blanc

Projet de Machine Learning sur le jeu de données *Wine Quality (white)*, avec deux problèmes sur la même cible :

- **Régression** : prédire la note de qualité du vin.
- **Classification** : prédire si la note est **supérieure ou égale à 7** (« bon vin ») ou non.

Tous les algorithmes (régression linéaire, noyau polynomial, régression logistique, descentes de gradient) sont codés à la main avec NumPy. Seuls les SVM (linéaire et gaussien) viennent de scikit-learn.

---

## 1. Organisation du projet

```
Code/
├── Data.py                      # 3 - chargement, nettoyage, séparation, standardisation
├── Modeles_Regression.py        # 1 - modèles de régression : fonctions + entrainement + descentes de gradient
├── Modeles_Classification.py    # 2 - modèles de classification : fonctions + entrainement + métriques
├── Outils.py                    # 4 - enregistrement des graphiques et tableaux en PNG
├── Evaluation_Regression.py     # 5 - visualisation, évaluation et comparaison (régression)
├── Evaluation_Classification.py # 6 - visualisation, évaluation et comparaison (classification)
├── Resultats/                   # tous les graphiques et tableaux (créé automatiquement)
└── README.md
Data/
└── winequality_white.csv
```

Dépendances : `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

**Exécution.** On lance `Evaluation_Regression.py` et `Evaluation_Classification.py` : chacun importe les fichiers de modèles, qui lancent l'entrainement. `Data.py` lancé seul produit les graphiques de visualisation des données. Tous les PNG sont écrits dans `Resultats/`.

### Répartition du travail
Ce projet a été réalisé avec l'aide de Claude (assistant IA d'Anthropic). La répartition est volontairement lisible dans l'organisation des fichiers.

| Fichier | Origine |
|---|---|
| `Modeles_Regression.py` | **Auteur du projet** : fonctions de score, coût, gradient, entrainement, prédiction (linéaire, moyenne, noyau polynomial), fonctions SGD / batch / complète. Corrections de Claude marquées `# [Claude]` |
| `Modeles_Classification.py` | **Auteur du projet** : régression logistique, SVM, classifieur majoritaire, métriques. Corrections de Claude marquées `# [Claude]` |
| `Data.py` | **Auteur du projet** : chargement, séparation, standardisation. Ajouts de Claude marqués `# [Claude]` (doublons, partition 70/15/15, graine, chemin, graphiques de visualisation) |
| `Outils.py` | **Claude** : fonctions de graphiques et de tableaux PNG |
| `Evaluation_Regression.py` | **Claude** pour les tableaux, graphiques, MAE, choix de r ; la fonction `MSE` et la structure (prédire sur train / validation / test) viennent de l'auteur |
| `Evaluation_Classification.py` | **Claude** pour les tableaux, matrices de confusion, FP / FN ; la structure d'origine (métriques train / validation, choix du modèle, métriques sur le test) vient de l'auteur |

Corrections apportées par Claude dans les fichiers de l'auteur (toutes marquées `# [Claude]`) :
- **Classification :** sigmoïde (`exp(-z)`) et coût logistique cohérent avec le gradient.
- **Régression :** référence constante (moyenne de `y_train`), détection d'une observation unique (`np.ndim`).
- **Descentes de gradient :** `entrainer_lineaire_CGD / SGD / BGD` gardent la structure d'origine mais ont été modifiées avec Claude (coût par époque, temps des mises à jour seulement, limite d'époques, pas par méthode).
- **Données :** suppression des doublons, vraie partition 70/15/15 commune aux deux scripts, graine fixe, chemin du CSV.

---

## 2. Protocole

**Cible.** La qualité du vin. Pour la régression, on cherche à la prédire. Pour la classification, on cherche à savoir si elle est supérieure ou égale à 7.

**Caractéristiques (11).** `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.

**Qualité des données.**
- Pas de valeurs manquantes (vérifié avec `df.isna().sum()`), et les données sont supposées propres vu leur source.
- Le fichier contient **937 lignes entièrement dupliquées**. Elles sont **supprimées avant la séparation** des données. Sinon, une même observation pourrait se retrouver à la fois en entraînement et en test, et l'évaluation serait trop favorable. Il reste donc 3961 vins.

**Transformation.** Standardisation (moyenne 0, écart-type 1) de toutes les entrées, car ce sont des valeurs numériques continues et non des classes. Le `StandardScaler` est **ajusté sur l'entraînement uniquement**, puis appliqué à la validation et au test.

**Séparation.** 70 % entraînement, 15 % validation, 15 % test, soit environ 2772 / 594 / 595 vins. Elle est stratifiée sur la note, avec une graine fixe (`SEED = 42`).
- L'entraînement sert à apprendre les paramètres.
- La validation sert à comparer les modèles et à choisir les hyperparamètres.
- Le test n'est utilisé **qu'une fois**, à la fin, uniquement pour la **référence** et le **modèle choisi**.
- Régression et classification utilisent **exactement les mêmes partitions** (elles viennent de `Data.py`).

**Reproductibilité.** La graine `SEED = 42` fixe les partitions, le SGD et le tirage des lots.

---

## 3. Visualisation des données

Trois graphiques, chacun avec un rôle précis avant de modéliser.

**Distribution des notes.** Elle montre que les notes se concentrent autour de 5 et 6 et que la classe « note ≥ 7 » est **minoritaire**. C'est ce qui justifie le choix des métriques en classification (section 5).

![Distribution des notes](Resultats/donnees_distribution_notes.png)

**Corrélations.** Elle montre quelles variables sont liées à la qualité, et surtout quelles variables sont **fortement corrélées entre elles** (par exemple `density`, `residual sugar` et `alcohol`). Cette collinéarité rend la descente de gradient lente, d'où le grand nombre d'itérations nécessaires.

![Corrélations](Resultats/donnees_correlations.png)

**Caractéristiques selon la classe.** Elle montre quelles variables séparent vraiment les bons vins des autres. Plus les deux boîtes sont décalées, plus la variable est informative.

![Caractéristiques selon la classe](Resultats/donnees_caracteristiques.png)

---

## 4. Régression

### Modèles
| Modèle | Rôle |
|---|---|
| **Moyenne** | Référence : prédit toujours la **moyenne des notes d'entraînement** |
| **Linéaire** | Descente de gradient sur le coût `1/2 · MSE` |
| **Noyau polynomial d = 2** | Noyau `(x·x' + r)^d`, descente de gradient sur les coefficients |
| **Noyau polynomial d = 3** | Idem, **modèle retenu** |

### Métriques
- **MSE** : pénalise fortement les grosses erreurs.
- **MAE** : erreur moyenne en points de note, plus facile à interpréter.

### Choix de l'hyperparamètre r
`r` ne s'apprend pas par descente de gradient. C'est un hyperparamètre, choisi sur la **validation** (jamais sur le test) : on entraîne le noyau de degré 3 pour plusieurs valeurs de `r` avec le même nombre d'itérations, et on garde celle qui donne la plus faible MSE de validation.

- `r = 0` : le noyau ne contient que les termes de degré exactement `d`.
- `r > 0` : il mélange les degrés de 0 à `d`, et plus `r` est grand, plus les petits degrés (proches du modèle linéaire) pèsent.
- Comme les données sont standardisées, `x·x'` vaut environ 11 (11 variables) : des valeurs de `r` entre 1 et 20 sont les plus pertinentes.

Pour cette recherche, le pas d'apprentissage est calculé automatiquement (`alpha = n / plus grande valeur propre de G`), ce qui reste stable quel que soit `r`. Le tableau donne un `r` **conseillé** : on le reporte ensuite à la main dans `Modeles_Regression.py` (si `r` change, revoir aussi les pas du noyau, ou les mettre à `None`).

![Choix de r](Resultats/regression_choix_r.png)

### Résultats (entraînement et validation)
Le tableau compare la référence et les trois modèles. Le modèle retenu est surligné.

![Tableau de validation](Resultats/regression_tableau_validation.png)

**Évolution du coût** pendant l'entraînement :

![Coûts de régression](Resultats/regression_couts.png)

**Note réelle contre note prédite** (validation). Une prédiction exacte est sur la diagonale : on y voit si le modèle surestime les petites notes et sous-estime les grandes.

![Réel vs prédit (validation)](Resultats/regression_reel_vs_predit_validation.png)

**Distribution des erreurs** (validation). Elle montre si les erreurs sont petites et si le modèle se trompe plutôt vers le haut ou vers le bas.

![Erreurs (validation)](Resultats/regression_erreurs_validation.png)

### Évaluation finale sur le test
Seuls la référence et le modèle choisi (noyau polynomial d = 3) sont évalués sur le test.

![Tableau de test](Resultats/regression_tableau_test.png)
![Réel vs prédit (test)](Resultats/regression_reel_vs_predit_test.png)
![Erreurs (test)](Resultats/regression_erreurs_test.png)

---

## 5. Classification

### Modèles
- **Classifieur majoritaire** : référence, prédit toujours la classe la plus fréquente (« note < 7 »).
- **Régression logistique** (sigmoïde, entropie croisée), avec trois seuils de décision : 30 %, 50 %, 70 %.
- **SVM linéaire**.
- **SVM à noyau gaussien**, avec `C ∈ {0.1, 1, 5}` et `gamma ∈ {0.1, 1, 5}`.

### Métriques et choix du modèle
On calcule quatre métriques : Accuracy, Precision, Recall et F1, plus le nombre de faux positifs (FP) et de faux négatifs (FN).

**Le choix ne se fait pas sur l'Accuracy.** Les vins de bonne qualité sont minoritaires, donc on obtient une bonne Accuracy en prédisant toujours « note < 7 » ou presque. Le classifieur majoritaire en est la preuve dans le tableau : son Accuracy est élevée alors que son Recall est nul, car il ne détecte aucun bon vin.

Le modèle est donc choisi en regardant les quatre critères, mais **principalement Precision et Recall, c'est-à-dire le F1** :
- **Precision** : parmi les vins que le modèle annonce « bons », quelle part l'est vraiment ? (limite les faux positifs)
- **Recall** : parmi les vrais bons vins, quelle part est retrouvée ? (limite les faux négatifs)
- **F1** : compromis entre les deux.

**Modèle retenu : SVM à noyau gaussien (C = 5, gamma = 1).**

![Tableau de validation de la classification](Resultats/classification_tableau_validation.png)

### Matrice de confusion en validation
Elle sert à ajuster `C` et `gamma` à la fin : elle montre combien de bons vins sont manqués (faux négatifs) et combien de vins moins bien notés sont sélectionnés à tort (faux positifs).

![Matrice de confusion (validation)](Resultats/classification_matrice_confusion_validation.png)

### Évaluation finale sur le test
Seuls la référence (classifieur majoritaire) et le modèle choisi sont évalués.

![Tableau de test](Resultats/classification_tableau_test.png)
![Matrice de confusion (test)](Resultats/classification_matrice_confusion_test.png)

### Coût de la régression logistique
![Coût logistique](Resultats/classification_couts.png)

---

## 6. Optimisation : comparaison des descentes de gradient

Comparaison de trois descentes de gradient sur la régression linéaire :
- **Complète** : tout l'entraînement à chaque mise à jour (1 mise à jour = 1 époque).
- **Stochastique (SGD)** : 1 observation par mise à jour (1 époque = n mises à jour).
- **Par lots** : lots de p = 5 observations.

**Objectif commun** : atteindre un coût inférieur à `eps`, défini comme le meilleur coût du modèle linéaire + 0,02. Il est donc atteignable.

Règles de mesure pour une comparaison équitable :
- Le coût (calculé sur tout l'entraînement) n'est évalué qu'**une fois par époque**.
- **Seul le temps des mises à jour est compté**, pas celui du calcul du coût.
- Chaque boucle a une **limite d'époques** (plus de boucle infinie) ; le tableau indique si l'objectif est atteint.

![Tableau d'optimisation](Resultats/optimisation_tableau.png)
![Coût selon les époques et selon le temps](Resultats/optimisation_couts.png)

Le graphique de gauche montre quelle méthode progresse le plus **par passage sur les données**. Celui de droite montre laquelle atteint un coût donné **le plus vite en secondes**.

---

## 7. Corrections apportées au code initial
- Sigmoïde corrigée (`exp(-z)`) et coût logistique cohérent avec le gradient.
- Référence de régression : moyenne de `y_train` (et non de `X_train`).
- Doublons supprimés avant la séparation.
- Séparation réellement 70/15/15, identique entre régression et classification.
- Graine fixée pour SGD et les lots, et chemin du CSV indépendant du dossier de lancement.
- Détection d'une observation unique par `np.ndim` (et non plus `len(X_aug) == 12`).
- Boucles d'optimisation bornées, et coût calculé une fois par époque.

## 8. Limites
- Les doublons sont supprimés plutôt que conservés : c'est un choix assumé, qui réduit le jeu à 3961 vins.
- Les hyperparamètres (`r`, `C`, `gamma`, seuil) sont choisis sur une seule validation, sans validation croisée.
- Les notes extrêmes (3 et 9) sont très rares, donc mal apprises par les modèles de régression.
