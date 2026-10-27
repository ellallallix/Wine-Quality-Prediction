# Projet de consolidation — Wine Quality

## Contexte et mission
Un laboratoire dispose d'analyses physicochimiques de vins blancs et d'une note donnée lors d'une évaluation sensorielle. Il souhaite déterminer si les analyses permettent d'anticiper cette note et de repérer les vins bien notés.
Le contexte de décision ci-dessus est pédagogique ; les mesures sont réelles.

Deux missions utilisent les mêmes caractéristiques :
- Régression : prédire `quality`, la note sensorielle.
- Classification : prédire la classe 1 si `quality >= 7`, et 0 sinon. Ce seuil définit le problème du projet ; il n'est pas un seuil officiel de qualité.

Le seuil 7 qui définit la cible est distinct du seuil appliqué à une probabilité pour décider d'une classe.
Une note entière est une évaluation ordonnée, pas un comptage d'événements : Poisson n'est pas imposé ici. La traiter comme une grandeur numérique pour une régression est un choix à discuter.

## Données fournies
`winequality_white.csv` est le fichier blanc original du jeu UCI Wine Quality, conservé sans modification. Séparateur : point-virgule. Décimales : point. Une ligne représente une observation ; 11 colonnes sont des mesures et `quality` est la cible.
Source : https://archive.ics.uci.edu/dataset/186/wine+quality
Fichier : https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
Référence : Cortez, Cerdeira, Almeida, Matos et Reis (2009), Modeling wine preferences by data mining from physicochemical properties.
Téléchargé le 26 octobre 2026. Ce jeu n'est pas celui de `sklearn.datasets.load_wine`, dont la cible est différente.

### Dictionnaire français / anglais
| Colonne | Signification |
|---|---|
| fixed acidity | Acidité fixe |
| volatile acidity | Acidité volatile |
| citric acid | Acide citrique |
| residual sugar | Sucre résiduel |
| chlorides | Chlorures |
| free sulfur dioxide | Dioxyde de soufre libre |
| total sulfur dioxide | Dioxyde de soufre total |
| density | Masse volumique |
| pH | pH |
| sulphates | Sulfates |
| alcohol | Teneur en alcool |
| quality | Note sensorielle, sur une échelle théorique de 0 à 10 |

## Ce que tu dois livrer
Un script ou notebook reproductible, les résultats des expériences et un court compte rendu expliquant tes décisions. Aucun squelette ni ordre de fonctions n'est imposé. Tu peux réutiliser tes anciennes fonctions en les adaptant ; tu dois pouvoir en expliquer le fonctionnement.

### 1. Un protocole défendable
Présenter les cibles, les caractéristiques disponibles au moment de prédire, la distribution des sorties, les points de qualité des données et les transformations retenues.
Prévoir entraînement, validation et test, avec les mêmes observations dans chaque partition pour comparer les deux missions. Justifier les proportions et fixer la génération aléatoire.
Réserver le test à l'évaluation finale des choix arrêtés. Les transformations apprises sur les données doivent respecter cette séparation.
Examiner les répétitions de lignes et documenter leur traitement : des copies identiques réparties entre train et test peuvent fausser l'évaluation. Si tu choisis de les retirer, expliquer l'hypothèse de cette décision ; des lignes identiques ne prouvent pas à elles seules qu'il s'agit du même prélèvement.

### 2. Une comparaison de modèles
Pour la régression :
- une référence constante ;
- ta régression linéaire multivariée en NumPy ;
- une version non linéaire de ton choix parmi les approches étudiées (caractéristiques polynomiales, régression à noyau ou régression localement pondérée).

Pour la classification :
- une référence constante ;
- ta régression logistique multivariée en NumPy ;
- une SVM avec scikit-learn, avec au moins deux noyaux comparés.

Tu choisis les hyperparamètres essayés et expliques pourquoi. On n'attend pas une recherche exhaustive. GDA, perceptron et softmax sont facultatifs : ne pas tout ajouter uniquement pour multiplier les modèles. Aucun arbre ni réseau neuronal n'est requis.
Une implémentation de bibliothèque peut servir de contrôle pour ton modèle linéaire ou logistique ; si ses réglages ajoutent une pénalisation, ce n'est plus exactement le même objectif.

### 3. Une vraie expérience sur l'optimisation
Implémenter, pour au moins un de tes deux modèles NumPy, les variantes complète, stochastique et par petits lots de la descente de gradient.
Fixer un objectif commun. Comparer leurs évolutions, le coût de calcul et les performances, en tenant compte du nombre de mises à jour, des époques et du temps écoulé. Expliquer comment tu rends la comparaison équitable et comment tu règles les pas et tailles de lots.
Ne pas comparer directement des pertes calculées sur des observations différentes comme si elles étaient un coût global commun.
L'implémentation des trois méthodes sur le deuxième modèle est un prolongement facultatif.

### 4. Une évaluation reliée à la décision
Régression : présenter au moins MSE et MAE, et expliquer ce qu'une erreur signifie dans l'unité de la note.
Classification : présenter accuracy, précision, rappel, F1 et les nombres de faux positifs et faux négatifs. Définir explicitement la classe positive.
Étudier l'effet du seuil de décision de la logistique. Dans le scénario pédagogique, on souhaite retrouver les vins bien notés tout en évitant de sélectionner trop de vins qui ne le sont pas. Choisir et justifier un compromis avant d'utiliser le test. Il n'y a pas de ratio de coût fourni : annoncer l'hypothèse choisie.
Ne pas transformer directement un score de SVM en probabilité.

### 5. Une conclusion honnête
Choisir un modèle pour chaque mission, puis l'évaluer une fois sur le test. Comparer aux références constantes.
Analyser quelques erreurs et les écarts entraînement/validation. Expliquer les limites des mesures disponibles et de la note sensorielle. Aucun seuil d'accuracy ou de MSE n'est imposé.
Si un modèle complexe ne gagne pas, c'est un résultat à interpréter, pas une raison de modifier le test.

## Livrables minimum
- Code exécutable et versions des dépendances utilisées.
- Tableau des modèles, hyperparamètres, métriques et temps pertinents.
- Courbes d'apprentissage comparables entre les trois optimisations.
- Au moins une visualisation utile par mission, commentée.
- Compte rendu bref : choix, résultats, erreurs et limites.

La réussite se juge sur la validité de la démarche, la compréhension du code et la qualité des conclusions, pas sur un score spectaculaire.

## Questions mathématiques à intégrer au compte rendu
1. Écrire les prédictions et coûts de tes deux modèles NumPy, puis retrouver leurs gradients. Préciser les dimensions et le traitement du biais.
2. Montrer comment une même expression d'apprentissage s'adapte à un lot complet, une observation et un petit lot ; expliquer le facteur de normalisation.
3. Expliquer pourquoi un noyau différent change les formes possibles sans imposer une nouvelle loi de probabilité à la cible.
4. Expliquer pourquoi la cible binaire ne doit pas figurer parmi les entrées de régression, et pourquoi la note ne doit pas figurer parmi celles de classification.

## Organisation et accompagnement
Tu choisis la structure du projet, tes fonctions, tes expériences et leur ordre. Tu peux conserver un journal court de tes essais pour ne pas répéter les mêmes erreurs.
Pour limiter le travail : commencer par le périmètre minimum, pas toutes les variantes du cours. Si un calcul à noyau devient trop coûteux, documenter un sous-échantillon commun pour cette comparaison plutôt que patienter indéfiniment.
En cas de blocage : préciser ce que tu voulais obtenir, ce que tu observes et ce que tu as déjà essayé. L'aide commencera par un diagnostic ou un indice ; le corrigé complet ne sera pas donné spontanément. Les visualisations complexes peuvent être fournies sur demande.
