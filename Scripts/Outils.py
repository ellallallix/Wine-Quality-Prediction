"""
4 - OUTILS
Fichier ajouté avec Claude (non fait par l'utilisateur) : enregistrement des graphiques et tableaux en PNG.
Fonctions utilitaires : enregistrement des graphiques et des tableaux (PNG).
Tout est enregistré dans le dossier "Resultats" placé à côté des scripts.
"""

#%% Libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DOSSIER = Path(__file__).resolve().parent / "Resultats"
DOSSIER.mkdir(exist_ok=True)

#%% Enregistrement

def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(DOSSIER / f"{nom}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def tableau_png(df, titre, nom, decimales=3, surligner=None):
    """Affiche un DataFrame sous forme de tableau, enregistré en PNG.
    surligner : nom (1re colonne) de la ligne à mettre en évidence, par exemple le modèle retenu."""

    texte = df.copy()
    for c in texte.columns:
        if pd.api.types.is_float_dtype(texte[c]):
            texte[c] = texte[c].map(lambda v: "-" if pd.isna(v) else f"{v:.{decimales}f}")

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(texte.columns)), 0.34 * (len(texte) + 1) + 0.9))
    ax.axis("off")
    ax.set_title(titre, fontsize=13, fontweight="bold", pad=10)
    tab = ax.table(cellText=texte.values, colLabels=texte.columns, cellLoc="center", loc="upper center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.auto_set_column_width(list(range(len(texte.columns))))
    tab.scale(1, 1.6)
    for (i, j), cellule in tab.get_celld().items():
        if i == 0:
            cellule.set_facecolor("#2F4B7C")
            cellule.set_text_props(color="white", fontweight="bold")
        elif surligner is not None and df.iloc[i - 1, 0] == surligner:
            cellule.set_facecolor("#FFE9A8")
        elif i % 2 == 0:
            cellule.set_facecolor("#EEF1F6")
    sauver(fig, nom)

#%% Graphiques de visualisation des données

def graphe_distribution_notes(y, titre, nom):
    """Nombre de vins par note, avec la séparation note < 7 / note >= 7 utilisée en classification."""
    notes = np.sort(np.unique(y))
    effectifs = [np.sum(y == note) for note in notes]
    couleurs = ["#2F4B7C" if note < 7 else "#E07B39" for note in notes]
    part = np.mean(y >= 7) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    barres = ax.bar(notes, effectifs, color=couleurs)
    ax.bar_label(barres)
    ax.set_xlabel("Note de qualité")
    ax.set_ylabel("Nombre de vins")
    ax.set_title(titre, fontsize=14, fontweight="bold")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2F4B7C"), plt.Rectangle((0, 0), 1, 1, color="#E07B39")],
              labels=["Note < 7 (classe 0)", f"Note ≥ 7 (classe 1) : {part:.1f} % des vins"])
    ax.grid(alpha=0.3, axis="y")
    sauver(fig, nom)


def graphe_correlations(df, titre, nom):
    """Matrice des corrélations entre toutes les colonnes (caractéristiques et qualité)."""
    C = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)), C.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(C)), C.columns)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f"{C.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Corrélation")
    ax.set_title(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)


def graphe_caracteristiques(df, y_bin, titre, nom):
    """Pour chaque caractéristique : boîtes à moustaches des vins de note < 7 et de note >= 7."""
    colonnes = [c for c in df.columns if c != "quality"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, c in zip(axes.flat, colonnes):
        ax.boxplot([df[c][y_bin == 0], df[c][y_bin == 1]], showfliers=False)
        ax.set_xticks([1, 2], ["< 7", "≥ 7"])
        ax.set_title(c)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(colonnes):]:
        ax.axis("off")
    fig.suptitle(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)

#%% Graphiques de régression

def graphe_reel_vs_predit(y, predictions, titre, nom):
    """Note réelle (x) contre note prédite (y). Une prédiction exacte est sur la diagonale."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 5), squeeze=False)
    for ax, (nom_modele, y_pred) in zip(axes[0], predictions.items()):
        # petit décalage horizontal : les notes réelles sont des entiers, les points se superposeraient
        ax.scatter(y + rng.uniform(-0.15, 0.15, len(y)), y_pred, s=10, alpha=0.4)
        ax.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], "r--", label="Prédiction exacte")
        ax.set_title(nom_modele)
        ax.set_xlabel("Note réelle")
        ax.set_ylabel("Note prédite")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)


def graphe_erreurs(y, predictions, titre, nom):
    """Distribution des erreurs (prédite - réelle) : > 0 = surestimation, < 0 = sous-estimation."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 4.5), squeeze=False)
    for ax, (nom_modele, y_pred) in zip(axes[0], predictions.items()):
        e = y_pred - y
        ax.hist(e, bins=30, color="#2F4B7C", alpha=0.8)
        ax.axvline(0, color="k", linestyle="--", label="Erreur nulle")
        ax.axvline(np.mean(e), color="r", label=f"Erreur moyenne = {np.mean(e):.2f}")
        ax.set_title(nom_modele)
        ax.set_xlabel("Erreur (note prédite - note réelle)")
        ax.set_ylabel("Nombre de vins")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)


def graphe_couts(courbes, titre, nom, xlabel, log_x=False, log_y=False):
    """courbes : dictionnaire {nom du modèle : liste des coûts}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for nom_courbe, couts in courbes.items():
        ax.plot(np.arange(1, len(couts) + 1), couts, label=nom_courbe)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(titre, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coût")
    ax.grid(alpha=0.3)
    ax.legend()
    sauver(fig, nom)

#%% Graphique de classification

def graphe_matrice_confusion(y, y_pred, titre, nom):
    vn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    vp = np.sum((y == 1) & (y_pred == 1))
    M = np.array([[vn, fp], [fn, vp]])
    etiquettes = [["Vrais négatifs", "Faux positifs"], ["Faux négatifs", "Vrais positifs"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            couleur = "white" if M[i, j] > M.max() / 2 else "black"
            ax.text(j, i, f"{M[i, j]}\n{etiquettes[i][j]}", ha="center", va="center", color=couleur, fontsize=12)
    ax.set_xticks([0, 1], ["Note < 7", "Note ≥ 7"])
    ax.set_yticks([0, 1], ["Note < 7", "Note ≥ 7"])
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)

#%% Graphique d'optimisation

def graphe_optimisation(resultats, titre, nom):
    """resultats : {méthode : (couts, temps)} avec un coût (et un temps cumulé) par époque."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nom_methode, (couts, temps) in resultats.items():
        axes[0].plot(np.arange(len(couts)), couts, marker="o", markersize=3, label=nom_methode)
        axes[1].plot(temps, couts, label=nom_methode)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("Époques (passages sur les données, échelle symlog)")
    axes[1].set_xlabel("Temps d'entraînement cumulé (s)")
    axes[0].set_title("Coût selon les époques")
    axes[1].set_title("Coût selon le temps")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Coût (échelle log)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(titre, fontsize=14, fontweight="bold")
    sauver(fig, nom)
