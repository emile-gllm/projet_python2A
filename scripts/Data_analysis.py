import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import prince

def calculer_acp_pays(data, pays_nom, colonnes_quanti):
    """
    Filtre les données pour un pays et calcule l'ACP sur les variables quantitatives.
    """
    #Filtrage sur le pays
    if isinstance(pays_nom, list):
        subset = data[data['COUNTRY'].isin(pays_nom)].copy()
    else:
        subset = data[data['COUNTRY'] == pays_nom].copy()
    
    # SELECTION:On ne garde que les variables quantitatives
    data_for_pca = subset[colonnes_quanti]

    #Normalisation 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)

    #Calcul de l'ACP
    pca = PCA()
    pca_features = pca.fit_transform(data_scaled)
    
    # On retourne les noms des colonnes quanti pour le cercle des corrélations
    return pca, pca_features, colonnes_quanti



def trace_cercle_et_variance(pca, colonnes_noms, titre_prefixe=""):
    """
    Barplot des variances expliquées par chaque axe et le cercle des corrélations
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))

    #Graphique de la variance 
    variance = pca.explained_variance_ratio_
    ax1.bar(range(1, len(variance) + 1), variance, color='skyblue')
    ax1.plot(range(1, len(variance) + 1), np.cumsum(variance), marker='o', color='red')
    ax1.set_title(f"{titre_prefixe} - Variance expliquée", fontsize=14)
    ax1.set_xlabel("Axes")
    ax1.set_ylabel("% de variance")

    #Cercle des corrélations
    pcs = pca.components_
    # On dessine les flèches
    for i, col in enumerate(colonnes_noms):
        ax2.arrow(0, 0, pcs[0, i], pcs[1, i], color='r', alpha=0.5, head_width=0.03)
        # Ajustement du texte pour qu'il ne chevauche pas les flèches
        ax2.text(pcs[0, i]*1.1, pcs[1, i]*1.1, col, color='black', 
                 ha='center', va='center', fontsize=11, fontweight='bold')

    #Dessin cercle unité
    circle = plt.Circle((0,0), 1, color='navy', fill=False, linestyle='--')
    ax2.add_artist(circle)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axhline(0, color='black', lw=1, alpha=0.5)
    ax2.axvline(0, color='black', lw=1, alpha=0.5)
    
    # Titres et labels avec les pourcentages de variance
    ax2.set_title(f"{titre_prefixe} - Cercle des corrélations (Axe 1 & 2)", fontsize=14, pad=10)
    ax2.set_xlabel(f"F1 ({variance[0]:.1%})", fontsize=12)
    ax2.set_ylabel(f"F2 ({variance[1]:.1%})", fontsize=12)
    
    #Pour que le cercle soit vraiment rond:
    ax2.set_aspect('equal')

    plt.tight_layout(pad=4.0)
    plt.show()



def obtenir_matrice_pca(data, pays_nom, colonnes_quanti):
    """
    Renvoie la matrice de covariance reconstruite directement par l'objet PCA.
    """
    # 1. Filtrage et sélection numérique (comme avant)
    subset = data[data['COUNTRY'] == pays_nom][colonnes_quanti].dropna()
    
    # 2. Normalisation (Obligatoire pour l'ACP)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(subset)
    
    # 3. Calcul de la PCA
    pca = PCA()
    pca.fit(data_scaled)
    
    # 4. Extraction de la matrice via la fonction native de sklearn
    matrice = pca.get_covariance()
    
    # Transformation en DataFrame pour la lecture
    return pd.DataFrame(matrice, index=colonnes_quanti, columns=colonnes_quanti)




    
def simplifier_rsod(x):
    """simplifie les catégories de RSOD_2b (passe de 10 à 4)"""
    if x == 0: return 'Binge-Jamais' # Utilise un tiret simple ici
    elif x in [7, 8, 9]: return 'Binge-Rare'
    elif x in [4, 5, 6]: return 'Binge-Regulier'
    elif x in [1, 2, 3]: return 'Binge-Intensif'
    else: return np.nan

def recoder_f1b(x):
    """
    Recodage spécifique pour f_1b (11 modalités en 4). Regroupe les fréquences par catégories de consommation.
    """
    if x in [1, 2, 3]: return 'Freq-Quotidien'
    elif x in [4, 5, 6]: return 'Freq-Regulier'
    elif x in [7, 8, 9]: return 'Freq-Occasionnel'
    elif x in [10, 11]: return 'Freq-Abstinent'
    else: return np.nan

def calculer_cos2_individus(pca, X_scaled):
    """
    Calcule la qualité de représentation (cos2) des individus sur les axes de l'ACP.
    
    Arguments:
        pca : L'objet PCA déjà entraîné (fit)
        X_scaled : Les données numériques centrées-réduites (StandardScaler)
    """
    # 1. Coordonnées des individus sur les axes (Principal Components)
    # ind_coords shape: (n_individus, n_composantes)
    ind_coords = pca.transform(X_scaled)
    
    # 2. Calcul de la distance au carré de chaque individu à l'origine (Inertie totale)
    # On somme le carré des valeurs sur chaque ligne
    dist2 = np.sum(X_scaled**2, axis=1)
    
    # 3. Calcul du cos2 : (coordonnée sur l'axe)^2 / distance_totale^2
    # On utilise np.newaxis pour permettre la division colonne par colonne
    cos2_ind = ind_coords**2 / dist2[:, np.newaxis]
    
    # 4. Mise en forme en DataFrame
    df_cos2 = pd.DataFrame(
        cos2_ind, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    return df_cos2

def identifier_individus_typiques(df_cos2_ind, data_original, axe=1, top_n=5):
    """
    Identifie et affiche les individus qui contribuent le plus à un axe.
    """
    col_axe = f'PC{axe}'
    
    # Tri des individus par cos2 décroissant sur l'axe choisi
    # On récupère les indices des n meilleurs
    indices_top = df_cos2_ind[col_axe].sort_values(ascending=False).head(top_n).index
    
    print(f"\n--- Top {top_n} des individus les mieux représentés sur l'axe {axe} ---")
    
    # On utilise .iloc car df_cos2_ind a un index réinitialisé (0, 1, 2...) 
    # qui correspond à l'ordre des lignes du subset filtré
    return data_original.iloc[indices_top]






def fait_acm(data, pays_nom, vars_actives):
    """
    Exécute l'ACM sur un pays donné en séparant les variables actives et illustratives.
    """
    # Filtrage sur le pays cible
    subset = data[data['COUNTRY'] == pays_nom].copy()
    # Préparation des données 
    df_actives = subset[vars_actives].astype(str)
    # ACM 
    mca = prince.MCA(
    n_components=3,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42)
    mca = mca.fit(df_actives)
        
    return mca

def plot_var_acm(acm, pays_nom):
    # Récupérer les pourcentages de variance expliquée
    eigenvalues = acm.eigenvalues_
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    #barplot des % de variance expliquées par axe
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7)
    plt.xlabel('Axe')
    plt.ylabel('Pourcentage de variance expliquée (%)')
    plt.title(f'Pourcentage de variance expliquée par chaque axe (ACM {pays_nom})')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    for i, v in enumerate(explained_variance_ratio * 100):
        plt.text(i + 1, v + 0.5, f"{v:.1f}%", ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def tracer_graphe_liaisons(mca, pays_nom):
    """
    Représente la force de liaison de chaque variable avec les deux axes.
    """
    #Extraction des contributions de chaque variable à la construction des axes
    contrib = mca.column_contributions_
    
    # On regroupe les contributions des modalités d'une même variable en une seule valeur par variable(ex: SD_1homme et SD_1femme et SD_1autre)
    contrib['Var'] = [str(c).rsplit('_', 1)[0] for c in contrib.index]
    liaison = contrib.groupby('Var').sum()
    
    # graphique
    plt.figure(figsize=(9, 7))
    
    # On trace les points
    plt.scatter(liaison[0], liaison[1], c='red', s=100, edgecolors='white', linewidth=1.5)
    
    # Ajout des noms des variables
    for i, txt in enumerate(liaison.index):
        plt.annotate(txt, (liaison[0].iloc[i], liaison[1].iloc[i]), 
                     xytext=(7, 7), textcoords='offset points', fontsize=11, fontweight='bold')

    limit = max(liaison[[0, 1]].max()) * 1.1
    plt.xlim(-0.01, limit)
    plt.ylim(-0.01, limit)
    
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f"Graphe des liaisons des variables - {pays_nom}", fontsize=13)
    plt.xlabel("Liaison avec l'Axe 1")
    plt.ylabel("Liaison avec l'Axe 2 ")
    
    plt.tight_layout()
    plt.show()

def plot_individus_acm(acm, data, pays_nom, vars_actives):
    # Filtre les données pour le pays
    subset = data[data['COUNTRY'] == pays_nom].copy()

    # Récupérer les coordonnées des individus 
    coords = acm.transform(subset[vars_actives].astype(str))

    # Trace le nuage des individus
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[0], coords[1], alpha=0.5) 
    plt.xlabel('Axe 1')
    plt.ylabel('Axe 2')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'Nuage des individus (ACM {pays_nom})')
    plt.grid()
    plt.show()

