import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

country_mapping = {
    10: 'Austria',
    11: 'Belgium',
    12: 'BosniaHerzegovina',
    13: 'Bulgaria',
    14: 'Catalunya',
    15: 'Croatia',
    16: 'Cyprus',
    17: 'Czech Republic',
    18: 'Denmark',
    19: 'Estonia',
    20: 'Finland',
    21: 'France',
    22: 'Germany',
    23: 'Greece',
    24: 'Hungary',
    25: 'Iceland',
    26: 'Ireland',
    27: 'Italy',
    28: 'Latvia',
    29: 'Lithuania',
    30: 'Luxembourg',
    31: 'Malta',
    32: 'Moldova',
    33: 'Netherlands',
    34: 'Norway',
    35: 'Poland',
    36: 'Portugal',
    37: 'Romania',
    38: 'Serbia',
    39: 'Slovakia',
    40: 'Slovenia',
    41: 'Spain',
    42: 'Sweden',
    43: 'United Kingdom',
}

def barplot_bycntry(data, colonne_y='sd_20month', y_name= "revenu mensuel moyen par pays (unité inconnue)", titre="Revenu Mensuel Moyen par Pays", source ="Source : [The Standard European Alcohol Survey – Wave 2]\nChamp : [33 pays européens]"):
    """
    Calcule la moyenne d'une variable par pays, trie les résultats 
    et affiche un barplot.
    """
    datalc = data.groupby('COUNTRY')[colonne_y].mean().sort_values(ascending=True).reset_index()
    
    # Création de la figure
    plt.figure(figsize=(10,8))
    sns.barplot(x='COUNTRY', y=colonne_y, data=datalc)

    plt.title(titre, fontsize=14)
    plt.xlabel("Pays", fontsize=12)
    plt.ylabel(f"Moyenne de {y_name}", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.figtext(0.1, -0.1, source , fontsize=10)
    plt.tight_layout()
    plt.show()



# Définir les taux de conversion (Unité Locale -> EUR) au 02/01/2020.
# Taux de change officiels du Journal officiel (BCE)
conversion_rates = {
    # Zone Euro (Taux fixe = 1)
    'Austria': 1.0, 'Belgium': 1.0, 'Cyprus': 1.0, 'Estonia': 1.0, 
    'Finland': 1.0, 'France': 1.0, 'Germany': 1.0, 'Greece': 1.0, 
    'Ireland': 1.0, 'Italy': 1.0, 'Latvia': 1.0, 'Lithuania': 1.0,
    'Luxembourg': 1.0, 'Malta': 1.0, 'Netherlands': 1.0, 'Portugal': 1.0, 
    'Slovakia': 1.0, 'Slovenia': 1.0, 'Spain': 1.0, 'Catalunya': 1.0,

    # Taux du Journal Officiel de l'UE
    'United Kingdom': 1 / 0.84828,   # GBP
    'Denmark': 1 / 7.4719,           # DKK
    'Sweden': 1 / 10.4728,          # SEK
    'Iceland': 1 / 136.90,          # ISK
    'Norway': 1 / 9.8408,           # NOK
    'Bulgaria': 1 / 1.9558,          # BGN
    'Czech Republic': 1 / 25.411,    # CZK
    'Hungary': 1 / 329.98,          # HUF
    'Poland': 1 / 4.2544,           # PLN
    'Romania': 1 / 4.7828,          # RON
    'Croatia': 1 / 7.4445,          # HRK
    
    # Autres pays (hors liste BCE -source = Google finance)
    'BosniaHerzegovina': 1 / 1.95583, # BAM 
    'Moldova': 1 / 19.33,            # MDL
    'Serbia': 1 / 117.85             # RSD
}



# Valeurs brutes de parité (PPP)
# Source : Eurostat 2020 (Table prc_ppp_ind_1)
# https://ec.europa.eu/eurostat/databrowser/view/prc_ppp_ind_1/default/table?lang=fr
# (on a complété certains pays avec les valeurs de la Banque Mondiale)
ppa_brut = {
    'Austria': 1.14085,
    'Belgium': 1.12107,
    'BosniaHerzegovina': 0.94,      # Mark convertible (BAM)
    'Bulgaria': 1.05695,            # Lev (BGN)
    'Catalunya': 0.94905,           # Euro (Base Espagne)
    'Croatia': 4.75327,             # Kuna (HRK - en 2020) -> sur ta photo: 0.64 pour l'indice
    'Cyprus': 0.90564,              # Euro
    'Czech Republic': 19.0198,      # Couronne (CZK) - Vu sur ta photo
    'Denmark': 9.96376,             # Couronne (DKK) - Vu sur ta photo
    'Estonia': 0.80649,             # Euro
    'Finland': 1.25870,             # Euro
    'France': 1.08489,              # Euro
    'Germany': 1.10647,             # Euro
    'Greece': 0.83004,              # Euro
    'Hungary': 222.032,             # Forint (HUF) - Vu sur ta photo
    'Iceland': 185.20,              # Couronne (ISK)
    'Ireland': 1.21387,             # Euro
    'Italy': 0.98978,               # Euro
    'Latvia': 0.73521,              # Euro
    'Lithuania': 0.67851,           # Euro
    'Luxembourg': 1.31205,          # Euro
    'Malta': 0.87277,               # Euro
    'Moldova': 7.50,                # Leu moldave (MDL)
    'Netherlands': 1.17064,         # Euro
    'Norway': 13.95,                # Couronne (NOK)
    'Poland': 2.65831,              # Zloty (PLN) - Vu sur ta photo
    'Portugal': 0.84971,            # Euro
    'Romania': 2.52753,             # Leu (RON) - Vu sur ta photo
    'Serbia': 62.2403,                # Dinar (RSD)
    'Slovakia': 0.7321,             # Euro
    'Slovenia': 0.8642,             # Euro
    'Spain': 0.94905,               # Euro
    'Sweden': 11.20,                # Couronne (SEK)
    'United Kingdom': 0.85,         # Livre (GBP)
}




def tableau_na(data):
    """

    Génère un tableau Pandas des valeurs manquantes.
    """
    # Calcul 
    df_na = data.isna().mean().to_frame(name='Taux de manquants').reset_index()
    df_na.columns = ['Variable', 'Taux de manquants']
    
    # Création du tableau 
    tableau = (df_na.style
               .format({'Taux de manquants': '{:.2%}'})
               .set_caption("TABLEAU RÉCAPITULATIF DES DONNÉES MANQUANTES<br>"
                            "Source : The Standard European Alcohol Survey – Wave 2 | Champ : 33 pays européens")
               .hide(axis='index')
               .set_table_styles([{'selector': 'caption', 'props': [('color', 'color'), ('font-size', '14px'), ('font-weight', 'bold')]}])
              )
    
    return tableau

def visu_na(data):
    """ Visualisation via heatmap des valeurs manquantes """
    plt.title("Visualisation des Valeurs Manquantes\nSource : SEAS-2 | Champ : 33 pays européens", fontsize=14)
    sns.heatmap(data.isna(), cbar=False)
    plt.show()



def tableau_na_pays(data, var="bsqf_alc"):
    """
    Renvoie un tableau du taux de NA par pays pour une variable (ou toutes).
    """
    # Calcul des moyennes de NA par Pays
    df_missing = data.groupby('COUNTRY').apply(lambda x: x.isna().mean(), include_groups=False)
    
    # Sélection de la variable d'intérêt ou tout le tableau
    resultat = df_missing[[var]] if var in df_missing.columns else df_missing

    titre = f"TAUX DE VALEURS MANQUANTES PAR PAYS ({var if var else 'Toutes variables'})"
    source_champ = "Source : SEAS-2 | Champ : 33 pays européens"
    
    tableau = (resultat.style
               .set_caption(f"<b>{titre}</b><br>{source_champ}")
               .background_gradient(cmap='Reds') # Colore les cellules selon le taux
               .set_table_styles([{'selector': 'caption', 'props': [('font-size', '14px')]}]))
    
    return tableau


def trace_table_contingence(data, var_x='RSOD_2b', var_y='f_1b'):
    """
    Génère une table de contingence et affichage des totaux.
    Note : On remplace les manquants par 0.
    """
    # Création de la table de contingence
    ct_f = pd.crosstab(data[var_y], data[var_x], margins=True, margins_name="Total").fillna(0)
    
    # Séparation de la table et des totaux
    core = ct_f.iloc[:-1, :-1]
    r_tot = ct_f.iloc[:-1, -1]  # Totaux lignes
    c_tot = ct_f.iloc[-1, :-1]  # Totaux colonnes
    g_tot = ct_f.iloc[-1, -1]   # Total général

    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Affichage de la table (gradient de couleur)
    im = ax.imshow(core, cmap='YlGnBu', aspect='auto')
    plt.colorbar(im, ax=ax, pad=0.1).set_label("Effectifs")

    ax.set_xticks(np.arange(len(core.columns)))
    ax.set_yticks(np.arange(len(core.index)))
    ax.set_xticklabels(core.columns)
    ax.set_yticklabels(core.index)

    # Boucle pour afficher le nombre d'observations (i: ligne, j: colonne)
    thresh = core.values.max() / 2.
    for i, j in np.ndindex(core.shape):
        val = core.iloc[i, j]
        ax.text(j, i, int(val), ha="center", va="center", 
                color="white" if val > thresh else "black")

    # Affichage des totaux 
    ax.text(len(core.columns), -0.7, "TOTAL", ha="center", fontweight='bold')
    ax.text(-0.7, len(core.index), "TOTAL", ha="right", va="center", fontweight='bold')

    for i, v in enumerate(r_tot):
        ax.text(len(core.columns), i, int(v), ha="center", va="center", fontweight='bold')
    for j, v in enumerate(c_tot):
        ax.text(j, len(core.index), int(v), ha="center", va="center", fontweight='bold')

    plt.figtext(0.1, -0.08, "Source : [The Standard European Alcohol Survey – Wave 2]\nChamp : [33 pays européens] \n Lecture: La probabilité qu'un individu ait consommé plus de 4/6 verres d'alcool un seul jour par an\nsachant qu'il a consommé de l'alcool un seul jour par an est de 811/939 (case(1,1)/total de la colonne) = 0.86.", fontsize=10)
   
    plt.title(f"Table de contingence : {var_y} vs {var_x}", pad=25)
    plt.xlabel(f"Codes {var_x}", labelpad=25)
    plt.ylabel(f"Codes {var_y}", labelpad=25)
    
    plt.tight_layout()
    plt.show()






def trace_camemberts_na(data, var_na='RSOD_2bisna', var_groupe='SD_1', 
                         dict_labels=None, titre_general="", source_texte=""):
    """
    Génère des camemberts pour la répartition des NA 
    selon les modalités d'une variable.
    """
    # Nettoyage des données pour le groupe (on ignore les NA de la variable de groupe)
    df_plot = data.dropna(subset=[var_groupe])
    modalites = sorted(df_plot[var_groupe].unique())
    
    fig, axes = plt.subplots(1, len(modalites), figsize=(4 * len(modalites), 6))
    
    # Si une seule modalité, axes n'est pas une liste, on le transforme
    if len(modalites) == 1:
        axes = [axes]

    # Paramètres esthétiques
    colors = ['#66b3ff', '#ff9999'] # Bleu pour Répondant, Rose pour Manquant
    labels_na = {0: 'Répondant', 1: 'Manquant'}

    for i, mod in enumerate(modalites):
        subset = df_plot[df_plot[var_groupe] == mod]
        counts = subset[var_na].value_counts().sort_index()
        
        # Création du camembert
        axes[i].pie(
            counts, 
            labels=[labels_na.get(x, x) for x in counts.index],
            autopct='%1.1f%%', 
            startangle=140,
            colors=colors,
            pctdistance=0.85, # Place le % vers l'extérieur
            labeldistance=1.1  # Place le texte (Répondant/Manquant) à l'extérieur
        )
        
        # Titre spécifique à chaque camembert
        nom_modalite = dict_labels.get(mod, f"Groupe {mod}") if dict_labels else f"Groupe {mod}"
        axes[i].set_title(nom_modalite, fontweight='bold', pad=20)

    # titre général
    plt.suptitle(titre_general, fontsize=16, fontweight='bold', y=1.05)
    
    plt.figtext(0.1, -0.1, source_texte, fontsize=10, ha='left', linespacing=1.5)
    
    plt.tight_layout()
    plt.show()




def trace_barplot_na(data, var_quanti='bsqf_alc', var_na='RSOD_2bisna', 
                              nb_groupes=10, source_texte=""):
    """
    Découpe une variable quantitative en quantiles
    et affiche la part de NA pour chaque groupe.
    """
    df_temp = data.copy()
    # Création des labels (D1, D2...)
    labels = [f'D{i}' for i in range(1, nb_groupes + 1)]
    
    # Découpage en quantiles
    df_temp['groupe_quanti'] = pd.qcut(df_temp[var_quanti], nb_groupes, 
                                      labels=labels, duplicates='drop')
    
    # Mapping du statut
    df_temp['statut_na'] = df_temp[var_na].map({0: 'Répondant', 1: 'Non répondant'})
    
    # Calcul des statistiques (table de contingence normalisée)
    tab = pd.crosstab(df_temp['groupe_quanti'], df_temp['statut_na'], normalize='index') * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    tab.plot(kind='bar', stacked=True, ax=ax, color=['#66b3ff', '#ff9999'], width=0.7)

    plt.title(f"Réponse à {var_na} selon les déciles de {var_quanti}", fontsize=15, pad=20, fontweight='bold')
    plt.xlabel(f"Déciles de {var_quanti} (du plus faible au plus élevé)", fontsize=12)
    plt.ylabel("Répartition (%)", fontsize=12)
    plt.legend(title='Statut', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Ajout des étiquettes de pourcentage sur les barres
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 3: # On n'affiche que si la barre est assez grande
            ax.text(x + width/2, y + height/2, f'{height:.1f}%', 
                    ha='center', va='center', fontsize=9, fontweight='bold', color='black')


    plt.figtext(0.1, -0.05, source_texte, fontsize=10, ha='left', linespacing=1.5)
    
    plt.tight_layout()
    plt.show()





def tableau_effectifs_pays(data):
    """
    Génère un tableau du nombre d'observations par pays.
    """
    # Calcul des effectifs
    df_count = data.groupby('COUNTRY').size().to_frame(name='Nombre d\'observations')
    
    # Tri décroissant des effectifs
    df_count = df_count.sort_values(by='Nombre d\'observations', ascending=False)


    titre = "RÉPARTITION DU NOMBRE DE RÉPONDANTS PAR PAYS"
    source_champ = "Source : SEAS-2 | Champ : 33 pays européens"
    
    return (df_count.style
            .format("{:,.0f}")
            .background_gradient(cmap='Blues')
            .set_caption(f"<b>{titre}</b><br>{source_champ}"))






def trace_carte_europe(data, europe, colonne, titre, label_legende, texte_info):
    """
    Fonction générique pour tracer une carte de l'Europe avec une variable donnée.
    """
    # Calcul des stats et fusion fond de carte/ données
    stats = data.groupby('COUNTRY')[colonne].mean().reset_index()
    map_data = europe.merge(stats, left_on='NAME', right_on='COUNTRY', how='left')
    
    vmax = map_data[colonne].quantile(0.95) # (quantile 95)- pour ne pas avoir outliers qui soient les seuls pays colorés - réduit l'échelle possible du gradient de couleur)

    fig, ax = plt.subplots(figsize=(15, 10))
    europe.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.5)
    
    map_data.dropna(subset=[colonne]).plot(
        column=colonne, ax=ax, legend=True, cmap='YlOrRd', 
        vmax=vmax, edgecolor='black', linewidth=0.3,
        legend_kwds={'label': label_legende, 'orientation': "horizontal", 'pad': 0.02, 'shrink': 0.7}
    )

    
    plt.figtext(0.15, 0.05, texte_info, fontsize=10, ha='left', linespacing=1.4)
    
    ax.set_xlim(-25, 45)
    ax.set_ylim(33, 72)
    ax.set_title(titre, fontsize=16, pad=20)
    ax.axis('off')
    plt.show()

def compare_cartes_ppa(data, europe, col_nom, col_ppa, texte_a, texte_b):
    """
     Fonction spécifique pour la double carte Nominal vs PPA.
    """
    stats = data.groupby('COUNTRY')[[col_nom, col_ppa]].mean().reset_index()
    map_data = europe.merge(stats, left_on='NAME', right_on='COUNTRY', how='left')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    vmax_commun = map_data[col_nom].quantile(0.95)
    
    style = {'cmap': 'YlOrRd', 'edgecolor': 'black', 'linewidth': 0.3, 
             'legend': True, 'vmax': vmax_commun, 
             'legend_kwds': {'orientation': "horizontal", 'pad': 0.02, 'shrink': 0.6}}

    # Carte A - nominal
    europe.plot(ax=ax1, color='#f0f0f0', edgecolor='white')
    map_data.dropna(subset=[col_nom]).plot(column=col_nom, ax=ax1, **style)
    ax1.set_title("A. Salaire mensuel moyen (Euro Nominal)", fontsize=15)
    ax1.text(0.1, -0.3, texte_a, transform=ax1.transAxes, fontsize=10, ha='left', linespacing=1.4)

    # Carte B -PPA
    europe.plot(ax=ax2, color='#f0f0f0', edgecolor='white')
    map_data.dropna(subset=[col_ppa]).plot(column=col_ppa, ax=ax2, **style)
    ax2.set_title("B. Salaire mensuel moyen (Euro PPA)", fontsize=15)
    ax2.text(0.1, -0.3, texte_b, transform=ax2.transAxes, fontsize=10, ha='left', linespacing=1.4)

    for ax in [ax1, ax2]:
        ax.set_xlim(-25, 45); ax.set_ylim(33, 72); ax.axis('off')
    
    plt.show()


