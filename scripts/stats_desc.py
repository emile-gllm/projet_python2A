import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


COULEURS_PAYS = {
    'France': '#3498db',    
    'Poland': '#e74c3c',    
    'Iceland': '#9b59b6',   
    'Bulgaria': '#2ecc71'   
}

def tableau_qualite_par_pays(df, variables_cibles, codes_refus=[]):
    """
    Génère un tableau de qualité  :
    - N : Nombre total d'observations
    - % Manquant : Valeurs nulles (NaN)
    - % Refus : Valeurs correspondant aux codes de non-réponse
    """
    print(f"{'='*50}\n RAPPORT QUALITÉ DES DONNÉES PAR PAYS\n{'='*50}")
    
    data_list = []
    
    pays_liste = df['COUNTRY'].unique()
    
    for pays in pays_liste:
        sub_df = df[df['COUNTRY'] == pays]
        n_total = len(sub_df)
        
        for var in variables_cibles:
            if var in df.columns:
                # Calcul des manquants
                n_miss = sub_df[var].isna().sum()
                pct_miss = (n_miss / n_total) * 100
                
                # refus
                n_refus = sub_df[var].isin(codes_refus).sum()
                pct_refus = (n_refus / n_total) * 100
                
                # creation de la ligne de résultat
                data_list.append({
                    'Pays': pays,
                    'Variable': var,
                    'N_Total': n_total,
                    '% Manquant (NaN)': pct_miss,
                    '% Refus (Refusal)': pct_refus
                })
    
    res_df = pd.DataFrame(data_list)
    final_table = res_df.pivot(index='Variable', columns='Pays').round(1)
    
    return final_table

def heatmap_manquants(df, variables_cibles):
    """
    Visualisation des données manquantes par pays.
    """
    # Calcul du % de manquants par Pays et Variable
    miss_data = df.groupby('COUNTRY')[variables_cibles].apply(lambda x: x.isna().mean() * 100).transpose()
    
    plt.figure(figsize=(12, len(variables_cibles) * 0.4 + 2))
    
    # Heatmap des manquants
    sns.heatmap(miss_data, annot=True, fmt=".1f", cmap="Reds", linewidths=.5, cbar_kws={'label': '% Manquant'})
    
    plt.title("Carte thermique des données manquantes (%)")
    plt.xlabel("")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

def analyse_cat(df, col, label=""):
    """
    1. Tableau : % par catégorie dans chaque pays + Effectifs.
    2. Graph : Diagramme en Barres empilées.
    """
    print(f"\n{'='*40}\n ANALYSE : {label} ({col})\n{'='*40}")
    
    #Le Tableau
    # Calcul des pourcentages par pays
    tab_pct = pd.crosstab(df['COUNTRY'], df[col], normalize='index') * 100
    # effectifs
    tab_count = pd.crosstab(df['COUNTRY'], df[col])
    
    print(">>> Tableau des Pourcentages (%) :")
    print(tab_pct.round(1))
    print("\n>>> Tableau des Effectifs (N) :")
    print(tab_count)

    #Le Graphique
    ax = tab_pct.plot(
        kind='barh', 
        stacked=True, 
        figsize=(10, 6), 
        colormap='viridis',
        edgecolor='white'
    )
    
    plt.title(f"Distribution : {label}")
    plt.xlabel("Pourcentage cumulé")
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajout des % dans les barres
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=9)
        
    sns.despine()
    plt.tight_layout()
    plt.show()

def analyse_num(df, col, label=""):
    """
    1. Tableau : Moyenne, médiane, écart-type, IQR, min/max, p10/p90.
    2. Graph : Boxplot + Densité (Distribution).
    """
    print(f"\n{'='*40}\n ANALYSE : {label} ({col})\n{'='*40}")
    
    # Le Tableau Statistique
    # Fonctions de regroupement
    def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
    def p10(x): return x.quantile(0.10)
    def p90(x): return x.quantile(0.90)
    
    # Agrégation
    stats = df.groupby('COUNTRY')[col].agg(
        ['mean', 'median', 'std', iqr, 'min', 'max', p10, p90]
    ).round(2)
    
    stats.columns = ['Moyenne', 'Médiane', 'Ecart-Type', 'IQR', 'Min', 'Max', 'P10', 'P90']
    print(stats)
    
    # Le Graphique
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    sns.boxplot(
        x='COUNTRY', y=col, data=df, ax=axes[0],
        palette=COULEURS_PAYS, showfliers=False # On cache les outliers extrêmes pour la lisibilité
    )
    axes[0].set_title(f"Boxplot : {label}")
    axes[0].set_xlabel("")
    
    # Densité
    sns.kdeplot(
        x=col, hue='COUNTRY', data=df, ax=axes[1],
        palette=COULEURS_PAYS, fill=True, alpha=0.2, linewidth=2
    )
    axes[1].set_title(f"Densité de distribution : {label}")
    axes[1].set_ylabel("Fréquence (Densité)")
    
    sns.despine()
    plt.tight_layout()
    plt.show()


def preferences_alcool(df):
    """
    Affiche la part de marché (en volume) de chaque type d'alcool par pays.
    """
    # Sélection des colonnes de consommation par type
    cols_alcool = ['cbsqf_beer', 'cbsqf_wine', 'cbsqf_spir']
    
    # Calcul de la moyenne par pays
    data_moyenne = df.groupby('COUNTRY')[cols_alcool].mean()
    
    data_pct = data_moyenne.div(data_moyenne.sum(axis=1), axis=0) * 100
    
    # Création du graphique
    ax = data_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', width=0.7)
    
    plt.title("Préférences de consommation (Part de marché en volume)")
    plt.ylabel("Pourcentage du volume total")
    plt.xlabel("")
    plt.legend(title='Type d\'alcool', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # afficher les % dans les barres
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=11)
        
    sns.despine()
    plt.show()




def analyse_sociale_croisee(df):
    """
    Heatmap montrant l'intensité du Binge Drinking 
    selon le pays et la classe sociale.
    """
    
    # On calcule la moyenne de fréquence Binge Drinking par Pays et Classe
    pivot = df.pivot_table(
        index='social_class', 
        columns='COUNTRY', 
        values='RSOD_2b', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=.5)
    
    plt.title("Le Gradient Social : Fréquence de Binge Drinking par Classe")
    plt.ylabel("Classe Sociale")
    plt.xlabel("")
    plt.show()

def ecart_hommes_femmes(df):
    """
    Visualise l'écart de volume consommé (bsqf_alc) entre hommes et femmes par pays.
    """
    plt.figure(figsize=(12, 6))
    
    
    df['Sexe_Label'] = df['SD_1'].map({1: 'Homme', 2: 'Femme', 3: 'missing'})
    sns.barplot(
        x='COUNTRY', 
        y='bsqf_alc', 
        hue='Sexe_Label', 
        data=df, 
        palette="muted", 
        errorbar=None
    )
    
    plt.title("Gender Gap : Qui consomme le plus de volume ?")
    plt.ylabel("Volume annuel moyen (cl d'alcool pur)")
    plt.xlabel("")
    plt.legend(title='Sexe')
    sns.despine()
    plt.show()

def evolution_par_age(df):
    """
    Courbe continue de la consommation selon l'âge exact.
    Permet de voir les 'pics' de consommation dans la vie.
    """
    plt.figure(figsize=(14, 7))
    
    # moyenne lissée par âge
    sns.lineplot(
        x='SD_2', 
        y='bsqf_alc', 
        hue='COUNTRY', 
        data=df, 
        palette=COULEURS_PAYS, 
        linewidth=2.5
    )
    
    plt.title("Trajectoire de vie : Évolution de la consommation selon l'âge")
    plt.xlabel("Âge du répondant")
    plt.ylabel("Volume d'alcool consommé")
    
    # Ajout de limites pour éviter les effets de bord aux âges extrêmes (ex: >80 ans)
    plt.xlim(18, 80) 
    
    sns.despine()
    plt.show()

def analyse_intensite_annuelle(df, col_vol='bsqf_alc', log_scale=False):
    """
    (a) Intensité annuelle
    - Tableau : Moyenne, Médiane, IQR, P90, P95.
    - Graphs : Boxplot + ECDF (Fonction de répartition).
    """
    print(f"\n{'='*40}\n(a) INTENSITÉ ANNUELLE ({col_vol})\n{'='*40}")
    
    # Tableau Statistiques
    def p90(x): return x.quantile(0.90)
    def p95(x): return x.quantile(0.95)
    def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
    
    
    sub_df = df.dropna(subset=[col_vol])
    
    stats = sub_df.groupby('COUNTRY')[col_vol].agg(
        ['mean', 'median', iqr, p90, p95]
    ).round(1)
    
    print(">>> Indicateurs de distribution :")
    print(stats)
    
    # Graphiques
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1 : Boxplot
    sns.boxplot(x='COUNTRY', y=col_vol, data=sub_df, palette=COULEURS_PAYS, ax=axes[0], showfliers=False)
    axes[0].set_title("Distribution des quantités (Boxplot sans outliers extrêmes)")
    if log_scale:
        axes[0].set_yscale('log')
        axes[0].set_title("Distribution (Échelle Log)")
    
    # Graphique 2 : Courbe ECDF
    sns.ecdfplot(data=sub_df, x=col_vol, hue='COUNTRY', palette=COULEURS_PAYS, linewidth=2, ax=axes[1])
    axes[1].set_title("Courbe cumulative (ECDF)")
    axes[1].set_xlabel("Volume annuel")
    axes[1].set_ylabel("Proportion cumulée de la population")
    
    sns.despine()
    plt.tight_layout()
    plt.show()



def analyse_composition(df):
    """
    Composition des boissons (Mix)
    - Calcule les parts (share_beer, etc.)
    - Graph : Barres des parts moyennes par pays.
    """
    print(f"\n{'='*40}\n(c) COMPOSITION DU PANIER (Mix Alcool)\n{'='*40}")
    
    # 1. Calcul des parts individuelles
    cols = ['cbsqf_beer', 'cbsqf_wine', 'cbsqf_spir']
    
    # Somme totale par individu
    df['total_vol_calc'] = df[cols].sum(axis=1)
    
    # filtre
    total_safe = df['total_vol_calc'].replace(0, np.nan)
    
    df['Share_Beer'] = df['cbsqf_beer'] / total_safe
    df['Share_Wine'] = df['cbsqf_wine'] / total_safe
    df['Share_Spir'] = df['cbsqf_spir'] / total_safe
    
    # 2. Calcul des Moyennes par Pays (Le "Mix Moyen")
    mix_moyen = df.groupby('COUNTRY')[['Share_Beer', 'Share_Wine', 'Share_Spir']].mean() * 100
    
    print(">>> Mix Moyen (%) par Pays :")
    print(mix_moyen.round(1))
    
    # 3. Graphique (Barres Simples )
    ax = mix_moyen.plot(kind='bar', stacked=True, color=['gold', '#800020', 'grey'], figsize=(10, 6))
    
    plt.title("Préférences culturelles : Part de marché moyenne des alcools")
    plt.ylabel("Part moyenne (%)")
    plt.xticks(rotation=0)
    plt.legend(["Bière", "Vin", "Spiritueux"], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajout des étiquettes
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=10, weight='bold')
        
    sns.despine()
    plt.tight_layout()
    plt.show()

def analyse_rsod_frequence(df, col_rsod='RSOD_2b'):
    """
    Fréquence du Binge Drinking (RSOD_2b)
    Graphique a Barres empilées.
    """
    print(f"\n{'='*40}\n(d-1) FRÉQUENCE BINGE DRINKING ({col_rsod})\n{'='*40}")
    
    mapping_rsod = {
        1: 'Every day', 2: '5-6 days a week',   
        3: '3-4 days a week',                
        4: '1 - 2 days a week', 5: '2-3 days a month',      
        6: '1 day in a month',            
        7: '6-11 days a year',
        8: '2-5 days a year',
        9: 'a single day in the past 12 months',
        0: 'I did not drink in the past 12 months'
    }
    
    # Création de la variable recodée
    col_recoded = 'RSOD_Simple'
    df[col_recoded] = df[col_rsod].map(mapping_rsod).fillna(df[col_rsod])
    
    # Tableau Croisé (% par Pays)
    tab = pd.crosstab(df['COUNTRY'], df[col_recoded], normalize='index') * 100
    
    
    print("Distribution des fréquences RSOD (%) :")
    print(tab.round(1))
    
    # Graphique Empilé
    ax = tab.plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6), 
        colormap='RdYlGn_r',
        edgecolor='white',
        width=0.75
    )
    
    plt.title("Fréquence du Binge Drinking (6+ verres) par Pays")
    plt.ylabel("% Population")
    plt.legend(title='Fréquence', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Labels
    for c in ax.containers:
        # On n'affiche que si la barre est assez grande (>5%
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='black', fontsize=9)
        
    sns.despine()
    plt.tight_layout()
    plt.show()

def analyse_lieux_rsod(df):
    """
    Contextes du Binge Drinking
    - Graphique : 4 panneaux (Small Multiples) avec barres empilées.
    """
    print(f"\n{'='*40}\n(d-2) LIEUX DU BINGE DRINKING (Contextes)\n{'='*40}")
    
    # Liste des variables lieux
    lieux_vars = {
        'RSOD_5a': 'Domicile (Home)',
        'RSOD_5b': 'Chez Amis',
        'RSOD_5c': 'Bar/Resto/Club',
        'RSOD_5d': 'Plein Air/Public'
    }
    # Création des sous-graphes
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (col, label) in enumerate(lieux_vars.items()):
        if col in df.columns:
            # Tableau croisé
            tab = pd.crosstab(df['COUNTRY'], df[col], normalize='index') * 100
            
            # Plot sur le sous-graphique (ax=axes[i])
            tab.plot(kind='bar', stacked=True, ax=axes[i], colormap='Blues', legend=False)
            
            axes[i].set_title(label, fontweight='bold')
            axes[i].set_ylabel("%")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis='x', rotation=0)

    # Légende commune
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Fréquence', loc='center right', bbox_to_anchor=(1.08, 0.5))
    
    plt.suptitle("Fréquence du Binge Drinking selon le Lieu", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()

    print("Calcul des Indicateurs de Synthèse...")
    
    # conversion en numeric
    cols_num = ['RSOD_5a', 'RSOD_5c']
    for c in cols_num:
        df[c+'_num'] = pd.to_numeric(df[c], errors='coerce')
    
    # Indicateur 1 : Home Drinking Dominance (% des gens qui boivent PLUS SOUVENT chez eux qu'au bar)
    # Si Code(Home) < Code(Bar), alors Fréq(Home) > Fréq(Bar) (si 1=Daily)
    df['is_home_dominant'] = df['RSOD_5a_num'] < df['RSOD_5c_num']
    
    # Indicateur 2 : Social Drinking Index (Ratio simple Bar / Home au niveau agrégé)
    # % qui boivent au moins mensuellement au Bar vs Home.
    
    # tableau récapitulatif par pays
    res_synth = df.groupby('COUNTRY').agg({
        'is_home_dominant': 'mean', # Donne le % de dominance domicile
    }) * 100
    
    res_synth.columns = ['% Dominance Domicile (Home > Bar)']
    print(res_synth.round(1))
    
    # graph pour la dominance 
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=res_synth.index, 
        y='% Dominance Domicile (Home > Bar)', 
        data=res_synth, 
        palette=COULEURS_PAYS
    )
    plt.title("Où le Binge Drinking est-il le plus ancré ?\n(% de la population buvant plus souvent à domicile qu'au bar)")
    plt.ylabel("% Dominance Domicile")
    plt.ylim(0, 100)
    sns.despine()
    plt.show()

