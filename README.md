# projet_python2A

# Profils des consommateurs d'alcools européens

## Objectif du projet
De nombreuses études semblent s'être intéressées à déterminer économétriquement les déterminants socio-économiques, voire psychologiques de la consommation d'alcool dans certains pays, notamment en Europe où la consommation est la plus élevée au monde. Ces études se concentrent souvent sur un pays en particulier à l'instar de l'étude menée en Angleterre (PLOS ONE - 2018) qui démontre notamment que si les groupes socio-économiques les plus aisés boivent plus fréquemment, les populations les plus défavorisées ont une probabilité bien plus élevée d'avoir une consommation excessive et nocive. Une étude menée sur des données tchèques en 2015 (Econstor) identifie elle, par exemple, que le sexe et le tabagisme sont des prédicteurs majeurs de l'abus d'alcool. Nous nous sommes alors demandé si ce genre de résultat était généralisable à tous les pays européens. En utilisant le 2021 Standard European Alcohol Survey (Deepseas), nous nous proposons donc de répondre à la question  suivante: les déterminants socio-économiques et psychologiques de la consommation d’alcool sont-ils semblables en Europe, ou leur poids varie-t-il selon les contextes nationaux et les profils de risque ?

## Structure du répertoire

Le répertoire est constitué :
- d'un dossier "Data" avec le codebook et un guide sur le codage de certaines variables (la base de données n'est pas mise en accès libre ici);
- d'un notebook où sont appelées des fonctions et où l'étude statistique est commentée;
- d'un dossier "Scripts" où sont enregistrés les fichiers contenant les fonctions;
- D'un fichier requirements.txt où sont listées les librairies à importer.

NB:(Pour exécuter le notebook, vous devez utiliser le lien transmis comme url).

## Récupération des données

Nos données sont issues de la base Deepseas wave 2, étude commandée par la Commission européenne en 2019. Au moment de la rédaction de cette présentation, le site sur lequel nous avons pris nos données ne semble plus accessible (https://www.deep-seas.eu). Vous pouvez cependant retrouver la page de l'archive du site sur Google Cached via ce même lien. 

Nous avons dû ensuite nettoyer la base en recodant des variables mal codées ou dans des unités non appropriées.

## Statistiques descriptives et traitement des valeurs manquantes

Après une analyse non exhaustive sur le plan européen (partie I du notebook), nous sélectionnons des pays qui nous semblent pouvoir donner différents profils de consommateurs d'alcool. Nous faisons des statistiques descriptives sur ceux-ci (partie II - A) et repérons les imputations les plus adaptées pour quelques variables contenant de nombreuses valeurs manquantes. 
Nous imputons alors les variables qui en valent le coup soit par la méthode KNN soit par la méthode MICE (partie II - B).
Nous essayons alors de cerner des profils d'individus en utilisant des ACP et des ACM par pays.

## Modélisation
Enfin, pour chaque pays choisi, nous essayons de modéliser la consommation d'alcool des individus selon leurs caractéristiques socio-économico-psychologiques via des régressions linéaires (Partie III). 

