# P7_HOME_SCORING
# OpenClassRooms_Projet7
#### <i>Implémentez un modèle de scoring</i>

## Présentation
La mission principale de ce projet est de prédire le risque de faillite d'un client pour une société de crédit. Pour cela, nous devons configurer un modèle de classification binaire et d'en analyser les différentes métriques.

Ce projet consiste à créer une API web avec un Dashboard interactif. Celui-ci devra a minima contenir les fonctionnalités suivantes :

 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

La partie features engineering n'étant pas la partie la plus importante de ce projet, il nous est proposé d'utiliser un Notebook disponible sur le site de Kaggle, dont le lien est disponible au bas de cette présentation.

## Construction

<u>Dans ce dépôt, vous trouverez :</u>

 - Un dossier avec le Notebook Jupyter pour l'étude des données, l'entraînement et la configuration du modèle de classification.
 - Un dossier avec la note technique qui explique en détails la construction et les résultats du modèle.
 - Un dossier avec la configuration locale de l'API. Dans le but de comprendre le fonctionnement de Flask, cette "version" de l'API s'appuie sur deux fichiers .py :
    - API.py qui est le fichier Flask contenant la partie backend.
    - DASHBOARD.py contient la partie Frontend codée avec Streamlit.
 - Dans la branche principale, la configuration déployée sur Azure. Cette version n'utilise que Streamlit qui a l'avantage d'être très simple à coder

## Modèle de classification
Le modèle retenu pour cet exercice est le modèle LGBMC. Le meilleur score (AUC) obtenu est de 0.78.
XGboost a obtenu un très leger meilleur score, mais est beaucoup plus lent.
Les prédictions obtenues avec le model sont assez déviantes (cf simulation d'écoulement des données /datadrift).
La caractéristique majeure de ces données sont le fort déséquilibrage des classes.
En utilisant, une rééquilibration artificielle (avec smote) intégré à une validation croisée, j ai pu sensiblement améliorer les prédictions.
Le modèle prédit plus de non solvables que leur attribution réelle. 
Le modèle pourrait être amélioré de deux manières:
 - optimisation des paramètres du model  
 - optimisation du choix des variables

## Dashboard

L'application est un **dashboard Streamlit autonome** (`main.py`) : prédiction et
valeurs SHAP sont calculées en local, directement à partir du modèle sérialisé.

> Historique : une première version séparait un backend **Flask** (API de scoring,
> `app.py`) d'un frontend **Streamlit** (`main.py`), le backend étant hébergé sur
> Heroku. L'offre gratuite Heroku ayant disparu, le backend n'est plus joignable.
> Le dashboard embarque désormais le modèle et se passe de toute API externe.

### Lancer en local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

### Déploiement (Streamlit Community Cloud)
1. Pousser le repo sur GitHub.
2. Sur share.streamlit.io : **New app** → sélectionner le repo et la branche.
3. **Main file path : `main.py`**.
4. Déployer. `runtime.txt` fixe Python 3.10, `packages.txt` installe `libgomp1`
   (requis par LightGBM) et `requirements.txt` liste les
   dépendances (versions alignées sur l'entraînement du modèle, scikit-learn 1.2.1).

Fichiers requis au runtime (versionnés dans le repo) : `main.py`, `LGBMC2.zip`
(modèle), `df_trainXY_sample.csv` (features des clients), `df_dashboardY.zip`
(données d'affichage), `SHAP_summary_GBC.png` (importance globale).

> `app.py`, `Procfile` et `setup.sh` sont des reliquats de l'ancienne configuration
> Heroku/Flask : inutiles pour Streamlit Cloud, conservés à titre d'archive.

## Données d'entrées
 - Lien de téléchargement des données d'entrées : https://www.kaggle.com/c/home-credit-default-risk/data 
 - Notebook de départ pour la partie Features Engineering : https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
