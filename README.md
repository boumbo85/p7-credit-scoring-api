# P7 — Credit Risk Scoring
#### OpenClassrooms · Implémentez un modèle de scoring

🔗 **Démo en ligne** : https://boumbo85-lw65qk7fnzxeebpfssaajo.streamlit.app

> Scoring de risque de défaut de crédit, avec explication SHAP par client, servi
> par un dashboard Streamlit autonome.

## Présentation
La mission principale de ce projet est de prédire le risque de faillite d'un client pour une société de crédit. Pour cela, nous devons configurer un modèle de classification binaire et d'en analyser les différentes métriques.

Ce projet consiste à créer une API web avec un Dashboard interactif. Celui-ci devra a minima contenir les fonctionnalités suivantes :

 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

La partie features engineering n'étant pas la partie la plus importante de ce projet, il nous est proposé d'utiliser un Notebook disponible sur le site de Kaggle, dont le lien est disponible au bas de cette présentation.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `main.py` | Dashboard Streamlit autonome (score + SHAP calculés en local) |
| `LGBMC2_calibrated.zip` | Modèle **calibré** (isotonic) servi par l'app |
| `LGBMC2.zip` | Modèle LightGBM brut d'origine (pipeline `imbalanced-learn`) |
| `df_trainXY_sample.csv` | Échantillon de 400 clients avec leurs features (prédiction) |
| `df_dashboardY.zip` | Données descriptives des clients (affichage) |
| `SHAP_summary_GBC.png` | Importance globale des variables (SHAP) |
| `requirements.txt`, `runtime.txt`, `packages.txt` | Configuration de déploiement |
| `app.py`, `Procfile`, `setup.sh` | Archive de l'ancienne version Flask/Heroku (non utilisés) |

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

## Démarche

### 1. Problématique
Un organisme de crédit veut estimer la probabilité qu'un client fasse défaut
sur son prêt, et surtout **rendre chaque décision explicable** à un conseiller
non data scientist. L'enjeu métier n'est pas symétrique : accorder un crédit à
un futur défaillant (faux négatif) coûte bien plus cher que refuser un bon
client (faux positif). Le modèle doit donc être optimisé pour ce coût, pas pour
l'exactitude brute.

### 2. Méthodologie
- **Données** : jeu Home Credit Default Risk (Kaggle), ~307 000 clients, fortement
  déséquilibré (~8 % de défauts).
- **Feature engineering** : agrégation des tables annexes (bureau, previous
  applications…), encodage one-hot, ratios métier (taux d'endettement, LTV,
  apport). ~166 variables en entrée du modèle.
- **Rééquilibrage** : `SMOTE` (sur-échantillonnage de la classe minoritaire)
  combiné à un `RandomUnderSampler`, dans une pipeline `imbalanced-learn` pour
  éviter toute fuite de données en validation croisée.
- **Modèle** : `LightGBM`, comparé à XGBoost (score très légèrement supérieur
  mais nettement plus lent) et à des baselines linéaires.
- **Métrique de sélection** : `Fβ` avec **β = 10**, qui pénalise fortement les
  faux négatifs — cohérent avec le coût métier.
- **Calibration** : `CalibratedClassifierCV` (isotonic) sur un jeu non rééquilibré,
  pour corriger les probabilités gonflées par SMOTE (voir Résultats).
- **Seuil de décision** : optimisé sur la matrice de coût (FN:FP = 10:1) plutôt que
  fixé à 0,5, appliqué aux probabilités **calibrées**.
- **Explicabilité** : `SHAP` (TreeExplainer) pour l'importance globale, l'explication
  **par client** (waterfall) et une **lecture métier** en langage naturel.
- **Suivi d'expériences** : `MLflow` pendant la phase de sélection de modèle
  (tracking des runs, métriques et paramètres).

### 3. Challenges
- Transformer un déséquilibre de classes en **décision métier** : rééquilibrer à
  l'entraînement, puis caler le seuil sur le coût réel d'un faux négatif.
- Rendre l'explicabilité **lisible pour un non-expert** (waterfall SHAP par client,
  pas seulement une importance globale).
- Industrialiser : passer d'un notebook à une application déployée et maintenable.

### 4. Résultats
- **Calibration** : score de Brier **0,55 → 0,07** après recalibration isotonic.
  La probabilité moyenne passe de ~0,77 à ~0,08, soit exactement le taux de défaut
  réel — les probabilités sont désormais interprétables.
- **Pouvoir discriminant** : AUC-ROC ≈ 0,62 et AUC-PR ≈ 0,12 sur un échantillon
  représentatif (la calibration ne change pas le classement, seulement l'échelle).
  Au seuil de coût retenu : ~66 % acceptés / ~34 % refusés, recall défaut ~0,50.
- Dashboard interactif : jauge de probabilité **avec le seuil de décision**, verdict,
  explication SHAP locale, lecture métier et informations descriptives du client.
- Application **déployée et accessible en ligne** (Streamlit Community Cloud).

### 5. Interprétation
Les variables les plus influentes rejoignent l'intuition métier : sources de
revenu, taux d'endettement, montant et ancienneté du crédit, historique externe.
Le waterfall SHAP permet à un conseiller de voir, pour **un** client donné, quelles
variables ont poussé le score vers le défaut ou l'acceptation.

### 6. Limites
- **Pouvoir discriminant modéré** : AUC-ROC ~0,62 sur données réelles. La calibration
  rend les probabilités fiables mais ne crée pas de signal — pour aller plus loin, il
  faudrait enrichir les variables (sources externes, historique) et ré-entraîner.
- **Data drift** : les simulations d'écoulement des données montrent une dérive des
  prédictions dans le temps → un ré-entraînement périodique serait nécessaire en
  production.
- **Variables protégées** : le modèle utilise encore genre, âge et situation
  familiale — exclues de l'explication affichée, mais à retirer du modèle en
  production (voir § Fairness & conformité).
- **Démo** : l'app embarque un échantillon de 400 clients (à la vraie distribution)
  pour rester légère.

### 7. Pistes d'amélioration (avec plus de temps)
- Retirer les variables protégées du modèle, ré-entraîner et mesurer l'équité
  entre groupes (fairness) — cf. § Fairness & conformité.
- Optimiser les hyperparamètres (Optuna) et la sélection de variables pour gagner en
  pouvoir discriminant.
- Ajouter une surveillance du drift (evidently) et un ré-entraînement planifié.
- Exposer une vraie API de scoring (FastAPI) si un découplage front/back redevient utile.

### 8. Retour critique
Le point le plus formateur a été de comprendre que **l'exactitude n'est pas la
bonne boussole** sur un problème déséquilibré à coût asymétrique, et que **SMOTE
règle l'apprentissage mais casse la calibration**. La version actuelle sépare donc
clairement les trois temps : rééquilibrer (apprentissage), **calibrer** (isotonic,
pour des probabilités fiables), puis **choisir le seuil** sur le coût métier
(décision). C'est cette chaîne, plus que l'AUC, qui fait la valeur du projet.

### 9. Fairness & conformité
Le jeu de données contient des **variables protégées** — genre (`CODE_GENDER`),
âge (`DAYS_BIRTH`) et situation familiale (`NAME_FAMILY_STATUS`). En crédit, motiver
une décision par ces caractéristiques est **discriminatoire et non conforme** (RGPD,
recommandations ACPR, AI Act).

État actuel et honnêteté sur la limite :
- **Dans l'explication affichée** (lecture métier), ces variables sont **exclues** :
  aucune décision n'est justifiée au conseiller par le genre, l'âge ou la situation
  familiale.
- **Dans le modèle**, elles sont **encore présentes** : le `.pkl` a été entraîné avec.
  Les masquer à l'affichage ne supprime pas le biais sous-jacent.

Correction attendue en production (non faite ici car elle impose un ré-entraînement,
hors périmètre de cette archive) :
1. retirer les variables protégées (et leurs proxys évidents) des features ;
2. ré-entraîner et re-calibrer ;
3. mesurer l'équité entre groupes (p. ex. *demographic parity*, *equal opportunity*)
   et documenter l'écart résiduel.

### 10. Conclusion
Un système de scoring de bout en bout, de la donnée brute à une application
déployée où **chaque décision reste auditable**. Projet de formation
(OpenClassrooms), reconstruit en application Streamlit autonome pour rester
démontrable sans backend à maintenir.

## Données d'entrées
 - Lien de téléchargement des données d'entrées : https://www.kaggle.com/c/home-credit-default-risk/data 
 - Notebook de départ pour la partie Features Engineering : https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
