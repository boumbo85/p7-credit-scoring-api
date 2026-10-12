# Retour critique — gestion du déséquilibre de classes

> Note de travail (revue technique). Consigne la réflexion sur ce qui aurait dû
> être fait pour traiter correctement le déséquilibre de classes, et comment
> faire évoluer l'app. Sert de base aux améliorations futures.

## Partie 1 — Détecter les classes déséquilibrées correctement

### Le vrai problème du modèle actuel
Ce n'est **pas** que le modèle ne détecte pas les défauts. C'est l'inverse : il les
détecte *trop*. Il classe presque tout le monde en "défaut". C'est un problème de
**calibration des probabilités**, pas de pouvoir discriminant.

La cause est précise : **SMOTE a été appliqué, puis le modèle a été figé tel quel,
et `predict()` utilise le seuil 0.5.**

Quand on rééquilibre artificiellement à 50/50 avec SMOTE, le modèle apprend à "voir"
un monde où 50 % des gens font défaut. Mais la réalité, c'est ~8 %. Du coup ses
probabilités de sortie sont systématiquement gonflées : un client à vrai risque 8 %
ressort à ~0.5-0.8. Au seuil 0.5, tout bascule en défaut. **La séparation existe
(AUC 0.64-0.78), mais le curseur est au mauvais endroit.**

### Ce qu'il aurait fallu faire (par ordre d'impact)

**1. Ne pas se fier au `predict()` par défaut — chercher le seuil.**
C'est 80 % du problème, et c'est gratuit. Au lieu de `clf.predict()` (seuil 0.5
implicite), on optimise un seuil sur `predict_proba` :

```python
proba = clf.predict_proba(X_val)[:, 1]
# tester les seuils, garder celui qui minimise le coût métier
seuils = np.linspace(0.05, 0.95, 181)
couts = [cout_metier(y_val, proba >= s) for s in seuils]
seuil_optimal = seuils[np.argmin(couts)]
```

Le `fbeta β=10` était déjà présent dans le notebook — mais le modèle sérialisé ne
l'applique pas au moment de la décision. C'est ça le chaînon manquant.

**2. Recalibrer les probabilités après SMOTE.**
SMOTE casse la calibration. On la répare :

```python
from sklearn.calibration import CalibratedClassifierCV
clf_cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
clf_cal.fit(X_calib, y_calib)   # sur un jeu NON rééquilibré !
```

Point crucial : on calibre sur des données à la **vraie** distribution (8 %),
jamais sur du SMOTE.

**3. Préférer les alternatives à SMOTE pour LightGBM.**
Pour un gradient boosting, SMOTE est rarement le bon choix. Les arbres gèrent le
déséquilibre nativement via la **pondération** :

```python
LGBMClassifier(scale_pos_weight=11)        # ratio neg/pos
# ou
LGBMClassifier(class_weight='balanced')
# ou le paramètre LightGBM dédié
LGBMClassifier(is_unbalance=True)
```

Ça pénalise les erreurs sur la classe rare **sans fabriquer de faux clients
synthétiques** et **sans détruire la calibration**. En banque, c'est l'approche
standard — on veut des probabilités qui veulent dire quelque chose, parce qu'elles
alimentent des décisions chiffrées (provisionnement, taux).

**4. Les bonnes métriques pour le déséquilibre.**
L'accuracy ne veut rien dire ici (prédire "jamais de défaut" = 92 % d'accuracy,
inutile). Les métriques à regarder :

- **AUC-PR** (précision-rappel) > AUC-ROC quand la classe positive est rare. C'est
  LE réflexe banque.
- **Recall de la classe défaut** à précision fixée (ou l'inverse).
- **Coût métier** : la vraie boussole. Combien coûte un FN vs un FP ? On optimise ça
  directement.
- **Brier score** pour vérifier que les probas sont calibrées.

### En une phrase
Rééquilibrer sert à l'**apprentissage**. Calibrer + choisir le seuil sert à la
**décision**. Le projet fait le premier, pas les deux derniers. C'est exactement ce
qui coince — et c'est une super histoire à raconter en entretien, parce que c'est
l'erreur classique que font 90 % des gens sur ce sujet.

## Partie 2 — Comment faire évoluer l'app

Distinction entre ce qui vaut le coup (effort minimal, impact crédibilité) et le
sur-investissement.

### Ça vaut le coup
1. **Corriger la décision** avec un seuil métier appliqué dans `main.py` → les
   clients ne seront plus tous "refusés". C'est le fix qui rend la démo honnête, et
   il est petit (quelques lignes). Ça débloque aussi l'agrandissement du dataset.
2. **Afficher la probabilité ET le seuil** sur la jauge, avec une mention "seuil
   optimisé sur coût métier (β=10)". Ça montre qu'on *comprend* le problème — ce qui
   compte plus que le chiffre lui-même.
3. **Une phrase "lecture métier"** sous le SHAP : "ce client est refusé car
   endettement élevé + apport faible". Un recruteur adore voir qu'on traduit le
   modèle en décision.

### À ne pas faire (pour une archive)
Réentraîner complètement, MLOps, monitoring de drift, API séparée… hors-sujet pour
un projet de formation archivé. Garder cette énergie pour les 3 projets phares
(anomaly detection, C. elegans, job-engine).

### Avis de praticien
Ce projet a surtout de la valeur comme **preuve qu'on sait parler risque, coût
asymétrique et calibration** — pas comme preuve de performance brute. Le meilleur
ROI n'est pas de "faire tourner un meilleur modèle", c'est de **raconter juste** :
AUC honnête, seuil orienté coût, limites assumées (calibration). Un recruteur senior
en risque bancaire sera bien plus impressionné par "j'ai compris que SMOTE casse la
calibration et pourquoi" que par un AUC gonflé.

Lien avec le positionnement **détection d'anomalies** (projet phare Capgemini) : la
fraude bancaire = classe ultra-rare (~0.1 %), même famille de problèmes. Les réflexes
sont identiques — AUC-PR, coût asymétrique, seuil — et souvent des approches
non-supervisées (Isolation Forest, autoencoders) quand les fraudes ne sont pas
étiquetées. Un pont à faire valoir en entretien.
