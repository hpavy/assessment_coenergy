# Coenergy Assessment compte rendu

- [Coenergy Assessment compte rendu](#coenergy-assessment-compte-rendu)
  - [Méthodologie \& Traitement des données](#méthodologie--traitement-des-données)
  - [Modélisation \& Contrainte Physique](#modélisation--contrainte-physique)
  - [Baselines](#baselines)
    - [1. Prédicteur Constant (Moyenne)](#1-prédicteur-constant-moyenne)
    - [2. Régression Linéaire](#2-régression-linéaire)
  - [Neural Network](#neural-network)
  - [Structure du projet](#structure-du-projet)
  - [Comment exécuter le code](#comment-exécuter-le-code)

## Méthodologie & Traitement des données

Nous disposons de 2 millions de simulations générées par coEnergy. Le jeu de données est divisé en trois sous-ensembles :
- **80 %** pour l'entraînement (`train`)
- **10 %** pour la validation (`val`)
- **10 %** pour le test (`test`)

**Normalisation Z** : Les données d'entrée et de sortie sont normalisées à l'aide d'une normalisation Z (standardisation) :

$$
\hat{X} = \frac{X - X_{mean}}{\sigma}
$$

Cette normalisation permet d'addimensionner les grandeurs et d'avoir des inputs et outputs de bonne échelle pour le modèle. 

> **Note importante** : Les statistiques de normalisation (moyenne et écart-type) sont calculées **uniquement sur le jeu d'entraînement** pour éviter toute fuite de données (data leakage) vers les jeux de validation et de test.

## Modélisation & Contrainte Physique

L'objectif est de prédire un vecteur d'indicateurs de performance thermique à partir des séries temporelles de température intérieure et de sollicitation thermique.

Les indicateurs à prédire sont :
- **Hb** : Décomposition du coefficient de déperdition thermique (HLC) par composant (murs, toit, etc.) en W/K.
- **CDT** : Constantes de temps ($\tau_b, \tau_n, \tau_{inf}$) en heures, caractérisant l'inertie thermique.

**Contrainte physique majeure** : Le HLC global est exactement égal à la somme de ses décompositions : $HLC = \sum Hb$.

Pour garantir cette contrainte par construction, le modèle ne prédit **que** les composantes `Hb` et `CDT`. Le `HLC` n'est pas prédit directement, mais dérivé a posteriori en sommant les composantes `Hb` prédites. Ainsi, la contrainte physique est respectée à 100%, quelles que soient les prédictions du modèle.

## Baselines

Avant de construire un modèle complexe, deux baselines (modèles de référence) ont été établies pour évaluer la difficulté du problème.

### 1. Prédicteur Constant (Moyenne)
Ce modèle prédit systématiquement la moyenne des cibles d'entraînement. Il établit le "plancher" de performance.

| Métrique | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) |
|---|---|---|---|---|---|
| Test | 25.231 | 11.841 | 7.941 | 1.579 | 5.896 |

### 2. Régression Linéaire
Ce modèle extrait des caractéristiques statistiques (moyenne, écart-type, max, min) sur 100 segments de la série temporelle, puis applique une régression linéaire simple. 

Une recherche automatique sur 20 configurations a été effectuée pour trouver les meilleurs paramètres. Les résultats montrent que la relation entre les entrées et les sorties est **hautement non-linéaire** (le modèle linéaire ne fait guère mieux que la moyenne).

| Métrique | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) | R² HLC | R² Hb | R² τ_b | R² τ_n | R² τ_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| Test | 20.669 | 10.505 | 7.609 | 1.447 | 5.691 | 0.320 | 0.156 | 0.082 | 0.160 | 0.069 |

## Neural Network (MLP)

Pour capturer les non-linéarités, un réseau de neurones simple (Multi-Layer Perceptron - MLP) a été implémenté. Il aplatit les séries temporelles et les fait passer à travers des couches linéaires avec des activations ReLU.

Une recherche automatique sur 20 configurations a permis d'identifier la meilleure architecture et les meilleurs hyperparamètres. L'entraînement final s'effectue sur l'ensemble du jeu de données avec la configuration optimale.

| Métrique | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) | R² HLC | R² Hb | R² τ_b | R² τ_n | R² τ_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| Test | 20.619 | 10.773 | 7.609 | 1.431 | 5.702 | 0.319 | 0.122 | 0.085 | 0.181 | 0.069 |

**Analyse** : Le MLP n'améliore pas la régression linéaire de façon significative — R² HLC 0.319 contre 0.320 pour le modèle linéaire. Aplatir la série temporelle détruit la structure temporelle, mais ce n'est pas la seule limite.

## CNN — meilleur modèle

Le MLP aplatit la série temporelle, ce qui détruit la structure temporelle. Un CNN 1D dilaté (`DilatedCNN`) la préserve. Choix d'architecture :

- **Stem convolutif à pas de 2 (×2)** : sous-échantillonne la séquence de 600 à 150 pas. Les cibles sont un gain statique et des constantes de temps en heures : la résolution au pas de temps ne porte pas de signal utile, et le réseau devient 4× moins coûteux. Un `MaxPool` a été écarté car il ne conserve que l'enveloppe supérieure d'une trace de température, en jetant la décroissance qui détermine justement les constantes de temps.
- **6 blocs résiduels à dilatation doublante (1 → 32)** : champ réceptif d'environ 1024 pas pour T = 600, donc chaque sortie voit la réponse entière. Indispensable pour τ_inf, dont l'ordre de grandeur atteint plusieurs dizaines d'heures.
- **Pooling global moyenne + max** au lieu du dernier pas de temps : le HLC est essentiellement un gain moyen sur la réponse. Le pooling moyen peut exprimer cette grandeur, un unique pas de temps final ne peut pas.
- **Pas de récurrence** : le réseau traite toute la séquence en parallèle. 0.19 M paramètres contre 1.41 M pour le MLP, soit 7× moins, et environ 25 ms par batch de 512.

| Métrique | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) | R² HLC | R² Hb | R² τ_b | R² τ_n | R² τ_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| Test | 19.541 | 10.203 | 7.333 | 1.390 | 5.554 | 0.390 | 0.190 | 0.145 | 0.228 | 0.110 |

### Comparaison des modèles (test, jeu complet)

Toutes les valeurs sont des MAE en unités physiques réelles (W/K et heures) — plus bas est meilleur.

| Modèle | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) | MAE totale |
|---|---|---|---|---|---|---|
| Constant (moyenne) | 25.231 | 11.841 | 7.941 | 1.579 | 5.896 | 52.488 |
| Régression linéaire | 20.669 | 10.505 | 7.609 | 1.447 | 5.691 | 45.921 |
| MLP | 20.619 | 10.773 | 7.609 | 1.431 | 5.702 | 46.134 |
| **CNN dilaté** | **19.541** | **10.203** | **7.333** | **1.390** | **5.554** | **44.022** |

**Analyse** : le CNN dilaté est le meilleur modèle sur les cinq cibles simultanément. Il réduit la MAE totale de 1.899 (4.1 %) par rapport à la régression linéaire, et de 2.112 (4.6 %) par rapport au MLP. Sur le HLC, l'erreur passe de 20.669 à 19.541 W/K.

Il faut garder la mesure juste : le gain est réel et systématique, mais il reste modeste en valeur absolue. Le CNN se situe à 23 % sous le prédicteur constant sur le HLC, là où la régression linéaire était déjà à 18 %. Autrement dit, l'architecture capture une part de non-linéarité que les modèles précédents manquaient, sans faire basculer le problème.

Le MLP, lui, n'apporte rien sur la régression linéaire (MAE totale 46.134 contre 45.921) : aplatir la série temporelle détruit la structure que le CNN exploite.


## Structure du projet

```
.
├── data/                       # Jeu de données (batch_*.pt) - non versionné
├── output/                     # Résultats d'entraînement (loss, metrics) - non versionné
├── src/coenergy/               # Bibliothèque principale (package installable)
│   ├── __init__.py
│   ├── util_assessement.py     # Loader fourni par coEnergy
│   ├── dataset.py              # QuentinDataset, split_data, Normalisation
│   ├── evaluate.py             # Métriques (MAE, R²) dans les unités physiques
│   ├── model.py                # Définition des modèles (SimpleMLP, DilatedCNN)
│   ├── training.py             # Boucle d'entraînement (AMP, best weights)
│   ├── baseline_constant.py    # Baseline 1 (Moyenne)
│   ├── baseline_linear.py      # Baseline 2 (Régression linéaire)
│   └── utils.py                # Config, seed, sauvegarde des résultats
├── main.py                     # Point d'entrée principal (entraînement NN)
├── baseline_constant.py        # Runner Baseline 1
├── baseline_linear.py          # Runner Baseline 2
├── config.yaml                 # Configuration (modèle, training, split)
├── pyproject.toml              # Dépendances et build system (hatchling)
├── uv.lock                     # Lock des dépendances
└── .gitignore
```

## Comment exécuter le code

Le projet utilise `uv` pour la gestion des dépendances et de l'environnement. Voici les commandes pour lancer les différentes étapes :

```bash
# 1. Installer les dépendances (si ce n'est pas déjà fait)
uv sync

# 2. Lancer les baselines
uv run python baseline_constant.py   # Prédicteur constant (Moyenne)
uv run python baseline_linear.py     # Régression linéaire (100 parts × 4 stats)

# 3. Lancer l'entraînement du Neural Network (MLP)
uv run python main.py

# 4. Lancer les notebooks d'analyse (optionnel)
uv run jupyter lab
```

La configuration du modèle et de l'entraînement (taux d'apprentissage, taille des batchs, epochs, etc.) se modifie facilement dans le fichier `config.yaml`.
