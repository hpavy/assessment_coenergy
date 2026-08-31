# Coenergy Assessment compte rendu

- [Coenergy Assessment compte rendu](#coenergy-assessment-compte-rendu)
  - [Méthodologie \& Traitement des données](#méthodologie--traitement-des-données)
  - [Modélisation \& Contrainte Physique](#modélisation--contrainte-physique)
  - [Baselines](#baselines)
    - [1. Prédicteur Constant (Moyenne)](#1-prédicteur-constant-moyenne)
    - [2. Régression Linéaire](#2-régression-linéaire)
  - [Neural Network](#neural-network)
  - [Structure du projet](#structure-du-projet)

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
| Test | 20.669 | 10.505 | 7.609 | 1.447 | 5.691 | 0.320 | 0.865 | 0.082 | 0.160 | 0.069 |

## Neural Network

Pour capturer les non-linéarités, un réseau de neurones simple (Multi-Layer Perceptron - MLP) a été implémenté. Il aplatit les séries temporelles et les fait passer à travers des couches linéaires avec des activations ReLU.

Une recherche automatique sur 20 configurations a permis d'identifier la meilleure architecture et les meilleurs hyperparamètres. L'entraînement final s'effectue sur l'ensemble du jeu de données avec la configuration optimale.

| Métrique | MAE HLC (W/K) | MAE Hb (W/K) | MAE τ_b (h) | MAE τ_n (h) | MAE τ_inf (h) | R² HLC | R² Hb | R² τ_b | R² τ_n | R² τ_inf |
|---|---|---|---|---|---|---|---|---|---|---|
| Test | 20.690 | 10.778 | 7.618 | 1.443 | 5.712 | 0.315 | 0.858 | 0.082 | 0.175 | 0.066 |

**Analyse** : Le MLP parvient à améliorer légèrement la prédiction des constantes de temps (τ_n passe de 0.160 à 0.175 en R²), mais peine à améliorer le HLC par rapport à la régression linéaire. Cela indique que l'aplatissement de la série temporelle détruit une partie de l'information temporelle cruciale, justifiant l'utilisation d'architectures plus complexes (comme les CNN 1D).

## Structure du projet

```
.
├── data/                       # Jeu de données (batch_*.pt) - non versionné
├── notebooks/                  # Notebooks d'analyse et pipeline final
├── output/                     # Résultats d'entraînement (loss, metrics) - non versionné
├── src/coenergy/               # Bibliothèque principale (package installable)
│   ├── __init__.py
│   ├── util_assessement.py     # Loader fourni par coEnergy
│   ├── dataset.py              # QuentinDataset, split_data, Normalisation
│   ├── evaluate.py             # Métriques (MAE, R²) dans les unités physiques
│   ├── model.py                # Définition des modèles (SimpleMLP)
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
