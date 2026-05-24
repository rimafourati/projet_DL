# Projet Deep Learning — Classification de Chiffres Manuscrits (MNIST)

Un réseau de neurones MLP (Multi-Layer Perceptron) construit avec PyTorch pour classifier les chiffres manuscrits du célèbre dataset MNIST.

---

## Objectif

Entraîner un modèle de Deep Learning capable de reconnaître les chiffres de 0 à 9 à partir d'images 28×28 pixels en niveaux de gris, en atteignant une précision d'environ **~98%** sur le jeu de test.

---

## Dataset

- **MNIST** — 70 000 images de chiffres manuscrits
  - 60 000 images d'entraînement
  - 10 000 images de test
  - Images : 28×28 pixels, niveaux de gris
  - 10 classes (chiffres 0 à 9)
- Téléchargement automatique via `torchvision.datasets.MNIST`

---

## Architecture du modèle (MLP)

```
Input (784) → FC(512) + BatchNorm + ReLU + Dropout(0.3)
            → FC(256) + BatchNorm + ReLU + Dropout(0.3)
            → FC(128) + ReLU
            → Output (10)
```

| Couche | Taille | Activation | Régularisation |
|--------|--------|------------|----------------|
| Input  | 784    | —          | —              |
| Hidden 1 | 512  | ReLU       | BatchNorm + Dropout(0.3) |
| Hidden 2 | 256  | ReLU       | BatchNorm + Dropout(0.3) |
| Hidden 3 | 128  | ReLU       | —              |
| Output | 10     | Softmax    | —              |

---

## Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimiseur | Adam |
| Epochs | 15 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss | CrossEntropyLoss |

---

## Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy test | ~98% |
| Device utilisé | GPU (CUDA) |

---

## Structure du projet

```
📦 projet-deep-learning-mnist
 ┣ 📓 projet-deep-learning-mnist.ipynb   # Notebook principal
 ┗ 📄 README.md                          # Ce fichier
```

---

## Contenu du notebook

1. **Imports & Configuration** — PyTorch, device (CPU/GPU), seed
2. **Chargement des données** — MNIST via torchvision, normalisation
3. **Exploration** — visualisation des images, distribution des classes
4. **DataLoaders** — batching avec `torch.utils.data.DataLoader`
5. **Modèle MLP** — définition avec `nn.Module` et `nn.Sequential`
6. **Entraînement** — boucle train/eval, sauvegarde du meilleur modèle
7. **Évaluation** — courbes loss/accuracy, matrice de confusion, rapport de classification
8. **Visualisation** — exemples de prédictions correctes et incorrectes

---

## Prérequis

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

> Sur **Kaggle** ou **Google Colab**, toutes les dépendances sont déjà installées.

---

## Lancer le projet

### Sur Kaggle
1. Importer le notebook `.ipynb`
2. Activer le GPU : **Settings → Accelerator → GPU P100**
3. Run All

### Sur Google Colab
1. Importer le notebook `.ipynb`
2. Activer le GPU : **Runtime → Change runtime type → T4 GPU**
3. Run All

### En local
```bash
jupyter notebook projet-deep-learning-mnist.ipynb
```

---

## Technologies utilisées

- **Python 3.10+**
- **PyTorch 2.x**
- **torchvision**
- **matplotlib / seaborn**
- **scikit-learn**


