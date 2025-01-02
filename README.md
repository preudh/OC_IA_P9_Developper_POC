# Projet P9 : Preuve de Concept - Classification d'Images avec VGG16 (Modèle de base) et ViT (Vision Transformer - Modèle amélioré)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monp9streamlit.westeurope.azurecontainer.io:8501)

## Objectifs du Projet
L'objectif de ce projet est de concevoir une preuve de concept (PoC) visant à démontrer l'efficacité de modèles de
classification d'images basés sur des architectures pré-entraînées (VGG16 et Vision Transformer - ViT). Cette preuve de
concept comprend :

1. **Analyse Exploratoire des Données** :
   - Fournir une visualisation claire des données à l'aide de graphiques interactifs.
   - Montrer des exemples d'images par catégorie.
2. **Classification d'Images** :
   - Implémentation des modèles VGG16 et ViT pour prédire la catégorie des images.
   - Comparaison des performances des deux modèles.
3. **Accessibilité** :
   - Conception d'une interface utilisateur accessible via Streamlit, conforme aux bonnes pratiques WCAG.
4. **Déploiement** :
   - Fournir une solution prête à être déployée sur Azure.



## Fonctionnalités Clés
### 1. Analyse Exploratoire des Données
- **Visualisation Interactive** : Nombre d'images par catégorie affiché à l'aide de graphiques interactifs (Plotly).
- **Exemples d'Images** : Une image représentative est affichée pour chaque catégorie.

### 2. Classification d'Images
- **Prédiction** : Les utilisateurs peuvent choisir une image et un modèle pour prédire la catégorie.
- **Validation des Prédictions** : Les utilisateurs peuvent valider manuellement les résultats des prédictions.

### 3. Accessibilité et Interface Utilisateur
- L'application Streamlit a été conçue pour être simple et accessible, avec des couleurs contrastées et des composants intuitifs.

### 4. Environnement Reproductible
- Un fichier "requirements.txt" est fourni pour installer les dépendances requises.

## Instructions pour l'Exécution
### Prérequis
1. **Python 3.11**
2. **Conda** (Anaconda ou Miniconda installé)

### Installation
1. Clonez le dépôt GitHub :
   
   
2. Créez l'environnement conda :
   ```bash
   conda env create -f environment.yml
   conda activate P9
   ```
3. Lancez l'application Streamlit en local :
   ```bash
   streamlit run app.py
   ```
ou 

4. Via Docker

## Builder l'image docker en local
docker build -t mon-image:latest .

# Tester l'image docker
docker run -p 8501:8501 mon-image:latest
  
5. Pour le déploiement sur azure afin de simplifier le déploiement, nous avons utilisé Azure Container Instances (ACI)
pour déployer notre application Streamlit. Comme il s'agit d'un POC (Proof of Concept) les data et models sont stockés dans le container.
Normalement, il faudrait stocker les données et les modèles dans un stockage Azure Blob et l'application Streamlit dans un container,
mais la volumétrie des données étant faible, nous avons opté pour cette solution de facilité.

## Bibliothèques Utilisées
Les bibliothèques essentielles incluent :
- **PyTorch** : Pour les modèles VGG16 et ViT.
- **Streamlit** : Pour le développement de l'interface utilisateur.
- **Plotly** : Pour les visualisations interactives.
- **Pandas** : Pour la manipulation des données.
- **Pillow** : Pour le traitement des images.

## Déploiement
- Déploiement sur Azure : http://monp9streamlit.westeurope.azurecontainer.io:8501
- Le déploiement avec du MlOps pourrait être envisagé pour une mise en production via GitHub Actions et Azure.

