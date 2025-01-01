# Projet P9 : Preuve de Concept - Classification d'Images avec VGG16 et ViT

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
- Un fichier `environment.yml` est fourni pour recréer l'environnement conda nécessaire à l'exécution du projet.

## Instructions pour l'Exécution
### Prérequis
1. **Python 3.11**
2. **Conda** (Anaconda ou Miniconda installé)

### Installation
1. Clonez le dépôt GitHub :
   ```bash
   git clone <url-du-depot>
   cd OC_IA_P9_Developper_POC
   ```
2. Créez l'environnement conda :
   ```bash
   conda env create -f environment.yml
   conda activate P9
   ```
3. Lancez l'application Streamlit en local :
   ```bash
   streamlit run app.py
   ```

## Bibliothèques Utilisées
Les bibliothèques essentielles incluent :
- **PyTorch** : Pour les modèles VGG16 et ViT.
- **Streamlit** : Pour le développement de l'interface utilisateur.
- **Plotly** : Pour les visualisations interactives.
- **Pandas** : Pour la manipulation des données.
- **Pillow** : Pour le traitement des images.

## Déploiement
- Déploiement sur Azure : [Lien vers l'Application]
- Le déploiement peut être automatisé via GitHub Actions.

