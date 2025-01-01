import streamlit as st
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import plotly.express as px

"""
Dashboard Streamlit : Analyse Exploratoire et Preuve de Concept pour la Classification d'Images

Ce dashboard répond aux objectifs suivants :

1. **Analyse Exploratoire des Données :**
   - Visualisation interactive de la distribution des catégories via un graphique (bar chart).
   - Présentation d'exemples d'images pour chaque catégorie afin d'illustrer le contenu du dataset.

2. **Moteur de Prédiction :**
   - Permet la sélection d'images individuelles via une liste déroulante.
   - Offre le choix entre deux modèles de classification :
       - **VGG16 (baseline)**
       - **ViT (improved)**

3. **Résultat des Prédictions :**
   - Affiche la prédiction pour l'image sélectionnée.
   - Compare la prédiction avec le label réel issu du fichier CSV.
   - Intègre une validation utilisateur interactive pour confirmer ou rejeter la prédiction.

4. **Accessibilité et Conformité aux Critères WCAG :**
   - Contraste des couleurs et navigation clavier.
   - Captions et descriptions textuelles pour les éléments visuels.

5. **Déploiement :**
   - sur azure

Cette application sert de preuve de concept, démontrant les résultats des prédictions et les améliorations apportées par l'utilisation du modèle ViT par rapport à VGG16.
"""


try:
    # 1) Charger directement les modèles complets
    vgg_model = torch.load("model/vgg_model_best_weights.pth", map_location="cpu")
    vgg_model.eval()

    vit_model = torch.load("model/vit_model_best_weights.pth", map_location="cpu")
    vit_model.eval()

    # 2) Lecture du CSV
    csv_path = os.path.join("data", "flipkart_images.csv")
    labels_df = pd.read_csv(csv_path, sep=",")

    if labels_df.empty:
        st.error(f"The file '{csv_path}' is empty. Please check its content.")
        st.stop()

    # Nettoyage
    labels_df["label"] = pd.to_numeric(labels_df["label"], errors="coerce")
    labels_df.dropna(subset=["label"], inplace=True)
    labels_df["label"] = labels_df["label"].astype(int)

    if labels_df.isnull().any().any():
        st.error("The CSV file contains missing or invalid values. Please check.")
        st.stop()

    # Vérifier dossier images
    dataset_path = os.path.join("data", "Images")
    if not os.path.exists(dataset_path):
        st.error(f"The directory '{dataset_path}' was not found. Check the project structure.")
        st.stop()

    # Vérifier si des images listées dans le CSV manquent dans le dossier
    missing_images = [
        img for img in labels_df["image"]
        if not os.path.exists(os.path.join(dataset_path, img))
    ]
    if missing_images:
        st.error(f"These images are listed in '{csv_path}' but missing in '{dataset_path}': {missing_images}")
        st.stop()

    # 3) Analyse Exploratoire
    st.title("Analyse Exploratoire des Données Visuelles")

    st.subheader("Distribution des images par catégorie")
    category_counts = labels_df["category"].value_counts()
    category_df = category_counts.reset_index()
    category_df.columns = ["Catégorie", "Nombre d'images"]

    fig = px.bar(
        category_df,
        x="Catégorie",
        y="Nombre d'images",
        title="Nombre d'images par catégorie"
    )
    st.plotly_chart(fig)

    # Exemples d'images
    st.subheader("Exemples d'images par catégorie")
    for category in category_counts.index:
        st.write(f"Exemple pour la catégorie : {category}")
        sample_image = labels_df[labels_df["category"] == category]["image"].iloc[0]
        image_path = os.path.join(dataset_path, sample_image)
        st.image(
            image_path,
            caption=f"Catégorie : {category}",
            use_container_width=True
        )

    # 4) Classification
    st.title("Prédictions de Classification d'Images")
    st.subheader("Navigation dans le dataset")

    selected_image = st.selectbox("Choisissez une image :", labels_df["image"])

    if selected_image:
        image_path = os.path.join(dataset_path, selected_image)
        image = Image.open(image_path)
        st.image(
            image,
            caption=f"Image sélectionnée : {selected_image}",
            use_container_width=True
        )

        # Choisir le modèle
        chosen_model = st.radio(
            "Choisissez le modèle pour la prédiction :",
            ("VGG16 (baseline)", "ViT (improved)")
        )

        if st.button("Lancer la prédiction"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            def predict_image(img, model, transform_fn, is_vit=False):
                model.eval()
                img_tensor = transform_fn(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    # Si le modèle est ViT, accéder à la clé 'logits'
                    if is_vit:
                        output = output.logits
                    _, pred = torch.max(output, 1)
                return pred.item()

            # Selon le choix de l'utilisateur, on utilise VGG ou ViT
            if chosen_model == "VGG16 (baseline)":
                prediction = predict_image(image, vgg_model, transform, is_vit=False)
            else:
                prediction = predict_image(image, vit_model, transform, is_vit=True)

            # Récupérer le label réel
            real_label = labels_df[labels_df["image"] == selected_image]["label"].values[0]
            real_category = labels_df[labels_df["image"] == selected_image]["category"].values[0]

            st.write(f"Label réel : {real_category} (ID : {real_label})")
            st.write(f"Prédiction avec {chosen_model} : Catégorie {prediction}")

            st.subheader("Valider les prédictions")
            validation = st.radio("Les prédictions sont-elles correctes ?", ("Oui", "Non"))
            if validation == "Oui":
                st.success("Prédictions validées !")
            else:
                st.error("Prédictions incorrectes. Analyse requise.")

    # 5) Style Streamlit
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f7f7f7;
            color: #000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("FileNotFoundError: Please make sure your model and CSV paths are correct.")
    st.stop()

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()









