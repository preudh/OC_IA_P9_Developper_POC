# Importation des bibliothèques
import streamlit as st
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import plotly.express as px

# Chargement du modèle VGG16 (baseline) et ViT (modèle amélioré)
vgg_model = torch.load("model/vgg_model_best_weights.pth")
vit_model = torch.load("model/vit_model_best_weights.pth")

# Fonction de prédiction
def predict_image(image, model, transform):
    model.eval()
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()

# Charger le fichier CSV contenant les correspondances
labels_df = pd.read_csv("data/flipkart_images_labels.csv")

# Chemin du dataset
dataset_path = "data/Images/"
categories = os.listdir(dataset_path)

# Analyse exploratoire des données
st.title("Analyse Exploratoire des Données Visuelles")

# Distribution des catégories
st.subheader("Distribution des images par catégorie")
category_counts = {cat: len(os.listdir(os.path.join(dataset_path, cat))) for cat in categories}
category_df = pd.DataFrame(list(category_counts.items()), columns=["Catégorie", "Nombre d'images"])

# Visualisation interactive
fig = px.bar(category_df, x="Catégorie", y="Nombre d'images", title="Nombre d'images par catégorie")
st.plotly_chart(fig)

# Exemples d'images par catégorie
st.subheader("Exemples d'images par catégorie")
for category in categories:
    st.write(f"Exemple pour la catégorie : {category}")
    image_path = os.path.join(dataset_path, category, os.listdir(os.path.join(dataset_path, category))[0])
    st.image(image_path, caption=f"Catégorie : {category}", use_column_width=True)

# Navigation dans le dataset et prédiction
st.title("Prédictions de Classification d'Images")
st.subheader("Navigation dans le dataset")

# Choisir une catégorie
selected_category = st.selectbox("Choisissez une catégorie :", categories)

if selected_category:
    # Choisir une image dans la catégorie sélectionnée
    image_files = os.listdir(os.path.join(dataset_path, selected_category))
    selected_image = st.selectbox("Choisissez une image :", image_files)

    # Afficher l'image sélectionnée
    image_path = os.path.join(dataset_path, selected_category, selected_image)
    image = Image.open(image_path)
    st.image(image, caption=f"Image sélectionnée : {selected_image}", use_column_width=True)

    # Bouton pour lancer les prédictions
    if st.button("Lancer la prédiction"):
        # Prétraitement pour la prédiction
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Prédictions
        vgg_prediction = predict_image(image, vgg_model, transform)
        vit_prediction = predict_image(image, vit_model, transform)

        # Obtenir le label réel depuis le DataFrame
        real_label = labels_df[labels_df['image'] == selected_image]['label'].values[0]
        real_category = labels_df[labels_df['image'] == selected_image]['category'].values[0]

        # Afficher les résultats
        st.write(f"Label réel : {real_category} (ID : {real_label})")
        st.write(f"Prédiction avec le modèle VGG16 (baseline) : Catégorie {vgg_prediction}")
        st.write(f"Prédiction avec le modèle ViT (modèle amélioré) : Catégorie {vit_prediction}")

        # Validation des prédictions
        st.subheader("Valider les prédictions")
        validation = st.radio("Les prédictions sont-elles correctes ?", ("Oui", "Non"))

        if validation == "Oui":
            st.success("Prédictions validées !")
        else:
            st.error("Prédictions incorrectes. Analyse requise.")

# Amélioration du contraste pour l'accessibilité
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f7f7;
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)

