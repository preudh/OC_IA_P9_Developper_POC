# Importation des bibliothèques

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
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

# Analyse exploratoire des données
# Charger les données du dataset
st.title("Analyse Exploratoire des Données Visuelles")
dataset_path = "data/Images/"
categories = os.listdir(dataset_path)

# Distribution des catégories
category_counts = {cat: len(os.listdir(os.path.join(dataset_path, cat))) for cat in categories}
category_df = pd.DataFrame(list(category_counts.items()), columns=["Catégorie", "Nombre d'images"])

# Visualisation
st.subheader("Distribution des images par catégorie")
fig = px.bar(category_df, x="Catégorie", y="Nombre d'images", title="Nombre d'images par catégorie")
st.plotly_chart(fig)

# Afficher des exemples d'images
st.subheader("Exemples d'images par catégorie")
for category in categories:
    st.write(f"Exemple pour la catégorie : {category}")
    image_path = os.path.join(dataset_path, category, os.listdir(os.path.join(dataset_path, category))[0])
    st.image(image_path, caption=f"Catégorie : {category}", use_column_width=True)


# Moteur de prédiction

st.title("Prédictions de Classification d'Images")

uploaded_image = st.file_uploader("Chargez une image pour tester les prédictions", type=["jpg", "png"])

if uploaded_image:
    # Afficher l'image chargée
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)

    # Prétraitement pour la prédiction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prédictions
    vgg_prediction = predict_image(image, vgg_model, transform)
    vit_prediction = predict_image(image, vit_model, transform)

    st.write(f"Prédiction avec le modèle VGG16 (baseline) : Catégorie {vgg_prediction}")
    st.write(f"Prédiction avec le modèle ViT (modèle amélioré) : Catégorie {vit_prediction}")


# Amélioration du contraste pour l'accessibilité
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f7f7;
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)