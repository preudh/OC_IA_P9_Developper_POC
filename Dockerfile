# Utiliser une image Python officielle comme base
FROM python:3.11-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Streamlit (obligatoire pour Docker mais le port sera redirigé par Azure)
EXPOSE 8501

# Commande par défaut pour démarrer l'application Streamlit
# Utiliser le port défini dans la variable d'environnement PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.enableCORS=false"]

