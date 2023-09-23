#!/bin/bash

# Config file for streamlit
mkdir -p ~/.streamlit/
#echo "\
#[general]\n\
#email = \"your-email@domain.com\"\n\
#" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
">  ~/.streamlit/config.toml

# Installer les dépendances communes aux deux applications depuis requirements.txt
# To create it, pip freeze > requirements.txt
#pip install -r requirements.txt

# Démarrer l'application Flask en arrière-plan (backend)
# gunicorn app:app &

# Attendre quelques secondes pour que l'application Flask démarre
#sleep 5

# Lancer l'application Streamlit (frontend)
#streamlit run main.py
