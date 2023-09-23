#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
">  ~/.streamlit/config.toml

# Installer les dépendances communes aux deux applications depuis requirements.txt
# To create it, pip freeze > requirements.txt
pip install -r requirements.txt

# Démarrer l'application Flask en arrière-plan (backend)
gunicorn app:app -b 0.0.0.0:$PORT &

# Attendre quelques secondes pour que l'application Flask démarre
#sleep 5

# Lancer l'application Streamlit (frontend)
#streamlit run main.py
