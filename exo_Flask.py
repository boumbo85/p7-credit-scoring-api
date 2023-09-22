import os
from flask import Flask

app = Flask(__name__)

# Lisez le numéro de port à partir de la variable d'environnement $PORT
port = int(os.environ.get('PORT', 5000))

@app.route('/')
def index():
    return "Hello world !"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
