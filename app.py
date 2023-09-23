from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
import shap
import zipfile
import os
print("Application lancée")

app = Flask(__name__)

# LOAD DATA
# Obtenez le chemin du répertoire actuel
#current_directory = os.path.dirname(__file__)
# Obtenez le chemin du répertoire parent du répertoire actuel
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

# Chemin vers le fichier ZIP
zip_file_path = os.path.join(parent_directory, 'df_train3_good-col.zip')
model_zip_file_path = os.path.join(parent_directory, 'LGBMC2.zip') #parent_directory

# Ouvrir le fichier ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extraire le fichier CSV souhaité dans un dossier temporaire
    zip_ref.extract('df_train3_good-col.csv', 'dossier_temporaire')

# Lire le fichier CSV extrait avec pandas
df = pd.read_csv('dossier_temporaire/df_train3_good-col.csv')

# LOAD MODEL
def fbeta10_scorer_custom(test_y, y_pred):
    return fbeta_score(test_y, y_pred, beta=10)
fbeta10_custom_scorer = make_scorer(fbeta10_scorer_custom)

"""def load_model(model_file_name):
    '''loading the trained model'''
    pickle_in = open(model_file_name, 'rb') 
    fbeta10_scorer = make_scorer(fbeta_score, beta=10)
    res = pickle.load(pickle_in)
    return res"""

def load_model_from_zip(zip_file_name, model_file_name):
    '''loading the trained model from a zip archive'''
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        with zip_ref.open(model_file_name, 'r') as pickle_file:
            fbeta10_scorer = make_scorer(fbeta_score, beta=10)
            res = pickle.load(pickle_file)
            return res

# Load the model
clf = load_model_from_zip(model_zip_file_path,'LGBMC2.pkl')
CORS(app)

# Test the connection
@app.route('/')
@cross_origin()

def index():
    return "Bienvenu(e) !"
    

@app.route('/prediction/<int:id_client>') # http://127.0.0.1:5001/prediction/100031
@cross_origin()
def prediction(id_client): 

    # recup ID client
    id_client = int(id_client)
    selected_row = df.loc[df['SK_ID_CURR'] == id_client]
    id_row = selected_row.index.values

    if selected_row.empty:
        return jsonify({'Error': 'Client ID not found'})
    
    # cree vecteur X = s
    X_selected = selected_row.iloc[:,:-2]
    
    # Recup results 
    y_pred = clf.predict(X_selected)
    y_pred_proba = clf.predict_proba(X_selected)

    # Recup feature importance Local
    model = clf.named_steps['lgbmclassifier'] # recup feature importance locale
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected) # shapley values from game theory
    base_value = explainer.expected_value[0] # single expected value as base, one for each class

    # Return the results as a dictionary
    res = {
            'ID': int(id_client), 
            'Prediction': y_pred.tolist(),
            'Predict_proba': y_pred_proba.tolist(),
            'Feature_Importance': shap_values[0].tolist(),
            'Expected_value': base_value
          }
    
    return jsonify(res)

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5001))
    # port = int(os.environ.get('PORT', $PORT))
    app.run() # debug=True port=5001 port=port 
