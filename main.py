######################################## 
# IMPORT LIBRARY
########################################
import pickle
from sklearn.metrics import fbeta_score, make_scorer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
import plotly.express as px
import zipfile
import os

######################################## 
# FONCTION
########################################

# Recup information generales 
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                    round(data["AMT_INCOME_TOTAL"].mean(), 2),
                    round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets

# Recup information personnelles clients 
def identite_client(data, id):
        list_col = ["AGE",'CODE_GENDER','NAME_FAMILY_STATUS','CNT_CHILDREN','OCCUPATION_TYPE','YEARS_EMPLOYED']
        data_client = data[data['SK_ID_CURR'] == id]
        return data_client[list_col]

def load_income_population(sample):
      df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
      df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
      return df_income

def identite_client_credit (data, id):
    list_col = ['RATE_INDEBTEDNESS','AMT_INCOME_TOTAL','LTV_ratio',
            'AMT_ANNUITY','AMT_GOODS_PRICE', 'APPORT','AMT_CREDIT',
            'CREDIT_ACTIVE', 'CLASS_RATE_INDEBTEDNESS',
            "NAME_INCOME_TYPE","FLAG_OWN_REALTY",'CLASS_RATE_INDEBTEDNESS']               
    data_client = data.loc[data['SK_ID_CURR'] == id]
    if len(data_client) == 0:
        st.error("No matching rows found for id {}".format(id))
        return None
    else:    
        return data_client[list_col]

# IMPORT MODEL
def load_model_from_zip(zip_file_name, model_file_name):
    '''loading the trained model from a zip archive'''
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        with zip_ref.open(model_file_name, 'r') as pickle_file:
            fbeta10_scorer = make_scorer(fbeta_score, beta=10)
            res = pickle.load(pickle_file)
            return res

######################################## 
# LOAD AND PREPARE DATA
########################################

# LOAD DATA
# Obtenez le chemin du répertoire actuel
current_directory = os.path.dirname(__file__)
zip_file_dashboard_path = os.path.join(current_directory, 'df_dashboardY.zip')

with zipfile.ZipFile(zip_file_dashboard_path, 'r') as zip_ref:
    # Extraire le fichier CSV souhaité dans un dossier temporaire
    zip_ref.extract('df_dashboardY.csv', 'dossier_temporaire')

# Lire le fichier CSV extrait avec pandas
df = pd.read_csv('df_trainXY_sample.csv')
df_dashboard = pd.read_csv('dossier_temporaire/df_dashboardY.csv')

# Aligner l'affichage sur les clients réellement scorables :
# on ne garde dans le dashboard que les SK_ID_CURR présents dans df (ceux du menu),
# pour que chaque client sélectionné ait bien ses infos descriptives.
# L'échantillon suit la vraie distribution (~8 % de défauts) : les graphes de
# distribution sont donc représentatifs de la population réelle.
df_dashboard = df_dashboard[df_dashboard['SK_ID_CURR'].isin(df['SK_ID_CURR'])].reset_index(drop=True)

# subdivise dataframe
selection_ID = df['SK_ID_CURR']

# Load the model
model_zip_file_path = os.path.join(current_directory,'LGBMC2.zip') 
clf = load_model_from_zip(model_zip_file_path,'LGBMC2.pkl')

# Source de données utilisée pour la prédiction (features brutes attendues par le modèle).
# Les 2 dernières colonnes sont SK_ID_CURR et TARGET -> on les retire pour obtenir X.
df_predict = df.copy()

# Explainer SHAP construit une seule fois sur l'estimateur final de la pipeline.
# (évite de le reconstruire à chaque interaction utilisateur)
_shap_explainer = shap.TreeExplainer(clf.named_steps['lgbmclassifier'])

# ──────────────────────────────────────────────────────────────────────
# SEUIL DE DÉCISION MÉTIER
# ──────────────────────────────────────────────────────────────────────
# Le modèle est entraîné sur données rééquilibrées (SMOTE) : ses probabilités
# sont décalées vers le haut. Le seuil 0.5 par défaut de clf.predict() refuserait
# donc quasiment tout le monde.
# On fixe à la place un seuil calé sur le COÛT MÉTIER : un faux négatif (accorder
# un crédit à un futur défaillant) coûte bien plus cher qu'un faux positif. Avec un
# ratio de coût FN:FP de 10:1 (cohérent avec le Fβ β=10 utilisé à l'entraînement),
# le seuil qui minimise le coût total sur un échantillon représentatif vaut ~0.81.
# À ce seuil : ~71 % des clients acceptés / ~29 % refusés, recall défaut ~0.45.
DECISION_THRESHOLD = 0.81


def compute_prediction(id_client):
    """Calcule la probabilité, la décision (seuil métier) et les valeurs SHAP localement.

    Remplace l'ancien appel à l'API Flask (Heroku) : tout est calculé en local.
    La décision n'utilise PAS clf.predict() (seuil 0.5) mais le seuil métier
    DECISION_THRESHOLD appliqué à la probabilité de défaut.
    """
    id_client = int(id_client)
    selected_row = df_predict.loc[df_predict['SK_ID_CURR'] == id_client]
    if selected_row.empty:
        return None

    X_selected = selected_row.iloc[:, :-2]

    y_pred_proba = clf.predict_proba(X_selected)
    proba_default = float(y_pred_proba[0][1])
    # Décision = la probabilité de défaut dépasse-t-elle le seuil métier ?
    decision = int(proba_default >= DECISION_THRESHOLD)

    shap_values = _shap_explainer.shap_values(X_selected)
    base_value = _shap_explainer.expected_value
    # TreeExplainer renvoie une liste [classe0, classe1] pour un classifieur binaire ;
    # on garde la classe 0, comme le faisait le backend Flask d'origine.
    if isinstance(shap_values, list):
        shap_vector = shap_values[0]
        base_scalar = base_value[0]
    else:
        shap_vector = shap_values
        base_scalar = base_value

    return {
        'ID': id_client,
        'Prediction': [decision],
        'Predict_proba': y_pred_proba.tolist(),
        'Feature_Importance': shap_vector.tolist(),
        'Expected_value': base_scalar,
        'Threshold': DECISION_THRESHOLD,
    }

def main() :
    # Titre
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Risque de Défaut de Crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center"> Aide à la décision attribution de Crédit…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    ######################################## 
    # SIDEBAR
    ########################################
    # Menu ID_client 
    option = st.sidebar.selectbox(
            'Sélectionner un identifiant client:',
            selection_ID, key= 'select_ID_client')
    
    # Selection facteur d influence à la prediction
    st.sidebar.header("Facteurs influents:")
    show_global_feature_importance = st.sidebar.checkbox("globale")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    show_client_feature_importance = st.sidebar.checkbox("du client")
    if show_client_feature_importance:
        number = st.sidebar.slider("Pick a number of features…", 0, 20, 5)
   
    # Information clients
    st.sidebar.header("**Informations Client**")
    show_client_private_info = st.sidebar.checkbox("Personnelles")
    show_client_banq_info = st.sidebar.checkbox("Banquaires")
    show_client_loan_info = st.sidebar.checkbox("Crédits")

     # Select xcolumn
    st.sidebar.header("Variables")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    names_feature = list(df_dashboard.iloc[:,1:-1].columns)
    
    show_client_feature1 = st.sidebar.checkbox("Variable 1")
    if show_client_feature1:
        selected_featured = st.sidebar.selectbox(
                'Veuillez sélectionnez une variable :',
                names_feature, key='feature_selectbox')
    
    show_client_feature2 = st.sidebar.checkbox("Variable 2")
    if show_client_feature2:
        selected_featured2 = st.sidebar.selectbox(
            'Veuillez sélectionnez une deuxième variable :',
            names_feature, key='feature2_selectbox')

    # Information generales
    st.sidebar.header("Informations Générales")
    # Ajouter un saut de ligne, sinon conflit et menu passe a droite
    st.sidebar.markdown("<br>", unsafe_allow_html=True) 

    # Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(df_dashboard)
    # Display of information in the sidebar ###
    ## Number of loans in the sample
    st.sidebar.markdown("<u> Nombre de crédits enregistrés :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    ## PieChart
    st.sidebar.markdown("<u>Proportion:</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)

    # Filtrer le df en fonction de l ID client
    X_selected_client = df_dashboard[df_dashboard['SK_ID_CURR'] == option]

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    # Display Decision
    ## Prédiction calculée localement (plus d'appel réseau vers Flask/Heroku)
    def get_prediction(id_client):
        result = compute_prediction(id_client)
        if result is None:
            st.error("Client introuvable dans les données.")
        return result

    
    ## Decision
    # Divisez la page en colonnes
    col1, col2 = st.columns(2)

    # Colonne de gauche pour le titre encadré
    with col1:
        titre_decision = """
            <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
            <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Décision</h1>
            </div>
        """
        st.markdown(titre_decision, unsafe_allow_html=True)

        # Colonne de droite pour les résultats de la décision
        result = get_prediction(option)
        if result is None:
            st.stop()
        # Prediction est une liste ([0] ou [1]) : on lit le 1er élément.
        # Classe 1 = défaut => crédit refusé.
        if result['Prediction'][0] == 0:
            decision_text = "LOAN GRANTED"
            decision_color = "green"
        else:
            decision_text = "LOAN REJECTED"
            decision_color = "red"
        
        # Afficher résultats
        decision_circle = f"""
             <div style="width: 150px; height: 150px; background-color: black;
                border-radius: 50%; display: flex; align-items: center;
                justify-content: center; margin: auto; border: 5px solid {decision_color};">
            <h2 style="color: {decision_color}; text-align: center; font-weight: bold;">{decision_text}</h2>
            </div>
            """
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown(decision_circle, unsafe_allow_html=True)

    # Display Customer solvability prediction
    with col2:
        titre_prediction = """
        <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
        <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Prédiction</h2>
        </div>
        """
        st.markdown(titre_prediction, unsafe_allow_html=True)

        ### Customer scoring results
        predict_proba = result['Predict_proba']
        resultat = predict_proba[0][1]
        score_client = float(round(resultat*100,2))
        seuil_pct = round(result['Threshold'] * 100, 1)  # seuil de décision en %

        ## Vizualisation customer scoring  with yield Gauge Graph
        # La barre blanche = probabilité de défaut du client.
        # Le repère rouge épais = seuil de décision métier : au-dessus => crédit refusé.
        fig_jauge = go.Figure(go.Indicator(
            mode ='gauge+number',
            value = score_client,
            number = {'suffix': ' %'},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': 'Probabilité de défaut', 'font': {'size': 24}},
            gauge = {'axis': {'range': [0, 100],
                            'tickwidth': 3,
                            'tickcolor': 'white'},
                'bar': {'color': 'white', 'thickness': 0.25},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': 'gray',
                # vert sous le seuil (accepté), rouge au-dessus (refusé)
                'steps': [{'range': [0, seuil_pct], 'color': 'rgba(16,185,129,0.55)'},
                            {'range': [seuil_pct, 100], 'color': 'rgba(220,38,38,0.55)'}],
                'threshold': {'line': {'color': 'red', 'width': 6},
                                'thickness': 0.9,
                                'value': seuil_pct}}))

        fig_jauge.update_layout(paper_bgcolor='black',
                                height=400, width=350,
                                font={'color': 'white', 'family': 'Arial'},
                                margin=dict(l=0, r=0, b=0, t=0, pad=0))
        st.plotly_chart(fig_jauge)
        st.markdown(
            f"<p style='text-align:center; font-size:13px; color:#bbb;'>"
            f"Seuil de décision : <b>{seuil_pct} %</b> "
            f"— optimisé sur le coût métier (Fβ, β=10), pas sur 0,5.</p>",
            unsafe_allow_html=True)
    
    ## Globale
    col1, col2 = st.columns(2)
    if  show_global_feature_importance: 
        with col1:
            titre_global = """
            <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
            <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Globale</h2>
            </div>
            """
            st.markdown(titre_global, unsafe_allow_html=True)
            st.image("SHAP_summary_GBC.png")

    ## Locale
    if show_client_feature_importance:
        with col2:
            titre_local = f"""
            <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
            <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Client {option}</h2>
            </div>
            """
            st.markdown(titre_local, unsafe_allow_html=True)
            shap.initjs()
            # chargement des données pour lesquelles on veut calculer les valeurs Shapley
            shap_value_selected_client = result['Feature_Importance'][0] # only one vector
            Expected_value = result['Expected_value'] # only one value

            # Feature importance locale -> un seul X
            names_feature = list(df.iloc[:,:-2].columns)
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=np.array(shap_value_selected_client),
                                                base_values=Expected_value,
                                                feature_names=names_feature),
                                                max_display=number) 
            st.pyplot(fig)

    # Display Client informations
    titre_dossier = """
        <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
        <h2 style="color: white; text-align: left; font-weight: bold; font-size: 24px;">ANALYSE DOSSIER CLIENT</h2>
        </div>
        """
    st.markdown(titre_dossier, unsafe_allow_html=True)

    # Customer information display : Customer Gender, Age, Family status, Children, …
    titre_info = """
                <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">INFORMATIONS CLIENT</h2>
                </div>
                """
    st.markdown(titre_info, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    infos_client = identite_client(df_dashboard, option)
    infos_client_bank = identite_client_credit (df_dashboard, option)
    if show_client_private_info:
        with col1:
            titre_private = """
            <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
            <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Personnelles</h2>
            </div>
            """
            st.markdown(titre_private, unsafe_allow_html=True)
            info_dict = {
            "Genre": infos_client["CODE_GENDER"].values[0],
            "Age": "{:.0f} ans".format(int(infos_client["AGE"])),
            "Family Status": infos_client["NAME_FAMILY_STATUS"].values[0],
            "Nombre d'enfants": "{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]),
            "Occupation": infos_client["OCCUPATION_TYPE"].values[0],
            "Embauché(e) depuis": "{:.0f} ans".format(infos_client["YEARS_EMPLOYED"].values[0])
            }
            st.table(info_dict)

            # Age distribution plot
            data_age = df_dashboard["AGE"] 
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor = 'k', color="goldenrod")
            ax.axvline(infos_client["AGE"].values, color="red", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)

    if show_client_banq_info:
        with col2: 
            titre_bank = """
            <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
            <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Bancaires</h2>
            </div>
            """
            st.markdown(titre_bank, unsafe_allow_html=True)

            # Info income
            info_dict_bank = {
            "Revenu total": "{:.0f} USD".format(infos_client_bank["AMT_INCOME_TOTAL"].values[0]),
            "Provenance des revenus": infos_client_bank["NAME_INCOME_TYPE"].values[0],
            "Propriétaire": infos_client_bank["FLAG_OWN_REALTY"].values[0]
            }
            st.table(info_dict_bank)
           
            # Average income
            st.markdown(f"<p style='background-color: lightseagreen; padding: 5px; border-radius: 5px; font-weight: bold;'>Revenu moyen ($USD) : {rev_moy}</p>", unsafe_allow_html=True)         

            # Income distribution plot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_dashboard['AMT_INCOME_TOTAL'], edgecolor = 'k', color="goldenrod")
            ax.axvline(int(infos_client_bank["AMT_INCOME_TOTAL"].values[0]), color="red", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig)

    # Info credits en cours
    col1, col2 = st.columns(2)
    if show_client_loan_info:
        with col1:
            titre_loan_ongoing = """
                <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Crédits en cours</h2>
                </div>
                """
            st.markdown(titre_loan_ongoing, unsafe_allow_html=True)
            #class_value = infos_client_bank['CLASS_RATE_INDEBTEDNESS'].values[0]
            #class_text = ""
            #if class_value == 'DANGEROUS':
            #    class_text = '<span style="color:tomato"> **DANGER : FORTEMENT ENDETTE**</span>'
            #elif class_value == 'GOOD':
               # class_text = '<span style="color:orange">**BON : ENDETTEMENT ACCEPTABLE** </span>'
            #elif class_value == 'EXCELLENT':
            #    class_text = '<span style="color:green">**EXCELLENT : ENDETTEMENT FAIBLE** </span>'
            #else:
            #    class_text = '<span style="color:blue"> INCONNU </span>'
            dict_loan_ongoning = {
                    "Crédit en cours ?": infos_client_bank["CREDIT_ACTIVE"].values[0],
                    "Annuité": "{:.0f} USD".format(infos_client_bank["AMT_ANNUITY"].values[0]),
                    "Taux d'endettement": "{:.0f} %".format(infos_client_bank["RATE_INDEBTEDNESS"].values[0])
                    #"Note": class_text
                    }
            st.table(dict_loan_ongoning)
       
            #if (infos_client_bank["CLASS_RATE_INDEBTEDNESS"] == 'DANGEROUS').any():
            #   st.container().markdown('<span style="color:red"> **DANGER : FORTEMENT ENDETTE**</span>', unsafe_allow_html=True)
            #elif (infos_client_bank["CLASS_RATE_INDEBTEDNESS"] == 'GOOD').all():
            #   st.container().markdown('<span style="color:green"> Apport ** BON : ENDETTEMENT ACCEPTABLE** </span>', unsafe_allow_html=True)
            #elif (infos_client_bank["CLASS_RATE_INDEBTEDNESS"] == 'EXCELLENT').all():
            #   st.container().markdown('<span style="color:green"> Apport **EXCELLENT : ENDETTEMENT FAIBLE** </span>', unsafe_allow_html=True)
            #else:
            #   st.container().markdown('<span style="color:blue"> INCONNU </span>', unsafe_allow_html=True)

        with col2:
            # Info credit demandé
            titre_loan = """
                <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">Crédits demandé</h2>
                </div>
                """
            st.markdown(titre_loan, unsafe_allow_html=True)
            apport_value = infos_client_bank["APPORT"].values[0]
            apport_text = ""
            if apport_value == 'LOW':
                apport_text = '<span style="color:red"> Apport **FAIBLE**: <20% </span>'
            elif apport_value == 'ENOUGH':
                apport_text = '<span style="color:green"> Apport **SUFFISANT**: >20% </span>'
            else:
                apport_text = '<span style="color:blue"> INCONNU </span>'

            dict_credit = {
                "Montant du crédit demandé": "{:.0f} USD".format(infos_client_bank["AMT_CREDIT"].values[0]),
                "Montant du bien": "{:.0f} USD".format(infos_client_bank["AMT_GOODS_PRICE"].values[0]),
                "Rapport LTV": "{:.0f} %".format(infos_client_bank["LTV_ratio"].values[0]),
                "Apport": apport_text
            }
            #st.table(dict_credit) 
            table_html = "<table>"
            for key, value in dict_credit.items():
                table_html += "<tr><td>{}</td><td>{}</td></tr>".format(key, value)
            table_html += "</table>"    
            st.markdown(table_html,unsafe_allow_html=True)
            ## AMT CREDIT
            st.markdown(f"<p style='background-color: lightseagreen; padding: 5px; border-radius: 5px; font-weight: bold;'>Montant moyen demandé(USD): {credits_moy}</p>", unsafe_allow_html=True)         
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
    # Display Distribution feature X
    list_QUANT = ['CNT_CHILDREN','AGE','AMT_INCOME_TOTAL', 'AMT_ANNUITY','AMT_CREDIT', 'AMT_GOODS_PRICE',
                        'YEARS_EMPLOYED',  'ANCIENNETE_BANK']
    list_QUAL = ['CLASS_RATE_INDEBTEDNESS','CREDIT_ACTIVE', 'CODE_GENDER',
                'NAME_CONTRACT_TYPE', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
        'OCCUPATION_TYPE', 'NAME_FAMILY_STATUS', 'APPORT']
    list_POURC  = ['RATE_INDEBTEDNESS','LTV_ratio']
    df_dashboard[list_QUANT] = df_dashboard[list_QUANT].apply(pd.to_numeric, errors='coerce')
    titre_distribution = """
                <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">DISTRIBUTION</h2>
                </div>
                """
    st.markdown(titre_distribution, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if show_client_feature1:
        if selected_featured in list_QUAL:
            titre_feature1 = f"""
                <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">{selected_featured}</h2>
                </div>
                """
            st.markdown(titre_feature1, unsafe_allow_html=True)
            # Compter le nombre de chaque genre
            counts = df_dashboard[selected_featured].value_counts()
            # Créer le graphique 
            fig, ax = plt.subplots(figsize=(5,5))
            fig = px.pie(values=counts, names=counts.index)        
            # Afficher le graphique dans Streamlit
            st.write(X_selected_client[selected_featured].values[0])
            st.plotly_chart(fig)

        else:
            with col1:
                titre_feature1 = f"""
                    <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                    <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">{selected_featured}</h2>
                    </div>
                    """
                st.markdown(titre_feature1, unsafe_allow_html=True)  

                if selected_featured in list_QUANT:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(df_dashboard[selected_featured], edgecolor = 'k', color="goldenrod")
                    ax.axvline(int(X_selected_client[selected_featured].values[0]), color="red", linestyle='--')
                    ax.set(title=selected_featured, xlabel='', ylabel='')
                    st.pyplot(fig)

                elif selected_featured in list_POURC:
                    # visu score yield Gauge Graph
                    pourc = int(X_selected_client[selected_featured])
                    if selected_featured == 'RATE_INDEBTEDNESS':
                        fig_jauge = go.Figure(go.Indicator(
                            mode='gauge+number+delta',
                            # Customer scoring in % df_dashboard['SCORE_CLIENT_%']
                            value=pourc,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            delta={'reference': df_dashboard[selected_featured].mean(),
                                'increasing': {'color': 'Crimson'},
                                'decreasing': {'color': 'Green'}},
                            gauge={'axis': {'range': [None, 100],
                                            'tickwidth': 3,
                                            'tickcolor': 'white'},
                                'bar': {'color': 'white', 'thickness': 0.20},
                                'bgcolor': 'white',
                                'borderwidth': 2,
                                'bordercolor': 'black',
                                'steps': [{'range': [0, 30], 'color': 'LimeGreen'},
                                            {'range': [30, 33], 'color': 'Orange'},
                                            {'range': [33, 100], 'color': 'red'}],
                                'threshold': {'line': {'color': 'white', 'width': 10},
                                                'thickness': 0.8,
                                                'value': pourc}}))
                    else :
                        fig_jauge = go.Figure(go.Indicator(
                            mode='gauge+number+delta',
                            # Customer scoring in % df_dashboard['SCORE_CLIENT_%']
                            value=pourc,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            delta={'reference': df_dashboard[selected_featured].mean(),
                                'increasing': {'color': 'Crimson'},
                                'decreasing': {'color': 'Green'}},
                            gauge={'axis': {'range': [None, 2],
                                            'tickwidth': 3,
                                            'tickcolor': 'white'},
                                'bar': {'color': 'white', 'thickness': 0.20},
                                'bgcolor': 'white',
                                'borderwidth': 2,
                                    'bordercolor': 'black',
                                'steps': [{'range': [0, 0.8], 'color': 'LimeGreen'},
                                            {'range': [0.8, 1], 'color': 'Orange'},
                                            {'range': [1, 2], 'color': 'red'}],
                                'threshold': {'line': {'color': 'white', 'width': 10},
                                                'thickness': 0.8,
                                                'value': pourc}}))
                        
                    fig_jauge.update_layout(paper_bgcolor='black',
                                            height=400, width=350,
                                            font={'color': 'white', 'family': 'Arial'},
                                            margin=dict(l=0, r=0, b=0, t=0, pad=0))
                    st.plotly_chart(fig_jauge)

    if show_client_feature2:
        if selected_featured2 in list_QUAL:
            titre_feature2 = f"""
                    <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                    <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">{selected_featured2}</h2>
                    </div>
                    """
            st.markdown(titre_feature2, unsafe_allow_html=True)
            # Compter le nombre de chaque genre
            counts = df_dashboard[selected_featured2].value_counts()
            # Créer le graphique 
            fig, ax = plt.subplots(figsize=(5,5))
            fig = px.pie(values=counts, names=counts.index)        
            # Afficher le graphique dans Streamlit
            st.write(X_selected_client[selected_featured2].values[0])
            st.plotly_chart(fig)
        else:
            with col2:
                titre_feature2 = f"""
                    <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                    <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">{selected_featured2}</h2>
                    </div>
                    """
                st.markdown(titre_feature2, unsafe_allow_html=True)  
                if selected_featured2 in list_QUANT:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(df_dashboard[selected_featured2], edgecolor = 'k', color="goldenrod")
                    ax.axvline(int(X_selected_client[selected_featured2].values[0]), color="red", linestyle='--')
                    ax.set(title=selected_featured2, xlabel='', ylabel='')
                    st.pyplot(fig)

                elif selected_featured2 in list_POURC:
                    # visu score yield Gauge Graph
                    pourc = int(X_selected_client[selected_featured2])
                    if selected_featured2 == 'RATE_INDEBTEDNESS':
                        fig_jauge = go.Figure(go.Indicator(
                            mode='gauge+number+delta',
                            # Customer scoring in % df_dashboard['SCORE_CLIENT_%']
                            value=pourc,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            delta={'reference': df_dashboard[selected_featured2].mean(),
                                'increasing': {'color': 'Crimson'},
                                'decreasing': {'color': 'Green'}},
                            gauge={'axis': {'range': [None, 100],
                                            'tickwidth': 3,
                                            'tickcolor': 'white'},
                                'bar': {'color': 'white', 'thickness': 0.20},
                                'bgcolor': 'white',
                                'borderwidth': 2,
                                'bordercolor': 'black',
                                'steps': [{'range': [0, 30], 'color': 'LimeGreen'},
                                            {'range': [30, 33], 'color': 'Orange'},
                                            {'range': [33, 100], 'color': 'red'}],
                                'threshold': {'line': {'color': 'white', 'width': 10},
                                                'thickness': 0.8,
                                                'value': pourc}}))
                    else :
                        fig_jauge = go.Figure(go.Indicator(
                            mode='gauge+number+delta',
                            # Customer scoring in % df_dashboard['SCORE_CLIENT_%']
                            value=pourc,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            delta={'reference': df_dashboard[selected_featured2].mean(),
                                'increasing': {'color': 'Crimson'},
                                'decreasing': {'color': 'Green'}},
                            gauge={'axis': {'range': [None, 2],
                                            'tickwidth': 3,
                                            'tickcolor': 'white'},
                                'bar': {'color': 'white', 'thickness': 0.20},
                                'bgcolor': 'white',
                                'borderwidth': 2,
                                    'bordercolor': 'black',
                                'steps': [{'range': [0, 0.8], 'color': 'LimeGreen'},
                                            {'range': [0.8, 1], 'color': 'Orange'},
                                            {'range': [1, 2], 'color': 'red'}],
                                'threshold': {'line': {'color': 'white', 'width': 10},
                                                'thickness': 0.8,
                                                'value': pourc}}))
                        
                    fig_jauge.update_layout(paper_bgcolor='black',
                                            height=400, width=350,
                                            font={'color': 'white', 'family': 'Arial'},
                                            margin=dict(l=0, r=0, b=0, t=0, pad=0))
                    st.plotly_chart(fig_jauge)

       
        # Analyse bivarie
        if show_client_feature1 and show_client_feature2:
            titre_feature2 = f"""
                    <div style="background-color: tomato; padding: 5px; border: 1px solid #E74C3C; border-radius: 5px; width: 100%;">
                    <h2 style="color: white; text-align: center; font-weight: bold; font-size: 18px;">COMPARAISON</h2>
                    </div>
                    """
            st.markdown(titre_feature2, unsafe_allow_html=True)

            list_QUANT_full = list_QUANT.copy()
            list_QUANT_full.extend(list_POURC)
            fig, ax = plt.subplots(figsize=(10, 10))
            if selected_featured in list_QUANT_full and selected_featured2 in list_QUANT_full:        
                fig = px.scatter(df_dashboard, x=selected_featured, y=selected_featured2, 
                                    color='CODE_GENDER') 
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig)

            if selected_featured in list_QUAL and selected_featured2 in list_QUAL: 
                # Compter le nombre d'occurrences pour chaque combinaison de valeurs des deux variables
                cross_tab = pd.crosstab(df_dashboard[selected_featured], df_dashboard[selected_featured2])

                # Créer le graphique en barres groupées
                fig = px.bar(cross_tab, x=cross_tab.index, y=cross_tab.columns, title=f"Comparison of {selected_featured} and {selected_featured2}", barmode='relative')
                st.plotly_chart(fig)     
            #else:
                #if selected_featured in list_QUANT_full and selected_featured2 in list_QUAL: 
                #    fig = px.box(df_dashboard, x=selected_featured2, y=selected_featured, points='all')
                  #  st.plotly_chart(fig)
                #else:
                    #fig = px.box(df_dashboard, x=selected_featured, y=selected_featured2, points='all')
                   # st.plotly_chart(fig)
                #print("Error, veuillez choisir 2 variables du meme type")
            # if selected_featured in list_QUAL and selected_featured2 in list_QUANT_full: 
        else:
            st.markdown("<i>…</i>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
