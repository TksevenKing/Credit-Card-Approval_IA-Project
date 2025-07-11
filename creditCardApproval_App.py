
import streamlit as st
import numpy as np
import pickle


# Chargement de mon modèle entraîné
with open("credit_approval_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔍 Système de Prédiction d'Approbation de Carte de Crédit")

st.markdown("Entrez les informations du client pour prédire si la demande sera **approuvée ou refusée**.")

# Entrées utilisateur
gender = st.selectbox("Genre (0:Femme, 1:Homme)", [0, 1])  # 0 = Femme, 1 = Homme (selon encodage)
age = st.slider("Âge", 18, 100, 35)
debt = st.number_input("Dette actuelle (en milliers)", 0.0, 100.0, 5.0)
married = st.selectbox("Marié(e)  (0:oui, 1:non)", [0, 1])
bank_customer = st.selectbox("Client bancaire  (0:oui, 1:non)", [0, 1])
education = st.slider("Niveau d'étude", 0, 20, 12)
# ethnicity = st.selectbox("Origine ethnique", [0, 1, 2])
ethnicity = 2 #Une valeur par defaut car elle n'a pas n'influence pas assez la prediction d'apres mes analyse je dois l'enlever
years_employed = st.slider("Années d'emploi", 0.0, 40.0, 3.0)
prior_default = st.selectbox("Défaut antérieur   (0:oui, 1:non)", [0, 1])
employed = st.selectbox("Actuellement employé(e)  (0:oui, 1:non)", [0, 1])
credit_score = st.slider("Score de crédit", 0, 100, 50)
citizen = st.selectbox("Statut de citoyenneté  (0:oui, 1:non)", [0, 1])
income = st.number_input("Revenu mensuel (en $)", 0.0, 100000.0, 2000.0)

# Créer le vecteur de features (ordre important)
features = np.array([[gender, age, debt, married, bank_customer, education,
                      ethnicity, years_employed, prior_default, employed,
                      credit_score, citizen, income]])

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ La demande de carte de crédit est APPROUVÉE.")
    else:
        st.error("❌ La demande de carte de crédit est REFUSÉE.")
