import streamlit as st
import numpy as np
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Besoin Crèche - Dimensionnement",
    page_icon="",
    layout="centered"
)

#Style de la page 
st.markdown("""
<style>
body {
    background-color: #F9FAFB;
    font-family: "Helvetica", sans-serif;
}
h1, h2, h3 {
    color: #003366;
}
div.stButton > button {
    background-color: #003366;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 1.1em;
}
div.stButton > button:hover {
    background-color: #004B8C;
}
.result-card {
    background-color: #DCE6F2;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
.top-banner {
    background-color: #003366;
    color: white;
    padding: 12px;
    text-align: center;
    border-radius: 0 0 15px 15px;
}
.footer {
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)
# Bande blue et et logo
st.markdown("""
        <div style="background-color:#02378E;padding:10px;border-radius:10px;text-align:center;">
        </div>
    """, unsafe_allow_html=True)

st.write(" ")
#st.markdown(""" <style> img.custom-img { height: 50px ; object-fit: cover; border-radius: 10px; } </style> """, unsafe_allow_html=True)
# Image d'accueil
st.image( "046-Creche-Babilou-Nice-Grenouilleres-10.jpg", use_container_width=True )

# Titre et descrition de l'outils
st.markdown("""
<h1 style='text-align:center;'> Outil de dimensionnement des besoins en crèche</h1>
<p style='text-align:center; color:gray; font-size:17px;'>
Obtenez une estimation du nombre de places à réserver en crèche selon le profil de votre entreprise.
Un outil simple et intuitif pour orienter vos choix.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Choix du type d'entreprise ------------------
st.markdown("### Choisissez la taille de l'entreprise")
type_entreprise = st.selectbox("Sélectionnez la taille de l'entreprise :",["moins de 1000", "plus de 1000"])

# ------------------ Chargement du modèle de régression quantile ------------------
model = joblib.load("preco_modele3.pkl")

# ------------------ Champs à saisir ------------------
if type_entreprise:
    st.markdown("---")
    st.markdown("#### Entrez les caractéristiques de votre entreprise")
    st.markdown("Si vous ne connaissez pas les caractéristiques, veuillez entrer uniquement le nombre de salariés et laissez les autres valeurs par défaut.")
    col1, col2 = st.columns(2)

    with col1:
        Nb_Sal = st.number_input("Nombre de salariés", min_value=1, value=250)
        Taux_F = st.number_input("Taux de féminisation", min_value=1.0, max_value=100.0, value=50.0)
        Age_moyen = st.number_input("Moyenne d'âge", min_value=1.0, max_value=100.0, value=41.0)
        #Prct_Cadre = st.number_input("Pourcentage de Cadre", min_value=0.0, max_value=100.0, value=2.0)

    with col2:
        #CA = st.number_input("Chiffre d'affaire", min_value=1.0, value=250000.0)
        Part_JE = st.number_input("Part des jeunes enfants", min_value=1.0, max_value=100.0, value=10.0)
        Prct_TpsP = st.number_input("Pourcentage du temps partiel", min_value=0.0, max_value=100.0, value=10.0)
        
    

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Estimer le besoin en places", use_container_width=True)

    # ------------------ Prédiction du besoin ------------------
    if predict_btn:

        seg_val = 1 if type_entreprise == "plus de 1000" else 0

        X_df = pd.DataFrame({
            "Nb_Sal": [Nb_Sal],
            "Seg_Entreprise": [seg_val],
            "Prct_TpsP": [Prct_TpsP],
            "Part_JE": [Part_JE],
            "Taux_F": [Taux_F],
            "Age_moyen": [Age_moyen]
        })
        pred = model.get_prediction(X_df).summary_frame(alpha=0.05)

        avg_cap = round(np.expm1(pred["mean"].iloc[0]))
        min_cap = round(avg_cap-avg_cap*0.10)
        max_cap = round(avg_cap+avg_cap*0.20)
        


        # ------------------ Affichage du résultat ------------------
        st.markdown("---")
        st.markdown("#### Estimation du besoin en crèche")

        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:#003366;'>Besoin estimé : <b>{avg_cap} places</b></h2>
            <p style='font-size:17px; color:#1E3A5F;'>
           Pour des entreprises au profil similaire, le besoin moyen est d’environ {avg_cap} places,
            avec une estimation variant entre {min_cap} et {max_cap} places.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(avg_cap / 300, 1.0))

        st.markdown("""
        <div style='text-align:center; color:gray; margin-top:10px;'>
            <em>Cette estimation repose sur les paramètres saisis.</em><br>
            Modifiez les valeurs pour simuler d'autres profils d'entreprise.
        </div>
        """, unsafe_allow_html=True)

## fin de page
st.markdown("""
<div class='footer'>
Réalisé par Thierno | Outil de dimensionnement © 2025
</div>
""", unsafe_allow_html=True)
