import streamlit as st
import numpy as np
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import base64

# Configuration de la page
st.set_page_config(
    page_title="Besoin Crèche - Dimensionnement",
    page_icon="",
    layout="centered"
)

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64("logo_babilou_transparent.png")

#Style de la page 
st.markdown("""
<style>
body {
    background-color: #F9FAFB;
    font-family: "Helvetica", sans-serif;
}
h1, h2, h3 {
    color: #02378E;
}
div.stButton > button {
    background-color: #02378E;
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
    background-color: #02378E;
    color: white;
    padding: 6px;
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
st.markdown(f"""
<style>
.top-banner {{
    background-color: #02378E;
    padding: 10px 0;;
    border-radius: 10px;
    text-align: center;
}}

.top-banner img {{
    height: 35px;
    display: block;
    margin: auto;
}}
</style>

<div class="top-banner">
    <img src="data:image/png;base64,{logo_base64}">
</div>
""", unsafe_allow_html=True)

st.write(" ")

st.image( "babilou-evry-agnescolombo-119.jpg", use_container_width=True )

# Titre et descrition de l'outils
st.markdown("""
<h1 style='text-align:center;color:#02378E;'> Combien de places en crèche pour vos collaborateurs ? Faites l’estimation en 1 minute </h1>
<p style='text-align:center; color:gray; font-size:17px;'>
Obtenez en quelques clics une estimation personnalisée du nombre de places en crèche nécessaires pour accompagner vos collaborateurs.
Un outil rapide et fiable pour éclairer votre stratégie RH.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Choix du type d'entreprise ------------------
st.markdown("### Parlez-nous de votre entreprise")
type_entreprise = st.selectbox("Quelle est la taille de votre entreprise ? :",["Moins de 1 000 salariés", "Plus de 1 000 salariés"])

# ------------------ Chargement du modèle de régression quantile ------------------
model = joblib.load("preco_modele3.pkl")

# ------------------ Champs à saisir ------------------
if type_entreprise:
    st.markdown("---")
    st.markdown("### Quelques informations pour une estimation sur mesure")
    st.markdown(" Vous ne disposez pas de toutes les données ? Renseignez simplement le nombre de collaborateurs — nous nous occupons du reste.")
    col1, col2 = st.columns(2)

    with col1:
        Nb_Sal = st.number_input("Nombre de salariés", min_value=1, value=250)
        Taux_F = st.number_input("Taux de féminisation", min_value=1., max_value=100., value=50.)
        Age_moyen = st.number_input("Moyenne d'âge", min_value=1.0, max_value=100.0, value=41.0)
        #Prct_Cadre = st.number_input("Pourcentage de Cadre", min_value=0.0, max_value=100.0, value=2.0)

    with col2:
        #CA = st.number_input("Chiffre d'affaire", min_value=1.0, value=250000.0)
        Part_JE = st.number_input("Part des jeunes enfants", min_value=1.0, max_value=100.0, value=10.0)
        Prct_TpsP = st.number_input("Pourcentage du temps partiel", min_value=0.0, max_value=100.0, value=10.0)
        
    

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(" Obtenir mon estimation personnalisée", use_container_width=True)

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
        max_cap = round(avg_cap+avg_cap*0.15)
        


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







