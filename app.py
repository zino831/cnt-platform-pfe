import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # <-- Ajoutez cette ligne
from cnt_platform.core import CarbonNanotube

# Configuration de la page
st.set_page_config(page_title="CNT Research Platform", layout="wide")

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.title("🔬 Paramètres & Menu")

# 1. Choix des indices (n, m)
n = st.sidebar.number_input("Indice n", value=10, min_value=1)
m = st.sidebar.number_input("Indice m", value=0, min_value=0)

# 2. Bouton de calcul
if st.sidebar.button("Lancer les calculs"):
    st.session_state.calc_done = True

# 3. MENU DE NAVIGATION
st.sidebar.divider()
option = st.sidebar.radio(
    "Choisir l'analyse :",
    ["Résumé & Structure", "Propriétés Électroniques", "Propriétés Mécaniques", "Thermique & Phonons", "Spectroscopie Optique"]
)

# --- LOGIQUE DE CALCUL ---
if 'calc_done' in st.session_state:
    # On crée le tube une seule fois
    tube = CarbonNanotube(n=n, m=m, length=1)
    
    # On effectue tous les calculs en arrière-plan
    with st.spinner("Calcul des propriétés en cours..."):
        tube.compute_electronics()
        tube.compute_mechanics()
        tube.compute_thermal()
        tube.compute_saito_dynamics()
        tube.compute_dispersion(n_points_q=50)
        tube.compute_optics()

    # --- AFFICHAGE SELON L'OPTION CHOISIE ---

    if option == "Résumé & Structure":
        st.header("📊 Résumé du Nanotube")
        col1, col2, col3 = st.columns(3)
        col1.metric("Diamètre", f"{tube.d_t:.3f} nm")
        col2.metric("Chiralité", f"({n}, {m})")
        col3.metric("Type", "Métallique" if (n-m)%3==0 else "Semi-conducteur")
        
        st.subheader("🧊 Structure 3D")
        # Ici votre code pour afficher la structure 3D
        view = tube.show_3d() 
        
        # On convertit l'objet py3Dmol en HTML pour que Streamlit puisse l'afficher
        if view is not None:
            components.html(view._make_html(), height=450, scrolling=False)
        else:
            st.warning("La vue 3D n'a pas pu être générée.") 

    elif option == "Propriétés Électroniques":
        st.header("⚡ Structure de Bandes & DOS")
        fig_elec = tube.show_electronics()
        st.pyplot(fig_elec)

    elif option == "Propriétés Mécaniques":
        st.header("🏗️ Constantes Élastiques")
        # Exemple d'affichage des résultats mécaniques
        st.write(f"**Module d'Young :** {tube.young_modulus:.2f} TPa")
        tube.show_mechanics()

    elif option == "Thermique & Phonons":
        st.header("🔥 Dynamique des Phonons")
        fig_disp = tube.plot_dispersion_and_dos()
        st.pyplot(fig_disp)
        
        st.divider()
        st.subheader("Analyse de Saito (RBM)")
        # On affiche les métriques de Saito qu'on avait préparées
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Fréquence RBM", f"{tube.f_rbm:.2f} cm⁻¹")
        col_s2.metric("Qualité (Overlap)", f"{tube.rbm_score*100:.2f} %")

    elif option == "Spectroscopie Optique":
        st.header("🌈 Spectres Raman & IR")
        fig_raman = tube.show_optics()
        if fig_raman:
            st.pyplot(fig_raman)

else:
    st.info("👋 Bienvenue ! Réglez les indices (n, m) dans la barre latérale et cliquez sur 'Lancer les calculs'.")
