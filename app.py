import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # <-- Ajoutez cette ligne
from cnt_platform.core import CarbonNanotube

st.set_page_config(
    page_title="CNT Dynamics Pro-Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INJECTION CSS CUSTOM (Effet Glassmorphism & Neon)
st.markdown("""
    <style>
    /* Supprimer les marges inutiles */
    .block-container { padding-top: 2rem; }
    
    /* Style des cartes de métriques */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #10b981;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
    }
    
    /* Bouton personnalisé */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #10b981;
        background-color: rgba(0, 242, 255, 0.1);
        color: #10b981;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #10b981;
        color: black;
        box-shadow: 0 0 20px #10b981;
    }
    
    /* Cacher le logo Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- TITRE AVEC EFFET ---
st.markdown("<h1 style='text-align: center; color: #10b981; border-bottom: 2px solid #10b981; padding-bottom: 10px;'>⚛️ CNT DYNAMICS & OPTICS PLATFORM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Modélisation avancée des Nanotubes de Carbone - PFE Master</p>", unsafe_allow_html=True)

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
        st.header(f"📊 Résumé du Nanotube ({n}, {m})")
        
        # On récupère les infos que votre fonction summary() affichait
        ntc_type = "Métallique" if tube.is_metallic else "Semi-conducteur"
        nb_atomes = len(tube.atoms)
        lz = tube.a_lattice
        
        # Affichage stylisé en 3 colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Type de NTC", value=ntc_type)
            st.metric(label="Chiralité", value=f"({n}, {m})")
            
        with col2:
            st.metric(label="Nombre d'atomes", value=nb_atomes)
            st.metric(label="Période Lz", value=f"{lz:.3f} Å")
            
        with col3:
            st.metric(label="Diamètre", value=f"{tube.d_t:.3f} nm")
            # Vous pouvez ajouter une autre métrique ici si besoin
            
        st.divider()
        
        # Affichage de la structure 3D juste en dessous
        st.subheader("🧊 Visualisation interactive")
        import streamlit.components.v1 as components
        view = tube.show_3d() 
        if view:
            components.html(view._make_html(), height=500)
        
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
        st.header("🏗️ Analyse de l'Élasticité")
        
        # 1. Affichage de la valeur numérique
        st.info(f"**Module d'Young calculé :** {tube.young_modulus:.2f} TPa")
        
        # 2. Affichage de la courbe (VÉRIFIEZ LE NOM DANS CORE.PY)
        # Si vous avez une fonction qui trace la courbe de tension/déformation :
        try:
            st.subheader("📈 Courbe Énergie vs Déformation")
            fig_mech = tube.show_mechanics()
            st.pyplot(fig_mech)
        except AttributeError:
            st.warning("La fonction de tracé de la courbe mécanique n'est pas encore liée ou porte un autre nom.")

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
