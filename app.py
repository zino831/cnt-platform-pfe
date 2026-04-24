import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # <-- Ajoutez cette ligne
from cnt_platform.core import CarbonNanotube

# --- Configuration de la page ---
st.set_page_config(page_title="Simulateur NTC", layout="wide")

st.title("🧪 Plateforme de Simulation des Nanotubes de Carbone")
st.markdown("Explorez toutes les propriétés physiques de votre nanotube en un clic !")

# --- BARRE LATÉRALE (Paramètres) ---
st.sidebar.header("Paramètres Géométriques")

n = st.sidebar.number_input("Indice de chiralité n", min_value=0, max_value=30, value=10, step=1)
m = st.sidebar.number_input("Indice de chiralité m", min_value=0, max_value=n, value=5, step=1)

# Bouton pour lancer le calcul
if st.sidebar.button("Lancer la Simulation"):
    
    with st.spinner(f"Calculs de tous les modules en cours pour le tube ({n},{m}). Veuillez patienter..."):
        
        # --- 1. INITIALISATION ET CALCULS ---
        tube = CarbonNanotube(n=n, m=m, length=1)
        
        tube.compute_electronics()
        tube.compute_mechanics()
        tube.compute_thermal()
        tube.compute_saito_dynamics()
        tube.compute_dispersion(n_points_q=50)
        tube.compute_optics()
        
        st.success("Tous les calculs sont terminés ! 🎉")
        
        # --- 2. AFFICHAGE DES RÉSULTATS ---
        st.header(f"Résultats Complets pour le Nanotube ({n}, {m})")
    # --- BLOC MODÈLE 3D (Version native avec components) ---
        st.subheader("1. Modèle 3D du Nanotube")
        
        # On récupère l'objet py3Dmol renvoyé par votre fonction
        fig_3d = tube.show_3d() 
        
        if fig_3d is not None:
            try:
                # py3Dmol possède une fonction interne pour générer le code HTML interactif
                html_3d = fig_3d._make_html()
                
                # Streamlit lit ce HTML et l'affiche comme une vraie fenêtre 3D !
                components.html(html_3d, width=800, height=600)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage 3D : {e}")
        
        
        
       
        st.subheader("2. Structure Électronique")
        tube.show_electronics()
        st.pyplot(plt.gcf())
        plt.clf() # Nettoie la figure pour éviter les superpositions
            
        st.subheader("4. Propriétés Thermiques")
        tube.show_thermal()
        st.pyplot(plt.gcf())
        plt.clf()

        
        st.subheader("3. Propriétés Mécaniques")
        tube.show_mechanics()
        st.pyplot(plt.gcf())
        plt.clf()
            
        # --- BLOC DYNAMIQUE DE SAITO ---
        st.subheader("4. Dynamique de Saito (Analyse RBM)")
        
        # On exécute la fonction pour qu'elle affiche le texte dans le terminal (optionnel)
        tube.show_saito_dynamics() 
        
        # On récupère les variables exactes de votre core.py pour les afficher sur le Web !
        if hasattr(tube, 'f_rbm'):
            col_s1, col_s2 = st.columns(2)
            
            # Affichage de belles boîtes de métriques
            col_s1.metric("Diamètre du tube", f"{tube.d_t:.3f} nm")
            col_s1.metric("Fréquence RBM", f"{tube.f_rbm:.2f} cm⁻¹")
            
            col_s2.metric("Fréquence théorique", f"{227/tube.d_t:.2f} cm⁻¹")
            col_s2.metric("Qualité (Overlap)", f"{tube.rbm_score*100:.2f} %")
            
            # Affichage de la bande G en texte normal
            st.markdown(f"**Fréquences Bande G :** {tube.wavenumbers_saito[-4]:.2f} cm⁻¹ et {tube.wavenumbers_saito[-1]:.2f} cm⁻¹")
        else:
            st.warning("Les données de la dynamique de Saito n'ont pas été trouvées.")

        # Affichage en pleine largeur pour les graphiques plus larges
        st.subheader("6. Dispersion des Phonons")
        tube.plot_dispersion_and_dos()
        st.pyplot(plt.gcf())
        plt.clf()
        
        # --- BLOC SPECTROSCOPIE (RAMAN & IR) ---
        st.header("7. Spectroscopie Optique")
        col_opt1, col_opt2 = st.columns([3, 1])
        
        with col_opt1:
            fig_raman = tube.show_optics()
            if fig_raman is not None:
                st.pyplot(fig_raman)
                plt.clf()
        
        with col_opt2:
            st.info("💡 Le spectre Raman montre les modes caractéristiques : Bande G, Bande D et RBM.")
            if hasattr(tube, 'raman'):
                st.write("Calcul terminé sur l'axe :")
                st.write(f"{min(tube.w_axis):.0f} à {max(tube.w_axis):.0f} cm⁻¹")
        
        