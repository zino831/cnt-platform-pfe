import numpy as np
import matplotlib.pyplot as plt
# Importation des modules spécialisés
from .geometry import generate_structure, plot_3d_structure
from .electronic import calculate_electronic_bands, plot_electronics_with_dos
from .mechanics import calculate_young_modulus, plot_mechanics_graph
from .optics import generate_raman_and_ir
from .thermal import calculate_phonons, calculate_cv, plot_thermal_cv, construire_matrice_dynamique_pbc, determiner_mode_rbm

class CarbonNanotube:
    """
    Classe Chef d'orchestre : Elle lie la géométrie, l'électronique 
    et les autres propriétés.
    """
    
    def __init__(self, n, m, length=1):
        self.n = n
        self.m = m
        self.length = length
        self.is_metallic = (n - m) % 3 == 0
        
        # Initialisation des attributs de données
        self.atoms = None
        self.a_lattice = None
        self.bands = None
        self.q_array = None
        
        # 1. On génère la géométrie dès la création de l'objet
        self._generate_geometry()

    def _generate_geometry(self):
        """Appelle le module geometry.py"""
        self.atoms, self.a_lattice = generate_structure(self.n, self.m, self.length)

    def show_3d(self):
        """Retourne la structure 3D interactive pour Streamlit"""
        if self.atoms is not None:
            return plot_3d_structure(self.atoms)  # On retourne la figure au lieu de faire .show()

    def compute_electronics(self, n_points=100):
        """Appelle le module electronic.py pour le calcul des bandes"""
        # Création de l'espace réciproque (q)
        self.q_array = np.linspace(0, np.pi/self.a_lattice, n_points)
        
       # Calcul via la fonction externe avec les bons arguments (n, m)
        self.elec_k_points, self.elec_bands = calculate_electronic_bands(
            self.n, 
            self.m, 
            num_k_points=1000
        )
        print(f"✅ Calcul électronique terminé pour ({self.n},{self.m})")

    def show_electronics(self):
        """Affiche les bandes électroniques et le DOS."""
        # On vérifie les nouvelles variables spécifiques
        if hasattr(self, 'elec_k_points') and hasattr(self, 'elec_bands'):
            plot_electronics_with_dos(self.elec_k_points, self.elec_bands, fermi_energy=0.0)
        else:
            print("❌ Erreur : Veuillez lancer compute_electronics() d'abord.")

    def summary(self):
        """Résumé des propriétés de l'objet"""
        ntc_type = "Métallique" if self.is_metallic else "Semi-conducteur"
        print(f"--- Propriétés du Nanotube ({self.n}, {self.m}) ---")
        print(f"Type : {ntc_type}")
        print(f"Nombre d'atomes : {len(self.atoms)}")
        print(f"Période spatiale Lz : {self.a_lattice:.3f} Å")
        print("-" * 40)

    def compute_mechanics(self):
        self.strains, self.energies, self.coefs, self.young_modulus = calculate_young_modulus(self.atoms, self.a_lattice)
        print(f"✅ Module de Young : {self.young_modulus:.0f} GPa")

    def show_mechanics(self):
        plot_mechanics_graph(self.strains, self.energies, self.coefs, self.young_modulus)

    def compute_optics(self):
        """Calcule les spectres Raman et IR à partir des phonons"""
        if hasattr(self, 'phonons'):
            # On utilise le module optics importé en haut du fichier
            from .optics import generate_raman_and_ir
            self.w_axis, self.raman, self.ir = generate_raman_and_ir(self)
            print("✅ Spectres Raman/IR générés")
        else:
            print("❌ Calculez d'abord la thermique (phonons) !")

    def show_optics(self):
        """Génère et retourne les spectres Raman et Infrarouge pour Streamlit."""
        
        # 1. On s'assure que les calculs sont faits
        if not hasattr(self, 'raman'):
            self.compute_optics()
            
        # 2. On appelle la fonction de tracé (elle doit créer un plot)
        # Note : Si generate_raman_and_ir crée déjà une figure, plt.gcf() la récupérera
        generate_raman_and_ir(self)
        
        # 3. TRÈS IMPORTANT : On retourne la figure actuelle
        fig = plt.gcf()
        return fig
    def compute_thermal(self, t_min=1, t_max=1000):
        self.q_thermal = np.linspace(0, np.pi/self.a_lattice, 20)
        self.phonons = calculate_phonons(self.atoms, self.a_lattice, self.q_thermal)
        
        self.temps = np.linspace(t_min, t_max, 100)
        self.cv = calculate_cv(self.phonons, self.temps)
        print("✅ Propriétés thermiques (Phonons & Cv) calculées.")

    def show_thermal(self):
    
        if hasattr(self, 'cv'):
            plot_thermal_cv(self.temps, self.cv)
        else:
            print("❌ Erreur : Calculez d'abord la thermique.")
    def compute_saito_dynamics(self):
        """Calcule les modes RBM et G avec les paramètres de Saito"""
        # On importe les fonctions ici au cas où
        import numpy as np
        from .thermal import construire_matrice_dynamique_pbc, determiner_mode_rbm
        
        # Calcul de la matrice
        D = construire_matrice_dynamique_pbc(self.atoms, q_z=0)
        eigenvalues, eigenvectors = np.linalg.eigh(D)
        
        # ON ATTACHE TOUT À "self." POUR SAUVEGARDER EN MÉMOIRE
        self.wavenumbers_saito = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi * 2.99792458e10)
        
        pos = self.atoms.get_positions()
        idx, self.f_rbm, self.rbm_score = determiner_mode_rbm(pos, eigenvectors, self.wavenumbers_saito)
        self.d_t = (np.mean(np.sqrt(pos[:,0]**2 + pos[:,1]**2)) * 2) / 10
        
        print(f"✅ Dynamique de Saito calculée avec succès !")

    def show_saito_dynamics(self):
        """Affiche les résultats de la dynamique de Saito"""
        if hasattr(self, 'f_rbm'):
            print("\n" + "="*55)
            print(f"ANALYSE DU MODE RBM (SAITO - PBC)")
            print(f"Diamètre            : {self.d_t:.3f} nm")
            print(f"Fréquence RBM       : {self.f_rbm:.2f} cm⁻¹")
            print(f"Fréquence théorique : ~{227/self.d_t:.2f} cm⁻¹")
            print(f"Qualité (Overlap)   : {self.rbm_score*100:.2f} %")
            print(f"Fréquences Bande G  : {self.wavenumbers_saito[-4]:.2f}, {self.wavenumbers_saito[-1]:.2f} cm⁻¹")
            print("="*55)
        else:
            print("❌ Calculez d'abord compute_saito_dynamics()")
    def compute_dispersion(self, n_points_q=100):
        """Calcule les courbes de dispersion et la DOS sur la Zone de Brillouin."""
        import numpy as np
        from .thermal import construire_matrice_dynamique_pbc
        
        self.a_lattice = self.atoms.get_cell()[2, 2] 
        self.q_space = np.linspace(0, np.pi / self.a_lattice, n_points_q)
        
        print(f"Calcul de la dispersion pour {n_points_q} points q...")
        all_frequencies = []
        
        for qz in self.q_space:
            D_q = construire_matrice_dynamique_pbc(self.atoms, q_z=qz)
            vals = np.linalg.eigvalsh(D_q)
            freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi * 2.99792458e10)
            all_frequencies.append(sorted(freqs))
            
        self.all_frequencies = np.array(all_frequencies)
        print("✅ Dispersion calculée avec succès !")

    def plot_dispersion_and_dos(self):
        """Trace les courbes de dispersion et la Densité d'États (DOS)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Vérification de sécurité
        if not hasattr(self, 'all_frequencies'):
            print("❌ Erreur : Veuillez d'abord lancer compute_dispersion()")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True, 
                                       gridspec_kw={'width_ratios': [3, 1]})

        # A. Tracé de la Relation de Dispersion
        q_norm = self.q_space * self.a_lattice / np.pi
        for i in range(self.all_frequencies.shape[1]):
            ax1.plot(q_norm, self.all_frequencies[:, i], 
                     color='midnightblue', alpha=0.7, lw=0.8)

        ax1.set_title(f"Relation de Dispersion - CNT ({self.n},{self.m})", fontsize=12)
        ax1.set_xlabel(r"Vecteur d'onde $q_z$ ($\pi/a$)")
        ax1.set_ylabel("Fréquence (cm⁻¹)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1700) # Coupe la fréquence max autour de la bande G
        ax1.grid(True, linestyle=':', alpha=0.6)

        # B. Tracé de la Densité d'États (DOS)
        flat_freqs = self.all_frequencies.flatten()
        ax2.hist(flat_freqs, bins=200, orientation='horizontal', 
                 color='royalblue', alpha=0.8, density=True)

        ax2.set_title("DOS (Phonons)", fontsize=12)
        ax2.set_xlabel("Intensité (u.a.)")
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        #plt.show()
        return plt.gcf()
        
        # --- BLOC PROPRIÉTÉS OPTIQUES ---
        st.subheader("7. Propriétés Optiques")
        
        # Affichage du graphique
        fig_optics = tube.plot_optics()
        if fig_optics is not None:
            st.pyplot(fig_optics)
            plt.clf()
            
        # Affichage des valeurs exactes
        if hasattr(tube, 'transitions'):
            st.markdown("**Énergies de Transition (Règle de Kataura) :**")
            cols_opt = st.columns(len(tube.transitions))
            
            for i, (nom, energie) in enumerate(tube.transitions.items()):
                # Conversion approchée de l'énergie (eV) en longueur d'onde (nm)
                longueur_onde = 1240 / energie 
                cols_opt[i].metric(f"Transition {nom}", f"{energie:.2f} eV", f"~ {longueur_onde:.0f} nm")
