import numpy as np
import matplotlib.pyplot as plt
from .thermal import construire_matrice_dynamique_pbc

def generate_raman_and_ir(tube, gamma_width=10.0):
    """
    Génère les spectres Raman et Infrarouge (IR) empiriques pour un nanotube.
    """
    # 1. Extraction des fréquences au centre de la zone (q=0)
    D_gamma = construire_matrice_dynamique_pbc(tube.atoms, q_z=0.0)
    vals = np.linalg.eigvalsh(D_gamma)
    
    # Conversion en cm^-1
    freqs_gamma = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi * 2.99792458e10)
    freqs_gamma = np.sort(freqs_gamma)
    
    # 2. Axe des nombres d'onde (de 100 à 1800 cm^-1)
    x_wavenumbers = np.linspace(100, 1800, 2000)
    raman_intensities = np.zeros_like(x_wavenumbers)
    ir_intensities = np.zeros_like(x_wavenumbers)
    
    def lorentzian(x, x0, gamma, A):
        return A * (gamma**2 / ((x - x0)**2 + gamma**2))

    # 3. Attribution empirique des intensités selon les règles de sélection
    for w in freqs_gamma:
        if w < 10: 
            continue # On ignore les translations pures
            
        int_raman = 0.02 # Bruit de fond Raman
        int_ir = 0.02    # Bruit de fond IR
        
        # --- Règles RAMAN ---
        if 150 < w < 400:     # Mode RBM (Respiration radiale)
            int_raman = 1.0
        elif 1550 < w < 1620: # Bande G (Étirement symétrique C-C)
            int_raman = 0.8
            
        # --- Règles INFRAROUGE (IR) ---
        if 800 < w < 900:     # Déformation asymétrique hors-plan
            int_ir = 0.6
        elif 1500 < w < 1550: # Étirement asymétrique C-C
            int_ir = 0.9

        # Ajout aux spectres
        raman_intensities += lorentzian(x_wavenumbers, w, gamma_width, int_raman)
        ir_intensities += lorentzian(x_wavenumbers, w, gamma_width, int_ir)

    # 4. Affichage des deux spectres
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Tracé Raman
    ax1.plot(x_wavenumbers, raman_intensities, color='darkgreen', lw=2)
    ax1.fill_between(x_wavenumbers, raman_intensities, color='lightgreen', alpha=0.3)
    ax1.set_title(f"Spectre Raman Simulé - CNT ({tube.n},{tube.m})", fontsize=14)
    ax1.set_ylabel("Intensité Raman (u.a.)", fontsize=12)
    # Lignes de repère typiques pour un CNT
    ax1.axvline(x=283, color='gray', linestyle='--', alpha=0.5)
    ax1.text(285, 0.8, 'RBM', color='gray')
    ax1.axvline(x=1580, color='gray', linestyle='--', alpha=0.5)
    ax1.text(1585, 0.6, 'Bande G', color='gray')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Tracé IR
    ax2.plot(x_wavenumbers, ir_intensities, color='darkred', lw=2)
    ax2.fill_between(x_wavenumbers, ir_intensities, color='lightcoral', alpha=0.3)
    ax2.set_title(f"Spectre Infrarouge (IR) Simulé - CNT ({tube.n},{tube.m})", fontsize=14)
    ax2.set_xlabel("Nombre d'onde (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Absorbance IR (u.a.)", fontsize=12)
    # Lignes de repère
    ax2.axvline(x=850, color='gray', linestyle='--', alpha=0.5)
    ax2.text(855, 0.4, 'Hors-plan', color='gray')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(100, 1750)
    plt.tight_layout()
    #plt.show()
    return x_wavenumbers, raman_intensities, ir_intensities