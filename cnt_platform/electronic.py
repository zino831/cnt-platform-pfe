import numpy as np
import matplotlib.pyplot as plt

def calculate_electronic_bands(n, m, num_k_points=1000):
    """
    Calcule la structure de bandes électroniques EXACTE d'un nanotube de carbone (n,m)
    en utilisant les vecteurs primitifs de la zone de Brillouin du graphène.
    """
    t = 2.7  # Intégrale de transfert (eV)
    
    # Paramètres géométriques
    dR = np.gcd(2*n + m, 2*m + n)
    N = 2 * (n**2 + n*m + m**2) // dR  # Nombre d'hexagones
    
    # Coefficients des vecteurs de translation
    t1 = (2*m + n) // dR
    t2 = -(2*n + m) // dR
    
    # Composantes des vecteurs réciproques K1 (circonférentiel) et K2 (axial)
    x1, y1 = -t2 / N, t1 / N
    x2, y2 = m / N, -n / N
    
    # Vecteur d'onde k 1D le long de l'axe du tube (de -0.5 à 0.5)
    k_1d = np.linspace(-0.5, 0.5, num_k_points)
    
    # Matrice pour stocker les énergies
    bands = np.zeros((num_k_points, 2 * N))
    
    # Calcul exact des énergies
    for mu in range(N):
        for i, k in enumerate(k_1d):
            # Coordonnées fractionnaires dans la base réciproque du graphène
            ka = mu * x1 + k * x2
            kb = mu * y1 + k * y2
            
            # Formule des liaisons fortes universelle du graphène
            # f(k) = |1 + exp(i*2pi*ka) + exp(i*2pi*kb)|
            val = 1 + np.exp(1j * 2 * np.pi * ka) + np.exp(1j * 2 * np.pi * kb)
            energy = t * np.abs(val)
            
            bands[i, 2*mu] = energy       # Bande de conduction
            bands[i, 2*mu + 1] = -energy  # Bande de valence
            
    # Tri pour obtenir des courbes continues
    bands = np.sort(bands, axis=1)
    
    # On remet l'axe k de -pi à pi pour l'affichage
    k_axis = k_1d * 2 * np.pi
    
    return k_axis, bands

# ... (Laissez votre fonction plot_electronics_with_dos juste en dessous) ...

def plot_electronics_with_dos(k_axis, bands, fermi_energy=0.0):
    """
    Affiche la structure de bandes électroniques et la DOS côte à côte,
    et calcule automatiquement la valeur du gap énergétique.
    """

    # --- 1. Calcul Automatique du GAP ---
    # On isole les énergies au-dessus et en-dessous de Fermi avec une petite tolérance
    # pour éviter les erreurs numériques proches de 0.
    tol = 1e-4
    conduction_energies = bands[bands > fermi_energy + tol]
    valence_energies = bands[bands < fermi_energy - tol]
    
    if len(conduction_energies) > 0 and len(valence_energies) > 0:
        cbm = np.min(conduction_energies) # Conduction Band Minimum
        vbm = np.max(valence_energies)    # Valence Band Maximum
        gap = cbm - vbm
    else:
        gap = 0.0
        
    # Détermination de la nature du tube
    if gap < 0.05:
        nature_text = "Métallique (Gap ≈ 0.00 eV)"
    else:
        nature_text = f"Semi-conducteur (Gap = {gap:.3f} eV)"

    # --- 2. Calcul du DOS électronique ---
    all_energies = np.array(bands).flatten()
    E_min, E_max = np.min(all_energies), np.max(all_energies)
    energy_grid = np.linspace(E_min - 0.5, E_max + 0.5, 1000)
    
    dos = np.zeros_like(energy_grid)
    sigma = 0.05  # Smearing
    for E in all_energies:
        dos += np.exp(-((energy_grid - E)**2) / (2 * sigma**2))
        
    # --- 3. Configuration de la Figure ---
    fig, (ax_bands, ax_dos) = plt.subplots(1, 2, figsize=(10, 6), 
                                           gridspec_kw={'width_ratios': [3, 1]}, 
                                           sharey=True)
    
    # Ajout du titre principal avec la valeur du gap !
    fig.suptitle(f"Propriétés Électroniques : {nature_text}", fontsize=16, fontweight='bold', color='darkred')
    
    # Tracé 1 : Structure de bandes
    if len(bands.shape) > 1:
        for i in range(bands.shape[1]):
            ax_bands.plot(k_axis, bands[:, i], color='royalblue', lw=1.5)
    else:
        ax_bands.plot(k_axis, bands, color='royalblue', lw=1.5)
        
    ax_bands.axhline(y=fermi_energy, color='red', linestyle='--', label='Niveau de Fermi')
    ax_bands.set_title("Structure de Bandes", fontsize=14)
    ax_bands.set_xlabel("Vecteur d'onde $k$ (1D)", fontsize=12)
    ax_bands.set_ylabel("Énergie (eV)", fontsize=12)
    ax_bands.grid(True, linestyle=':', alpha=0.6)
    ax_bands.set_xlim(np.min(k_axis), np.max(k_axis))
    
    # Tracé 2 : DOS électronique
    ax_dos.plot(dos, energy_grid, color='indigo', lw=2)
    ax_dos.fill_betweenx(energy_grid, 0, dos, color='mediumpurple', alpha=0.4)
    ax_dos.axhline(y=fermi_energy, color='red', linestyle='--')
    ax_dos.set_title("DOS Électronique", fontsize=14)
    ax_dos.set_xlabel("DOS (u.a.)", fontsize=12)
    ax_dos.grid(True, linestyle=':', alpha=0.6)
    ax_dos.set_xticks([]) 
    
    plt.ylim(-3.0, 3.0)
    
    # Ajustement pour ne pas que le titre principal (suptitle) chevauche les graphiques
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig