# =============================================================================
# CNT SIMULATION HUB - CORE ENGINE (Fichier Unique)
# =============================================================================

# --- IMPORTS GLOBAUX ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ase.build import nanotube
from ase.io import write
import py3Dmol
import io


# =============================================================================
# 1. MODULE GÉOMÉTRIE
# =============================================================================
def generate_structure(n, m, length=1):
    atoms = nanotube(n, m, length=length)
    a_lattice = atoms.get_cell()[2, 2]
    return atoms, a_lattice

def plot_3d_structure(atoms):
    xyz_file = io.StringIO()
    write(xyz_file, atoms, format='xyz')
    xyz_string = xyz_file.getvalue()

    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, 'xyz')
    view.setStyle({'stick': {'color': 'grey', 'radius': 0.15}, 
                   'sphere': {'scale': 0.25, 'color': 'midnightblue'}})
    view.zoomTo()
    return view


# =============================================================================
# 2. MODULE ÉLECTRONIQUE
# =============================================================================
def calculate_electronic_bands(n, m, num_k_points=1000):
    t = 2.7  
    dR = np.gcd(2*n + m, 2*m + n)
    N = 2 * (n**2 + n*m + m**2) // dR  
    t1 = (2*m + n) // dR
    t2 = -(2*n + m) // dR
    x1, y1 = -t2 / N, t1 / N
    x2, y2 = m / N, -n / N
    k_1d = np.linspace(-0.5, 0.5, num_k_points)
    bands = np.zeros((num_k_points, 2 * N))
    
    for mu in range(N):
        for i, k in enumerate(k_1d):
            ka = mu * x1 + k * x2
            kb = mu * y1 + k * y2
            val = 1 + np.exp(1j * 2 * np.pi * ka) + np.exp(1j * 2 * np.pi * kb)
            energy = t * np.abs(val)
            bands[i, 2*mu] = energy       
            bands[i, 2*mu + 1] = -energy  
            
    bands = np.sort(bands, axis=1)
    k_axis = k_1d * 2 * np.pi
    return k_axis, bands

def plot_electronics_with_dos(k_axis, bands, fermi_energy=0.0):
    tol = 1e-4
    conduction_energies = bands[bands > fermi_energy + tol]
    valence_energies = bands[bands < fermi_energy - tol]
    
    if len(conduction_energies) > 0 and len(valence_energies) > 0:
        gap = np.min(conduction_energies) - np.max(valence_energies)
    else:
        gap = 0.0
        
    nature_text = "Métallique (Gap ≈ 0.00 eV)" if gap < 0.05 else f"Semi-conducteur (Gap = {gap:.3f} eV)"

    all_energies = np.array(bands).flatten()
    energy_grid = np.linspace(np.min(all_energies) - 0.5, np.max(all_energies) + 0.5, 1000)
    
    dos = np.zeros_like(energy_grid)
    for E in all_energies:
        dos += np.exp(-((energy_grid - E)**2) / (2 * 0.05**2))
        
    plt.style.use('dark_background')
    fig, (ax_bands, ax_dos) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    fig.suptitle(f"Propriétés Électroniques : {nature_text}", fontsize=16, color='#10b981')
    
    if len(bands.shape) > 1:
        for i in range(bands.shape[1]):
            ax_bands.plot(k_axis, bands[:, i], color='#00f2ff', lw=1.5)
    else:
        ax_bands.plot(k_axis, bands, color='#00f2ff', lw=1.5)
        
    ax_bands.axhline(y=fermi_energy, color='#e11d48', linestyle='--')
    ax_bands.set_title("Structure de Bandes")
    ax_bands.set_xlabel("Vecteur d'onde $k$ (1D)")
    ax_bands.set_ylabel("Énergie (eV)")
    ax_bands.grid(True, alpha=0.2)
    
    ax_dos.plot(dos, energy_grid, color='#8b5cf6', lw=2)
    ax_dos.fill_betweenx(energy_grid, 0, dos, color='#8b5cf6', alpha=0.4)
    ax_dos.axhline(y=fermi_energy, color='#e11d48', linestyle='--')
    ax_dos.set_title("DOS Électronique")
    ax_dos.grid(True, alpha=0.2)
    ax_dos.set_xticks([]) 
    
    plt.ylim(-3.0, 3.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# =============================================================================
# 3. MODULE MÉCANIQUE
# =============================================================================
def calculate_young_modulus(atoms, a_lattice):
    coords0 = atoms.get_positions()
    k_bond = 29.3  
    r0 = 1.42      
    strains = np.linspace(-0.02, 0.02, 15)
    energies = []
    
    for epsilon in strains:
        coords_strained = coords0.copy()
        coords_strained[:, 2] *= (1.0 + epsilon)
        L_strained = a_lattice * (1.0 + epsilon)
        
        coords_plus = coords_strained.copy()
        coords_plus[:, 2] += L_strained
        
        dist_0 = distance_matrix(coords_strained, coords_strained)
        dist_plus = distance_matrix(coords_strained, coords_plus)
        
        bonds_0 = dist_0[(dist_0 > 0.1) & (dist_0 < 1.8)]
        bonds_plus = dist_plus[(dist_plus > 0.1) & (dist_plus < 1.8)]
        
        E_0 = np.sum(0.5 * k_bond * (bonds_0 - r0)**2) / 2.0
        E_plus = np.sum(0.5 * k_bond * (bonds_plus - r0)**2)
        energies.append(E_0 + E_plus)
        
    energies = np.array(energies) - np.min(energies)
    coefs = np.polyfit(strains, energies, 2)
    
    rayon = np.mean(np.sqrt(coords0[:, 0]**2 + coords0[:, 1]**2))
    volume = 2 * np.pi * rayon * a_lattice * 3.35
    Y_GPa = (2 * coefs[0] / volume) * 160.2176
    return strains, energies, coefs, Y_GPa

def plot_mechanics_graph(strains, energies, coefs, Y_GPa):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 4))
    plt.plot(strains*100, energies, 'o', color='#00f2ff')
    plt.plot(strains*100, np.polyval(coefs, strains), color='#e11d48', lw=2)
    plt.title(f"Module de Young : {Y_GPa/1000:.2f} TPa", color='#10b981')
    plt.xlabel("Déformation (%)")
    plt.ylabel("Énergie (eV)")
    plt.grid(alpha=0.2)
    return fig


# =============================================================================
# 4. MODULE THERMIQUE ET DYNAMIQUE
# =============================================================================
def generer_neighbor_list_pbc(pos_atoms, lattice_constant, acc=1.42):
    n_atoms = len(pos_atoms)
    dist_theoriques = [acc, np.sqrt(3)*acc, 2*acc, np.sqrt(7)*acc]
    neighbor_list = []
    for i in range(n_atoms):
        atom_neighbors = []
        for cell_shift in [-2, -1, 0, 1, 2]:
            T_vec = np.array([0, 0, cell_shift * lattice_constant])
            for j in range(n_atoms):
                R_ij = (pos_atoms[j] + T_vec) - pos_atoms[i]
                dist = np.linalg.norm(R_ij)
                if dist < 0.1: continue 
                rank = 0
                for r, d_target in enumerate(dist_theoriques):
                    if abs(dist - d_target) < 0.2:
                        rank = r + 1
                        break
                if rank > 0:
                    atom_neighbors.append({'idx': j, 'rank': rank, 'R': R_ij, 'dist': dist})
        neighbor_list.append(atom_neighbors)
    return neighbor_list

def construire_matrice_dynamique_pbc(atoms_obj, q_z, mass=12.011):
    pos = atoms_obj.get_positions()
    n_atoms = len(pos)
    lattice_constant = atoms_obj.get_cell()[2, 2]
    m_carbon = mass / (6.022e23 * 1e3) 
    
    dyn_mat = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=complex)
    neighbor_list = generer_neighbor_list_pbc(pos, lattice_constant)
    
    phi_params = {
        1: {'r': 365.0, 'ti': 245.0}, 2: {'r': 88.0,  'ti': -32.3},
        3: {'r': 30.0,  'ti': -52.5}, 4: {'r': -19.0, 'ti': 22.9}
    }
    
    for i in range(n_atoms):
        for neighbor in neighbor_list[i]:
            j, rank, R, dist = neighbor['idx'], neighbor['rank'], neighbor['R'], neighbor['dist']
            u = R / dist
            fr, fti = phi_params[rank]['r'], phi_params[rank]['ti']
            K_local = (fr - fti) * np.outer(u, u) + fti * np.eye(3)
            phase = np.exp(1j * q_z * R[2])
            term = (K_local / m_carbon) * phase
            dyn_mat[3*i:3*i+3, 3*j:3*j+3] -= term
            dyn_mat[3*i:3*i+3, 3*i:3*i+3] += (K_local / m_carbon)
            
    return dyn_mat


# =============================================================================
# 5. MODULE OPTIQUE (MÉTHODES INTÉGRÉES DANS LA CLASSE)
# =============================================================================

class CarbonNanotube:
    """Classe principale liant toutes les propriétés physiques."""
    def __init__(self, n, m, length=1):
        self.n = n
        self.m = m
        self.length = length
        self.is_metallic = (n - m) % 3 == 0
        self.atoms, self.a_lattice = generate_structure(self.n, self.m, self.length)

    def show_3d(self):
        return plot_3d_structure(self.atoms)

    # --- ÉLECTRONIQUE ---
    def compute_electronics(self):
        self.elec_k_points, self.elec_bands = calculate_electronic_bands(self.n, self.m)

    def plot_electronics(self):
        if hasattr(self, 'elec_k_points'):
            return plot_electronics_with_dos(self.elec_k_points, self.elec_bands)

    # --- MÉCANIQUE ---
    def compute_mechanics(self):
        self.strains, self.energies, self.coefs, self.young_modulus = calculate_young_modulus(self.atoms, self.a_lattice)
        self.young_modulus_tpa = self.young_modulus / 1000 # Conversion en TPa

    def plot_mechanics(self):
        if hasattr(self, 'strains'):
            return plot_mechanics_graph(self.strains, self.energies, self.coefs, self.young_modulus)

    # =========================================================================
    # --- VOS NOUVELLES FONCTIONS D'OPTIQUE (IR & RAMAN) ---
    # =========================================================================

    def plot_ir_saito(self):
        """
        Spectre Infrarouge selon le modèle de Saito (VOTRE CODE EXACT)
        """
        # --- 1. CONFIGURATION DYNAMIQUE ---
        pos = self.atoms.get_positions()
        n_atoms = len(pos)
        nt = 3 * n_atoms 

        # --- 2. ATTRIBUTION DES CHARGES ---
        qc = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_atoms)])

        # --- 3. DYNAMIQUE ET RÉPONSE ---
        D = construire_matrice_dynamique_pbc(self.atoms, q_z=0)
        eigenvalues, eigenvectors = np.linalg.eigh(D)
        fre_rad = np.sqrt(np.maximum(eigenvalues.real, 0))
        amassc = 12.011 / (6.022e23 * 1e3) 

        id_pts = 3000
        x_range_cm1 = np.linspace(1, 1800, id_pts)
        x_range_rad = x_range_cm1 * (2 * np.pi * 2.99792458e10)
        gamma_rad = 1.0e11 # Largeur de raie ajustable
        cm = 1j

        epsd_xx = np.zeros(id_pts)
        epsd_zz = np.zeros(id_pts)

        for ii in [0, 2]: # 0 pour X (transverse), 2 pour Z (longitudinal)
            q_vec = np.zeros(nt)
            for i in range(n_atoms):
                q_vec[i * 3 + ii] = qc[i] / np.sqrt(amassc)
            
            for j in range(nt):
                tt = np.dot(q_vec, eigenvectors[:, j].real)
                force_j = tt * tt
                
                denom = (x_range_rad**2 - fre_rad[j]**2) - cm * gamma_rad
                sc = x_range_rad * force_j / (denom * np.pi)
                
                if ii == 0:
                    epsd_xx += np.imag(sc)
                else:
                    epsd_zz += np.imag(sc)

        # --- 4. TRACÉ ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_range_cm1, epsd_xx, label='Polarisation XX (Transverse)', color='#00f2ff', lw=2)
        ax.plot(x_range_cm1, epsd_zz, label='Polarisation ZZ (Longitudinale)', color='#e11d48', linestyle='--', lw=2)

        ax.axvspan(800, 900, color='#10b981', alpha=0.15, label='Zone attendue (Flexion)')
        ax.set_title(f"Spectre IR Polarisé - Modèle de Saito - CNT ({self.n},{self.m})", color='#10b981', fontsize=14)
        ax.set_xlabel("Fréquence (cm⁻¹)")
        ax.set_ylabel("Intensité (Im(sc))")
        ax.legend(facecolor='#161b22')
        ax.grid(True, alpha=0.2)
        
        return fig

    def plot_raman_non_resonant(self):
        """
        Spectre Raman Non-Résonant.
        REMPLACEZ CE CODE PAR CELUI DE VOTRE JUPYTER NOTEBOOK.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- CODE PROVISOIRE (À REMPLACER) ---
        x = np.linspace(100, 1800, 1000)
        y = np.exp(-((x - 1580)**2)/200) + 0.5 * np.exp(-((x - 200)**2)/100) # G-band & RBM fictives
        ax.plot(x, y, color='#f59e0b', lw=2)
        ax.set_title(f"Spectre Raman Non-Résonant - CNT ({self.n},{self.m})", color='#10b981')
        ax.set_xlabel("Fréquence (cm⁻¹)")
        ax.set_ylabel("Intensité Raman")
        ax.grid(True, alpha=0.2)
        # ---------------------------------------
        
        return fig

    def plot_raman_resonant(self):
        """
        Spectre Raman Résonant (Intégration JDOS etc.)
        REMPLACEZ CE CODE PAR CELUI DE VOTRE JUPYTER NOTEBOOK.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- CODE PROVISOIRE (À REMPLACER) ---
        x = np.linspace(100, 1800, 1000)
        y = 2.0 * np.exp(-((x - 1580)**2)/150) + 1.5 * np.exp(-((x - 280)**2)/80)
        ax.plot(x, y, color='#8b5cf6', lw=2)
        ax.set_title(f"Spectre Raman Résonant (Couplage e-ph) - CNT ({self.n},{self.m})", color='#10b981')
        ax.set_xlabel("Fréquence (cm⁻¹)")
        ax.set_ylabel("Intensité Résonante")
        ax.grid(True, alpha=0.2)
        # ---------------------------------------
        
        return fig

    def plot_final_analysis(self):
        """
        Analyse finale (Synthèse de votre code Jupyter)
        REMPLACEZ CE CODE PAR CELUI DE VOTRE JUPYTER NOTEBOOK.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- CODE PROVISOIRE (À REMPLACER) ---
        ax.text(0.5, 0.5, "Espace réservé pour la synthèse finale", 
                color='#00f2ff', fontsize=16, ha='center')
        ax.set_title("Analyse Avancée Optique", color='#10b981')
        ax.axis('off')
        # ---------------------------------------
        
        return fig
