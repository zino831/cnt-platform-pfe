import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def calculate_phonons(atoms, z_period, q_array):
    """Calcule les fréquences de vibration (phonons)."""
    coords = atoms.get_positions()
    num_atoms = len(coords)
    num_q = len(q_array)
    
    # Paramètres de raideur
    k_bond = 29.3  # eV/Å^2
    m_carbon = 12.011 * 1.0364e-4 # Conversion de masse
    
    all_freqs = np.zeros((num_q, 3 * num_atoms))
    
    dist_0 = distance_matrix(coords, coords)
    adj_0 = (dist_0 > 0.1) & (dist_0 < 1.8)
    
    coords_plus = coords.copy()
    coords_plus[:, 2] += z_period
    dist_p = distance_matrix(coords, coords_plus)
    adj_p = (dist_p > 0.1) & (dist_p < 1.8)

    for i, q in enumerate(q_array):
        D = np.zeros((3 * num_atoms, 3 * num_atoms), dtype=complex)
        phase = np.exp(1j * q * z_period)
        
        for a in range(num_atoms):
            # Interactions intra-cellule
            for b in np.where(adj_0[a])[0]:
                vec = coords[b] - coords[a]
                dist = np.linalg.norm(vec)
                K = k_bond * np.outer(vec, vec) / (dist**2)
                for d1 in range(3):
                    for d2 in range(3):
                        D[3*a+d1, 3*a+d2] += K[d1, d2]
                        D[3*a+d1, 3*b+d2] -= K[d1, d2]
            
            # Interactions inter-cellule
            for b in np.where(adj_p[a])[0]:
                vec = coords_plus[b] - coords[a]
                dist = np.linalg.norm(vec)
                K = k_bond * np.outer(vec, vec) / (dist**2)
                for d1 in range(3):
                    for d2 in range(3):
                        D[3*a+d1, 3*a+d2] += K[d1, d2]
                        D[3*a+d1, 3*b+d2] -= K[d1, d2] * phase
                        D[3*b+d1, 3*a+d2] -= K[d1, d2] * np.conj(phase)
        
        # Fréquences en cm^-1
        eigenvals = np.linalg.eigvalsh(D)
        freqs = np.sqrt(np.abs(eigenvals) / m_carbon) * 33.356
        all_freqs[i, :] = np.sort(freqs)
    
    return all_freqs    

def calculate_cv(all_freqs, T_array):
    """Calcule la capacité thermique Cv(T)."""
    import numpy as np # Ajouté par sécurité
    kb = 8.617e-5 # eV/K
    cv_values = []
    
    freqs_flat = all_freqs.flatten()
    
    # --- LA CORRECTION EST ICI ---
    # On supprime les fréquences nulles (ou < 1 cm^-1) qui faussent le calcul
    freqs_flat = freqs_flat[freqs_flat > 1.0]
    # -----------------------------
    
    freqs_ev = freqs_flat / 8065.5 # Conversion cm^-1 vers eV
    
    for T in T_array:
        if T < 0.1:
            cv_values.append(0)
            continue
        x = freqs_ev / (kb * T)
        
        # On demande à Python d'ignorer les alertes d'infini mathématique
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            term = (x**2 * np.exp(x)) / (np.exp(x) - 1)**2
            # Remplace les calculs infinis (NaN) par 0 (physiquement correct)
            term = np.nan_to_num(term) 
            
        cv_values.append(np.sum(term))
        
    return np.array(cv_values)



def plot_thermal_cv(T_array, cv_values):
    plt.figure(figsize=(6, 4))
    plt.plot(T_array, cv_values, color='firebrick', lw=2.5)
    plt.title("Capacité Thermique $C_v$")
    plt.xlabel("Température (K)")
    plt.ylabel("$C_v$ (unités arbitraires)")
    plt.grid(True, linestyle=':')
    plt.xlim(0, 1000)
    plt.ylim(bottom=0)
    #plt.show()

def generer_neighbor_list_pbc(pos_atoms, lattice_constant, acc=1.42):
    """Identifie les voisins en tenant compte de la périodicité."""
    n_atoms = len(pos_atoms)
    dist_theoriques = [acc, np.sqrt(3)*acc, 2*acc, np.sqrt(7)*acc]
    tolerance = 0.2
    
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
                    if abs(dist - d_target) < tolerance:
                        rank = r + 1
                        break
                if rank > 0:
                    atom_neighbors.append({'idx': j, 'rank': rank, 'R': R_ij, 'dist': dist})
        neighbor_list.append(atom_neighbors)
    return neighbor_list

def construire_matrice_dynamique_pbc(atoms_obj, q_z, mass=12.011):
    """
    Calcule la matrice dynamique 3N x 3N pour un tube infini (PBC).
    """
    pos = atoms_obj.get_positions()
    n_atoms = len(pos)
    lattice_constant = atoms_obj.get_cell()[2, 2]
    m_carbon = mass / (6.022e23 * 1e3) # g/mol -> kg/atome
    
    dyn_mat = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=complex)
    neighbor_list = generer_neighbor_list_pbc(pos, lattice_constant)
    
    # Paramètres de Saito (N/m)
    phi_params = {
        1: {'r': 365.0, 'ti': 245.0}, 2: {'r': 88.0,  'ti': -32.3},
        3: {'r': 30.0,  'ti': -52.5}, 4: {'r': -19.0, 'ti': 22.9}
    }
    
    for i in range(n_atoms):
        for neighbor in neighbor_list[i]:
            j, rank, R, dist = neighbor['idx'], neighbor['rank'], neighbor['R'], neighbor['dist']
            
            # Tenseur de force local
            u = R / dist
            fr, fti = phi_params[rank]['r'], phi_params[rank]['ti']
            K_local = (fr - fti) * np.outer(u, u) + fti * np.eye(3)
            
            # Facteur de phase de Bloch exp(i * q_z * R_z)
            phase = np.exp(1j * q_z * R[2])
            
            # Remplissage par blocs 3x3
            term = (K_local / m_carbon) * phase
            dyn_mat[3*i:3*i+3, 3*j:3*j+3] -= term
            
            # Sum Rule pour la stabilité (Invariance par translation)
            dyn_mat[3*i:3*i+3, 3*i:3*i+3] += (K_local / m_carbon)
            
    return dyn_mat

def determiner_mode_rbm(pos_atoms, eigenvectors, wavenumbers):
    """Identifie le mode RBM par projection radiale pure."""
    n_atoms = len(pos_atoms)
    v_rbm_ideal = []
    for i in range(n_atoms):
        xi, yi = pos_atoms[i, 0], pos_atoms[i, 1]
        r = np.sqrt(xi**2 + yi**2)
        v_rbm_ideal.extend([xi/r, yi/r, 0] if r > 1e-6 else [0, 0, 0])
        
    v_rbm_ideal = np.array(v_rbm_ideal)
    v_rbm_ideal /= np.linalg.norm(v_rbm_ideal)

    overlaps = [np.abs(np.vdot(v_rbm_ideal, eigenvectors[:, m])) for m in range(3 * n_atoms)]
    idx_rbm = np.argmax(overlaps)
    return idx_rbm, wavenumbers[idx_rbm], overlaps[idx_rbm]