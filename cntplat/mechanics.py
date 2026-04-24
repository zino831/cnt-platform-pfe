import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def calculate_young_modulus(atoms, a_lattice):
    """Calcule le module de Young via un test de traction."""
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
    plt.figure(figsize=(6, 4))
    plt.plot(strains*100, energies, 'o')
    plt.plot(strains*100, np.polyval(coefs, strains), 'r-')
    plt.title(f"Module de Young : {Y_GPa:.0f} GPa")
    plt.xlabel("Déformation (%)")
    plt.ylabel("Énergie (eV)")
    #plt.show()