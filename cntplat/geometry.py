# cnt_platform/geometry.py

from ase.build import nanotube
from ase.io import write
import py3Dmol
import io

def generate_structure(n, m, length=1):
    """
    Génère les coordonnées du nanotube en utilisant ASE.
    Retourne l'objet 'atoms' et la période spatiale 'a_lattice' (Lz).
    """
    # 1. Création du tube
    atoms = nanotube(n, m, length=length)
    
    # 2. Extraction de la période spatiale (longueur de la boîte selon Z dans ASE)
    a_lattice = atoms.get_cell()[2, 2]
    
    return atoms, a_lattice

def plot_3d_structure(atoms):
    """
    Convertit un objet ASE en visualisation interactive py3Dmol.
    """
    # Convertir en chaîne XYZ
    xyz_file = io.StringIO()
    write(xyz_file, atoms, format='xyz')
    xyz_string = xyz_file.getvalue()

    # Paramétrage de la vue 3D
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, 'xyz')
    view.setStyle({'stick': {'color': 'grey', 'radius': 0.15}, 
                   'sphere': {'scale': 0.25, 'color': 'midnightblue'}})
    view.zoomTo()
    
    return view