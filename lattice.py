import numpy as np
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science' , 'notebook' , 'grid'])
import numba as nb
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import json

# Load configuration fresh each time the module is reloaded
def _load_lattice_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = _load_lattice_config()

class Lattice: 
    N = config["lattice"]["N"]  # Load lattice size from config
    
    # Remove static lattices - generate fresh ones each time
    lattice_n = None  # Will be generated on demand
    lattice_p = None  # Will be generated on demand
    
    @staticmethod
    def reload_config():
        """Reload config and update N"""
        global config
        config = _load_lattice_config()
        Lattice.N = config["lattice"]["N"]
    
    @staticmethod
    def generate_lattice_n(seed=None):
        """Generate a new negative-dominant lattice (75% negative spins)"""
        if seed is not None:
            np.random.seed(seed)
        N = Lattice.N
        init_random = np.random.random((N, N))
        lattice = np.zeros((N, N))
        lattice[init_random >= 0.75] = 1
        lattice[init_random < 0.75] = -1
        return lattice
    
    @staticmethod
    def generate_lattice_p(seed=None):
        """Generate a new positive-dominant lattice (75% positive spins)"""
        if seed is not None:
            np.random.seed(seed)
        N = Lattice.N
        init_random = np.random.random((N, N))
        lattice = np.zeros((N, N))
        lattice[init_random >= 0.25] = 1
        lattice[init_random < 0.25] = -1
        return lattice

class Lattice_energy: 
    def get_energy(lattice): #applies the nearest neighbor summation 
        kern = generate_binary_structure(2,1) 
        kern[1][1] = False 
        energies_of_lattice = -lattice * convolve(lattice, kern, mode= 'constant', cval=0 )
        return 0.5 * energies_of_lattice.sum()

if __name__ == "__main__":
    print (Lattice_energy.get_energy(Lattice.lattice_n))