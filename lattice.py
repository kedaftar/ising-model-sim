import numpy as np
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science' , 'notebook' , 'grid'])
import numba as nb
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

class Lattice: 
    N = 50  # Default lattice size
#shows initial lattice of spins 
    init_random = np.random.random((N,N))
    lattice_n = np.zeros((N,N))
    lattice_n[init_random >= 0.75] = 1
    lattice_n[init_random < 0.75] = -1
    init_random = np.random.random((N,N))
    lattice_p = np.zeros((N,N))
    lattice_p[init_random >= 0.25] = 1
    lattice_p[init_random < 0.25] = -1

class Lattice_energy: 
    def get_energy(lattice): #applies the nearest neighbor summation 
        kern = generate_binary_structure(2,1) 
        kern[1][1] = False 
        energies_of_lattice = -lattice * convolve(lattice, kern, mode= 'constant', cval=0 )
        return 0.5 * energies_of_lattice.sum()

if __name__ == "__main__":
    print (Lattice_energy.get_energy(Lattice.lattice_n))