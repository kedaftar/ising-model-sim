import numpy as np
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science' , 'notebook' , 'grid'])
import numba as nb
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

#must fix graphing issues later

#50 by 50 lattice
N = 50
#shows initial lattice of spins 
init_random = np.random.random((N,N))
lattice_n = np.zeros((N,N))
lattice_n[init_random >= 0.75] = 1
lattice_n[init_random < 0.75] = -1
init_random = np.random.random((N,N))
lattice_p = np.zeros((N,N))
lattice_p[init_random >= 0.25] = 1
lattice_p[init_random < 0.25] = -1

plt.imshow(lattice_p)

