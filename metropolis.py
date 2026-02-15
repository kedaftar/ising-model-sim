import numpy as np
import numba as nb
import lattice
from lattice import Lattice
from visualization import (
    plot_lattice,
    plot_spin_and_energy_vs_time,
    plot_standard_deviation,
    plot_magnetization_vs_beta,
    plot_energy_vs_beta,
    plot_phase_diagram,
    show_plots
)
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science' , 'notebook' , 'grid'])

N = Lattice.N

#beta = 1/(k*T) where k is boltzmann constant and T is temperature in kelvin
#calculates change in energy 
@nb.njit(nogil=True)
def change_in_energy (x,y, spin_array, spin_initial, spin_final): #calculates change in energy 
    energy_initial = 0.0 
    energy_final = 0.0 

    if x > 0: #boundary conditions 
        energy_initial += -spin_initial * spin_array[x-1,y]
        energy_final   += -spin_final   * spin_array[x-1,y]
    if x < N-1:
        energy_initial += -spin_initial * spin_array[x+1,y]
        energy_final   += -spin_final   * spin_array[x+1,y]
    if y > 0: 
        energy_initial += -spin_initial * spin_array[x,y-1]
        energy_final   += -spin_final   * spin_array[x,y-1]
    if y < N-1: 
        energy_initial += -spin_initial * spin_array[x,y+1]
        energy_final   += -spin_final   * spin_array[x,y+1]

    return energy_initial, energy_final


#takes in initial lattice (2d grid of spins), number of time steps and temperature (BJ). 
#returns final lattice (total spin of atoms)
@nb.njit(nogil=True)
def metropolis (spin_arr, times, BJ, energy):

    spin_array = spin_arr.copy() #empty array to not modify original
    net_spins = np.zeros (times) #empty array to store net spins 
    net_energy = np.zeros (times) #empty array to store net energy 

    for t in range (times): #beggining of algo 

        for _ in range (N*N):
            #picks random points on the array 
            x = np.random.randint (0,N)
            y = np.random.randint (0,N)

            #flip sign of spin at random point
            spin_initial = spin_array[x,y] # initial spin 
            spin_final = spin_initial * -1 # final proposed spin

            energy_initial, energy_final = change_in_energy(
                x, y, spin_array, spin_initial, spin_final
            )

            #decides whether to accept or reject the proposed spin flip 
            delta_E = energy_final - energy_initial
            if (delta_E > 0) * (np.random.random() < np.exp(-BJ * delta_E)): #probability of doing flip based on comparing (<) to a random value between 1 and 0
                spin_array[x,y] = spin_final #when the line above is true flip occurs 
                energy += delta_E #current energy of system 
            elif delta_E <= 0: #if energy decreases flip always occurs
                spin_array[x,y] = spin_final
                energy += delta_E

        net_spins[t] = spin_array.sum() #total net spins at time step 
        net_energy[t] = energy #total net energy at time step

    return spin_array,net_spins, net_energy



# --- statistical approach (blocking) ---
def block_stats(x, n_blocks=20):
    x = np.asarray(x)
    if n_blocks < 2:
        return x.mean(), x.std(ddof=1)

    block_size = len(x) // n_blocks
    if block_size < 2:
        # not enough samples to form blocks
        return x.mean(), x.std(ddof=1)

    x = x[:block_size * n_blocks]
    blocks = x.reshape(n_blocks, block_size)
    block_means = blocks.mean(axis=1)

    # mean of block means; std of block means as an uncertainty estimate
    return block_means.mean(), block_means.std(ddof=1)


#takes in initial lattice (2d grid of spins), list of BJs (beta values)
#returns mean magnetization and mean energy (with uncertainty) for each BJ
def get_spin_energy(lattice0, BJs, sweeps=5000, burn_in=1000, thin=10, n_blocks=20):

    ms = np.zeros(len(BJs))
    m_std = np.zeros(len(BJs))

    energy_means = np.zeros(len(BJs))
    energy_std = np.zeros(len(BJs))

    for i, bj in enumerate(BJs):

        lattice_run = lattice0.copy()

        #initial energy for this starting lattice
        energy0 = lattice.Lattice_energy.get_energy(lattice_run)

        # run metropolis (Numba compiled)
        _,spins, energies = metropolis(lattice_run.astype(np.int64), sweeps, bj, energy0)

        # discard burn in
        spins_eq = spins[burn_in:]
        energies_eq = energies[burn_in:]

        # thin samples (reduce autocorrelation a bit)
        spins_s = spins_eq[::thin]
        energies_s = energies_eq[::thin]

        # per-spin observables
        M = np.abs(spins_s) / lattice_run.size
        E = energies_s / lattice_run.size

        # simple stats (if you ever want to compare)
        # ms[i] = M.mean()
        # energy_means[i] = E.mean()
        # energy_std[i] = E.std(ddof=1)

        # blocking stats (better for correlated samples)
        ms[i], m_std[i] = block_stats(M, n_blocks=n_blocks)
        energy_means[i], energy_std[i] = block_stats(E, n_blocks=n_blocks)

    return ms, m_std, energy_means, energy_std

#tester 
if __name__ == "__main__":
    print("="*60)
    print("ISING MODEL SIMULATION - TESTING AND VISUALIZATION")
    print("="*60)
    
    # Generate fresh lattice for testing
    test_lattice = Lattice.generate_lattice_n()
    
    # Test 1: Single run with time evolution
    print("\n[Test 1] Running single Metropolis simulation...")
    energy0 = lattice.Lattice_energy.get_energy(test_lattice)
    final_lattice,spins, energies = metropolis(
        test_lattice.astype(np.int64),
        1000,
        0.5,
        energy0
    )
    fig_init, _ = plot_lattice(test_lattice, title="Initial Lattice")
    fig_final, _ = plot_lattice(final_lattice, title="Final Lattice")

    print(f"Initial energy: {energy0}")
    print(f"Final energy: {energies[-1]}")
    print(f"Final magnetization: {spins[-1]}")
    
    # Generate graphs for single run
    print("\n[Test 1] Generating visualization plots...")
    fig_lattice, _ = plot_lattice(test_lattice, title="Initial Lattice Configuration")

    fig_time, _ = plot_spin_and_energy_vs_time(
    np.arange(len(spins)),
    spins,
    energies,
    title="Spin and Energy Evolution (β=0.5)"
)

    fig_std_time, _ = plot_standard_deviation(
    spins,
    title="Magnetization Standard Deviation Over Time"
)

    show_plots(fig_lattice, fig_time, fig_std_time,
          save_path='./output', filename_prefix='single_run')

    
    # Test 2: Phase diagram across multiple temperatures
    print("\n" + "="*60)
    print("[Test 2] Running temperature sweep for phase diagram...")
    print("="*60)
    
    # Generate fresh lattice for phase diagram test
    test_lattice_phase = Lattice.generate_lattice_n()
    
    BJs = np.array([0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.0])
    
    ms, m_std, e_means, e_std = get_spin_energy(
        test_lattice_phase,
        BJs,
        sweeps=5000,
        burn_in=1000,
        thin=10,
        n_blocks=20
    )

   # Generate phase diagram plots
    fig_mag, _ = plot_magnetization_vs_beta(
    BJs, ms, m_std,
    title="Magnetization vs Inverse Temperature"
)

    fig_energy, _ = plot_energy_vs_beta(
    BJs, e_means, e_std,
    title="Energy vs Inverse Temperature"
)

    fig_phase, _ = plot_phase_diagram(
    BJs, ms, e_means, m_std, e_std
)

    fig_std_comparison, _ = plot_standard_deviation(
    np.array([m_std, e_std]),
    labels=['Magnetization', 'Energy'],
    title="Standard Deviation Comparison Across Temperatures"
)

    fig_lattice_initial, _ = plot_lattice(
    test_lattice_phase,
    title="Initial Lattice Configuration"
)

    fig_lattice_final, _ = plot_lattice(
    final_lattice,
    title="Final Lattice Configuration (After Metropolis)"
)

# Debug: ensure these are Figures (not tuples)
# print(type(fig_mag), type(fig_energy), type(fig_phase), type(fig_std_comparison))

# show_plots(
#     fig_mag, fig_energy, fig_phase, fig_std_comparison,
#     save_path='./output', filename_prefix='phase_diagram',
# )
# show_plots(
#     fig_lattice_initial, fig_lattice_final,
#     save_path='./output', filename_prefix='Lattice'
# )

# np.savez(
#     "./output/single_run_data.npz",
#     lattice0=Lattice.lattice_n,
#     lattice_final=final_lattice,
#     spins=spins,
#     energies=energies,
#     beta=0.5,
#     N=N
# )

