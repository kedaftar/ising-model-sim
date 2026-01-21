import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_lattice, plot_spin_and_energy_vs_time
from visualization import plot_standard_deviation


# load saved simulation output
data = np.load("./output/single_run_data.npz")

lattice0 = data["lattice0"]
lattice_final = data["lattice_final"]
spins = data["spins"]
energies = data["energies"]
beta = data["beta"]

# plot initial and final lattice
fig_init, _ = plot_lattice(lattice0, title="Initial Lattice")
fig_final, _ = plot_lattice(lattice_final, title="Final Lattice")

# plot time series
fig_time, _ = plot_spin_and_energy_vs_time(
    len(spins),
    spins,
    energies,
    title=f"Spin and Energy Evolution (β={beta})"
)
fig_std_spin, _ = plot_standard_deviation(
    spins,
    title="Magnetization Standard Deviation"
)

fig_std_energy, _ = plot_standard_deviation(
    energies,
    title="Energy Standard Deviation"
)

plt.show()
