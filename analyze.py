import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_lattice, plot_spin_and_energy_vs_time
from visualization import plot_standard_deviation
import glob
import os


def get_latest_run_file(output_dir="./output"):
    """Find the most recent run_data_*.npz file"""
    pattern = os.path.join(output_dir, "run_data_*.npz")
    files = glob.glob(pattern)
    
    if not files:
        # Fall back to old filename if no timestamped files found
        old_file = os.path.join(output_dir, "single_run_data.npz")
        if os.path.exists(old_file):
            print(f"Using legacy file: {old_file}")
            return old_file
        else:
            raise FileNotFoundError(f"No run data files found in {output_dir}")
    
    # Get the newest file by modification time
    latest_file = max(files, key=os.path.getmtime)
    mod_time = os.path.getmtime(latest_file)
    import time
    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
    
    print(f"Found {len(files)} run data file(s)")
    print(f"Loading latest run: {os.path.basename(latest_file)}")
    print(f"File timestamp: {readable_time}")
    
    return latest_file


# Load the most recent simulation output
print("="*70)
print("ANALYZE - Loading Simulation Data")
print("="*70)
data_file = get_latest_run_file()
data = np.load(data_file)

print(f"\nLoaded data from: {os.path.basename(data_file)}")
print(f"File path: {data_file}")
print(f"Available keys: {list(data.keys())}")

lattice_init = data["lattice_init"]
lattice_final = data["lattice_final"]
spins = data["spins"]
energies = data["energies"]

# Get the correct beta value - prioritize beta_single (the one actually used for the simulation)
if "beta_single" in data:
    beta = float(data["beta_single"])
    print(f"Beta used for single simulation: {beta}")
elif "BJs" in data:
    BJs = data["BJs"]
    beta = BJs[0] if len(BJs) > 0 else "unknown"
    print(f"Beta values (phase diagram): {BJs}")
    print(f"Using first beta value: {beta}")
else:
    beta = "unknown"
    print("No beta information found")

print(f"\nData summary:")
print(f"  Initial lattice shape: {lattice_init.shape}")
print(f"  Number of sweeps: {len(spins)}")
print(f"  Final energy: {energies[-1]:.2f}")
print(f"  Final magnetization: {spins[-1]:.2f}")
print(f"  Beta (β): {beta}")

# plot initial and final lattice
fig_init, _ = plot_lattice(lattice_init, title="Initial Lattice")
fig_final, _ = plot_lattice(lattice_final, title="Final Lattice")

# plot time series
fig_time, _ = plot_spin_and_energy_vs_time(
    np.arange(len(spins)),
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
