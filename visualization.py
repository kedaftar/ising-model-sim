import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import time

plt.style.use(['science', 'notebook', 'grid'])
# Ensure interactive mode is off to prevent caching
plt.ioff()


def plot_lattice(lattice, title="Lattice Configuration", cmap='coolwarm'):
    """
    Plot a 2D lattice of spins.
    
    Parameters:
    -----------
    lattice : np.ndarray
        2D array representing the spin lattice (-1 or 1 values)
    title : str
        Title for the plot
    cmap : str
        Colormap to use for visualization
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create a completely new figure (no caching)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Make a copy to ensure no reference issues
    lattice_data = np.array(lattice, copy=True)
    
    im = ax.imshow(lattice_data, cmap=cmap, interpolation='nearest')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spin')
    
    fig.tight_layout()
    return fig, ax


def plot_spin_and_energy_vs_time(times, spins, energies, title="Spin and Energy vs Time"):
    """
    Plot magnetization (net spin) and energy as a function of time steps.
    
    Parameters:
    -----------
    times : np.ndarray or int
        Time steps (can be range or array of time values)
    spins : np.ndarray
        Array of net spin values at each time step
    energies : np.ndarray
        Array of energy values at each time step
    title : str
        Title for the plot
    
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axes objects
    """
    if isinstance(times, int):
        times = np.arange(times)
    
    # Create completely new figure
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    
    # Make copies of data
    times_data = np.array(times, copy=True)
    spins_data = np.array(spins, copy=True)
    energies_data = np.array(energies, copy=True)
    
    # Plot net spins
    ax1.plot(times_data, spins_data, linewidth=1.5, color='steelblue')
    ax1.set_ylabel('Net Spin (Magnetization)')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Plot energies
    ax2.plot(times_data, energies_data, linewidth=1.5, color='darkred')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Total Energy')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_standard_deviation(data_array, labels=None, title="Standard Deviation Analysis"):
    """
    Plot standard deviation of multiple datasets or observables.
    
    Parameters:
    -----------
    data_array : np.ndarray or list of np.ndarray
        Array(s) of data to calculate standard deviation for.
        Can be a 2D array where each row is a different observable,
        or a list of 1D arrays.
    labels : list of str
        Labels for each dataset/observable
    title : str
        Title for the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create new figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Handle different input formats and copy data
    if isinstance(data_array, list):
        data_array = np.asarray(data_array, dtype=object)
    else:
        data_array = np.array(data_array, copy=True)
    
    # Calculate standard deviations
    if data_array.ndim == 1:
        # Single array: calculate running std
        window_size = max(1, len(data_array) // 50)
        running_std = np.array([
            np.std(data_array[max(0, i-window_size):i+window_size])
            for i in range(len(data_array))
        ])
        ax.plot(running_std, linewidth=2, color='purple')
        ax.set_ylabel('Running Standard Deviation')
        ax.set_xlabel('Time Step')
    else:
        # Multiple arrays
        if labels is None:
            labels = [f'Observable {i+1}' for i in range(len(data_array))]
        
        stds = np.std(data_array, axis=1)
        x_pos = np.arange(len(stds))
        
        ax.bar(x_pos, stds, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Standard Deviation')
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig, ax


def plot_magnetization_vs_beta(betas, magnetizations, m_std=None, title="Magnetization vs Temperature"):
    """
    Plot magnetization as a function of beta (inverse temperature).
    
    Parameters:
    -----------
    betas : np.ndarray
        Array of beta (inverse temperature) values
    magnetizations : np.ndarray
        Array of magnetization values corresponding to each beta
    m_std : np.ndarray, optional
        Standard deviations for error bars
    title : str
        Title for the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create new figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Copy data
    betas_data = np.array(betas, copy=True)
    mags_data = np.array(magnetizations, copy=True)
    
    if m_std is not None:
        m_std_data = np.array(m_std, copy=True)
        ax.errorbar(betas_data, mags_data, yerr=m_std_data, fmt='o-', 
                   linewidth=2, markersize=8, capsize=5, color='steelblue',
                   ecolor='darkblue', alpha=0.8)
    else:
        ax.plot(betas_data, mags_data, 'o-', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_xlabel('β = 1/(k_B T)')
    ax.set_ylabel('Magnetization (|M|)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax


def plot_energy_vs_beta(betas, energies, e_std=None, title="Energy vs Temperature"):
    """
    Plot energy as a function of beta (inverse temperature).
    
    Parameters:
    -----------
    betas : np.ndarray
        Array of beta (inverse temperature) values
    energies : np.ndarray
        Array of energy values corresponding to each beta
    e_std : np.ndarray, optional
        Standard deviations for error bars
    title : str
        Title for the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create new figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Copy data
    betas_data = np.array(betas, copy=True)
    energies_data = np.array(energies, copy=True)
    
    if e_std is not None:
        e_std_data = np.array(e_std, copy=True)
        ax.errorbar(betas_data, energies_data, yerr=e_std_data, fmt='o-', 
                   linewidth=2, markersize=8, capsize=5, color='darkred',
                   ecolor='red', alpha=0.8)
    else:
        ax.plot(betas_data, energies_data, 'o-', linewidth=2, markersize=8, color='darkred')
    
    ax.set_xlabel('β = 1/(k_B T)')
    ax.set_ylabel('Energy per spin (E/N)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax


def plot_phase_diagram(betas, magnetizations, energies, m_std=None, e_std=None):
    """
    Plot both magnetization and energy on the same figure with dual y-axes.
    
    Parameters:
    -----------
    betas : np.ndarray
        Array of beta (inverse temperature) values
    magnetizations : np.ndarray
        Array of magnetization values
    energies : np.ndarray
        Array of energy values
    m_std : np.ndarray, optional
        Standard deviations for magnetization
    e_std : np.ndarray, optional
        Standard deviations for energy
    
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axes objects
    """
    # Create new figure
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    
    # Copy data
    betas_data = np.array(betas, copy=True)
    mags_data = np.array(magnetizations, copy=True)
    energies_data = np.array(energies, copy=True)
    
    # Magnetization on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('β = 1/(k_B T)')
    ax1.set_ylabel('Magnetization (|M|)', color=color1)
    
    if m_std is not None:
        m_std_data = np.array(m_std, copy=True)
        line1 = ax1.errorbar(betas_data, mags_data, yerr=m_std_data, fmt='o-',
                            linewidth=2, markersize=8, capsize=5, color=color1, alpha=0.8)
    else:
        line1 = ax1.plot(betas_data, mags_data, 'o-', linewidth=2, markersize=8, color=color1, alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Energy on right y-axis
    ax2 = ax1.twinx()
    color2 = 'darkred'
    ax2.set_ylabel('Energy per spin (E/N)', color=color2)
    
    if e_std is not None:
        e_std_data = np.array(e_std, copy=True)
        line2 = ax2.errorbar(betas_data, energies_data, yerr=e_std_data, fmt='s-',
                            linewidth=2, markersize=8, capsize=5, color=color2, alpha=0.8)
    else:
        line2 = ax2.plot(betas_data, energies_data, 's-', linewidth=2, markersize=8, color=color2, alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.suptitle('Phase Diagram: Magnetization and Energy vs Temperature')
    fig.tight_layout()
    
    return fig, (ax1, ax2)


def show_plots(*figures, save_path=None, filename_prefix='plot'):
    """
    Save matplotlib figures as PDFs with timestamps. Figures are saved, not displayed.
    
    Parameters:
    -----------
    *figures : matplotlib figure objects
        Variable number of figure objects to save
    save_path : str, optional
        Directory path to save figures. If None, saves to current directory.
    filename_prefix : str
        Prefix for saved files (default: 'plot'). Files will be named with timestamps
        to prevent overwriting: 'plot_20260214_143022_1.pdf', etc.
    
    Returns:
    --------
    saved_files : list of str
        List of full paths to saved PDF files
    
    Examples:
    ---------
    # Save single figure
    fig, ax = plot_lattice(lattice)
    show_plots(fig)
    
    # Save multiple figures to custom directory
    fig1, ax1 = plot_lattice(lattice)
    fig2, (ax2a, ax2b) = plot_spin_and_energy_vs_time(times, spins, energies)
    show_plots(fig1, fig2, save_path='./output', filename_prefix='ising')
    
    # Files will be saved as: ./output/ising_20260214_143022_1.pdf, etc.
    """
    import os
    
    if len(figures) == 0:
        print("No figures provided to save.")
        return []
    
    # Create save directory if needed
    if save_path is None:
        save_path = os.getcwd()
    else:
        os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    saved_files = []
    
    # Save all figures as PDFs with unique timestamped names
    for i, fig in enumerate(figures):
        pdf_filename = os.path.join(save_path, f'{filename_prefix}_{timestamp}_{i+1}.pdf')
        # Force rendering and save
        fig.canvas.draw()
        fig.savefig(pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
        saved_files.append(pdf_filename)
        print(f"Saved: {pdf_filename}")
        # Explicitly close the figure to free memory and prevent caching
        plt.close(fig)
    
    return saved_files
