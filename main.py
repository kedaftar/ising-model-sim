"""
Ising Model Simulation - Full Integration Pipeline
Processes: lattice.py -> metropolis.py -> visualization.py -> analyze.py
"""

import numpy as np
import os
import sys
from pathlib import Path

# Import all modules
import lattice
from lattice import Lattice, Lattice_energy
import metropolis as metropolis 
from metropolis import metropolis as metropolis_algo, get_spin_energy, block_stats
from visualization import (
    plot_lattice,
    plot_spin_and_energy_vs_time,
    plot_standard_deviation,
    plot_magnetization_vs_beta,
    plot_energy_vs_beta,
    plot_phase_diagram,
    show_plots
)
import analyze

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
import json

with open("config.json", "r") as f:
    config = json.load(f)


class IssingModelPipeline:
    """
    Full integration pipeline for Ising Model simulation and analysis.
    Orchestrates: Lattice initialization -> Metropolis simulation -> Visualization -> Analysis
    """
    
    def __init__(self, output_dir='./output'):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.lattice_init = None
        self.lattice_final = None
        self.spins = None
        self.energies = None
        self.ms = None
        self.m_std = None
        self.e_means = None
        self.e_std = None
        self.BJs = None
        
        print("="*70)
        print("ISING MODEL SIMULATION - INTEGRATION PIPELINE")
        print("="*70)
    
    def step1_initialize_lattice(self, use_lattice='lattice_n'):
        """
        Step 1: Initialize lattice from lattice.py
        
        Parameters:
        -----------
        use_lattice : str
            Which lattice to use: 'lattice_n' or 'lattice_p'
        
        Returns:
        --------
        np.ndarray : Initial lattice configuration
        """
        print("\n" + "-"*70)
        print("STEP 1: INITIALIZE LATTICE")
        print("-"*70)
        
        if use_lattice == 'lattice_n':
            self.lattice_init = Lattice.lattice_n.copy()
            lattice_name = "Lattice (negative dominant)"
        else:
            self.lattice_init = Lattice.lattice_p.copy()
            lattice_name = "Lattice (positive dominant)"
        
        initial_energy = Lattice_energy.get_energy(self.lattice_init)
        
        print(f"✓ Lattice initialized: {lattice_name}")
        print(f"  Size: {self.lattice_init.shape}")
        print(f"  Initial energy: {initial_energy:.2f}")
        print(f"  Mean spin: {self.lattice_init.mean():.4f}")
        
        return self.lattice_init
    
    def step2_run_single_simulation(self, beta=0.5, sweeps=1000):
        """
        Step 2: Run single Metropolis simulation from metropolis.py
        
        Parameters:
        -----------
        beta : float
            Inverse temperature (1/(k_B*T))
        sweeps : int
            Number of MC sweeps
        
        Returns:
        --------
        tuple : (spins, energies)
        """
        print("\n" + "-"*70)
        print("STEP 2: RUN SINGLE METROPOLIS SIMULATION")
        print("-"*70)
        
        energy0 = Lattice_energy.get_energy(self.lattice_init)
        print(f"Running Metropolis algorithm...")
        print(f"  β = {beta} (1/(k_B*T))")
        print(f"  Sweeps: {sweeps}")
        
        self.spins, self.energies = metropolis_algo(
            self.lattice_init.astype(np.int64),
            sweeps,
            beta,
            energy0
        )
        
        self.lattice_final = self.lattice_init.copy()
        
        print(f"✓ Simulation completed")
        print(f"  Final energy: {self.energies[-1]:.2f}")
        print(f"  Final magnetization: {self.spins[-1]:.2f}")
        print(f"  Energy range: [{self.energies.min():.2f}, {self.energies.max():.2f}]")
        
        return self.spins, self.energies
    
    def step3_run_phase_diagram(self, BJs=None, sweeps=5000, burn_in=1000, thin=10):
        """
        Step 3: Run temperature sweep for phase diagram from metropolis.py
        
        Parameters:
        -----------
        BJs : np.ndarray
            Array of beta values to scan
        sweeps : int
            Number of MC sweeps per temperature
        burn_in : int
            Number of sweeps to discard as equilibration
        thin : int
            Thinning factor for sample collection
        
        Returns:
        --------
        tuple : (ms, m_std, e_means, e_std)
        """
        print("\n" + "-"*70)
        print("STEP 3: PHASE DIAGRAM - TEMPERATURE SWEEP")
        print("-"*70)
        
        if BJs is None:
            BJs = np.array([0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.0])
        
        self.BJs = BJs
        
        print(f"Running temperature sweep...")
        print(f"  β values: {BJs}")
        print(f"  Sweeps per temperature: {sweeps}")
        print(f"  Burn-in: {burn_in}, Thin: {thin}")
        
        self.ms, self.m_std, self.e_means, self.e_std = get_spin_energy(
            self.lattice_init,
            BJs,
            sweeps=sweeps,
            burn_in=burn_in,
            thin=thin,
            n_blocks=20
        )
        
        print(f"✓ Temperature sweep completed")
        print(f"  Magnetization range: [{self.ms.min():.4f}, {self.ms.max():.4f}]")
        print(f"  Energy range: [{self.e_means.min():.4f}, {self.e_means.max():.4f}]")
        
        return self.ms, self.m_std, self.e_means, self.e_std
    
    def step4_generate_visualizations(self, save_plots=True):
        """
        Step 4: Generate all visualizations from visualization.py
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots as PDFs
        
        Returns:
        --------
        dict : Dictionary of all generated figures
        """
        print("\n" + "-"*70)
        print("STEP 4: GENERATE VISUALIZATIONS")
        print("-"*70)
        
        figures = {}
        
        # Lattice visualizations
        if self.lattice_init is not None:
            print("  Generating lattice plots...")
            figures['lattice_init'] = plot_lattice(
                self.lattice_init,
                title="Initial Lattice Configuration"
            )[0]
        
        # Single run visualizations
        if self.spins is not None and self.energies is not None:
            print("  Generating time evolution plots...")
            figures['time_evolution'] = plot_spin_and_energy_vs_time(
                np.arange(len(self.spins)),
                self.spins,
                self.energies,
                title="Spin and Energy Evolution"
            )[0]
            
            figures['std_spins'] = plot_standard_deviation(
                self.spins,
                title="Magnetization Standard Deviation Over Time"
            )[0]
            
            figures['std_energies'] = plot_standard_deviation(
                self.energies,
                title="Energy Standard Deviation Over Time"
            )[0]
        
        # Phase diagram visualizations
        if self.ms is not None and self.BJs is not None:
            print("  Generating phase diagram plots...")
            figures['magnetization_vs_beta'] = plot_magnetization_vs_beta(
                self.BJs, self.ms, self.m_std,
                title="Magnetization vs Inverse Temperature"
            )[0]
            
            figures['energy_vs_beta'] = plot_energy_vs_beta(
                self.BJs, self.e_means, self.e_std,
                title="Energy vs Inverse Temperature"
            )[0]
            
            figures['phase_diagram'] = plot_phase_diagram(
                self.BJs, self.ms, self.e_means,
                self.m_std, self.e_std
            )[0]
            
            figures['std_comparison'] = plot_standard_deviation(
                np.array([self.m_std, self.e_std]),
                labels=['Magnetization', 'Energy'],
                title="Standard Deviation Comparison"
            )[0]
        
        print(f"✓ Generated {len(figures)} visualizations")
        
        # Save plots
        if save_plots and figures:
            print(f"  Saving plots to {self.output_dir}...")
            show_plots(
                *figures.values(),
                save_path=self.output_dir,
                filename_prefix='ising_model'
            )
        
        return figures
    
    def step5_analyze_results(self):
        """
        Step 5: Analyze results (placeholder for analyze.py integration)
        
        Returns:
        --------
        dict : Analysis results
        """
        print("\n" + "-"*70)
        print("STEP 5: ANALYSIS")
        print("-"*70)
        
        analysis_results = {}
        
        if self.ms is not None:
            # Find critical behavior
            max_magnetization_idx = np.argmax(self.ms)
            critical_beta = self.BJs[max_magnetization_idx]
            analysis_results['critical_beta_approx'] = critical_beta
            
            print(f"✓ Analysis completed")
            print(f"  Approximate critical β: {critical_beta:.4f}")
            print(f"  Max magnetization: {self.ms[max_magnetization_idx]:.4f}")
            print(f"  Corresponding energy: {self.e_means[max_magnetization_idx]:.4f}")
        
        return analysis_results
    
    def run_full_pipeline(self, use_lattice='lattice_n', beta_single=0.5, 
                         sweeps_single=1000, BJs=None, sweeps_phase=5000):
        """
        Run the complete pipeline from initialization through analysis.
        
        Parameters:
        -----------
        use_lattice : str
            Initial lattice configuration
        beta_single : float
            Beta for single simulation
        sweeps_single : int
            Sweeps for single simulation
        BJs : np.ndarray
            Beta values for phase diagram
        sweeps_phase : int
            Sweeps for phase diagram
        
        Returns:
        --------
        dict : Complete results
        """
        try:
            # Step 1: Initialize lattice
            self.step1_initialize_lattice(use_lattice)
            
            # Step 2: Single simulation
            self.step2_run_single_simulation(beta_single, sweeps_single)
            
            # Step 3: Phase diagram
            self.step3_run_phase_diagram(BJs, sweeps_phase)
            
            # Step 4: Visualizations
            self.step4_generate_visualizations(save_plots=True)
            
            # Step 5: Analysis
            analysis = self.step5_analyze_results()
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"Output saved to: {self.output_dir}")
            print("="*70 + "\n")
            
            return {
                'lattice': self.lattice_init,
                'spins': self.spins,
                'energies': self.energies,
                'BJs': self.BJs,
                'ms': self.ms,
                'm_std': self.m_std,
                'e_means': self.e_means,
                'e_std': self.e_std,
                'analysis': analysis
            }
        
        except Exception as e:
            print(f"\n✗ ERROR in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point for the integration pipeline."""
    
    # Create and run pipeline
    pipeline = IssingModelPipeline(output_dir='./output')
    
    results = pipeline.run_full_pipeline(
        use_lattice='lattice_n',
        beta_single=0.5,
        sweeps_single=1000,
        BJs=np.array([0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.0]),
        sweeps_phase=5000
    )
    
    return results


if __name__ == "__main__":
    results = main()



