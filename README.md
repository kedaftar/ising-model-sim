# ising-model-sim
IMPORTANT NOTE this model uses convolve when setting up the Hamiltonian which means that: $E = - \sum_i s_i \sum_{j \in \mathrm{nn}(i)} s_j$  
This was done for the sake of code simplicity and numerical stability for the metropolis algorithim. This makes it so that there is no need to track individual bonds, local energy differences become trivial to compute and acceptable probabilities are sharply generated.
 
To use this model simply edit the config.json variables to meet your experimental requirments then run main.py which will automatically produce graphs and will give numeric values in the terminal winodw. If you closed graphs from the previous run and want to see them again go to analyze.py and press run. Please note that analyze.py without being edited will only show the most recent run.

Scientific information: 
    the lattice follow the following rules, 1. N * N 2D lattice with spin -1 and +1. This is done with the nearest neighbor summing (coupling) and no external field.

    