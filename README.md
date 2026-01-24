# ising-model-sim
IMPORTANT NOTE this model uses convolve when setting up the Hamiltonian which means that: $E = - \sum_i s_i \sum_{j \in \mathrm{nn}(i)} s_j$  
This was done for the sake of code simplicity and numerical stability for the metropolis algorithim. This makes it so that there is no need to track individual bonds, local energy differences become trivial to compute and acceptable probabilities are sharply generated. 
