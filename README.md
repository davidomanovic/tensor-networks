# QuantumChemistrySimulations

This repo serves as a container for everything done during my master's thesis specialization project.

In this repository you will find python scripts which contain:
- Using quantum computing to solve the electronic structure problem in quantum chemistry.
- Simulations of the Local Unitary Cluster Jastrow ansatz on a quantum circuit by utilizing the `ffsim` library and `qiskit`.
- Classical quantum circuit simulations with tensor network methods, via the `matrix_product_state` method from `qiskit-aer`.
- Sample based Quantum Diagonalization with samples from real IBM-Q jobs.
- Variational quantum eigensolver simulations on a matrix product state tensor network with various optimizers like **QN-SPSA**, **SPSA**,  **L-BFGS-B** and the **Linear Method for Optimizing Jastrow-Feenberg correlations**.
 
This repository contains scripts and usage of my development library `fqcsim` (see https://github.com/davidomanovic/fqcsim) for different experiments quantum computing applications. 

# Examples
### **Binding curve for Nitrogen in STO-6G for different ansätze:**
<img width="601" height="387" alt="image" src="https://github.com/user-attachments/assets/1445b622-2f80-4d44-9eb6-bcc56defb7d7" />

### **Convergence of different optimizer during VQE:**
<img width="712" height="570" alt="image" src="https://github.com/user-attachments/assets/8ec20973-124a-4ce0-ba9a-79eb961c61b9" />

### **Tensor Network Matrix Product State simulations:**
<img width="543" height="332" alt="image" src="https://github.com/user-attachments/assets/9edf8c70-d9a8-4f37-a09b-e09eb3e83c7d" />

### **Sample Based Quantum Diagonalization algorithm on actual Quantum Hardware:**
<img width="617" height="357" alt="image" src="https://github.com/user-attachments/assets/a92ba20e-b552-46b0-ac5a-8ca476d943a6" />


