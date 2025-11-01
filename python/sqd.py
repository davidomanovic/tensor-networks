"""
Sample based quantum diagonalization as presented originally:
https://quantum.cloud.ibm.com/docs/en/guides/qiskit-addons-sqd-get-started
"""

import pyscf
import pyscf.cc
import pyscf.mcscf
import ffsim
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from typing import Sequence
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from functools import partial
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch

open_shell = False
spin_sq = 0

R = 1.1

# Build N2 molecule
mol = pyscf.gto.Mole()
mol.build(
    atom=[["N", (0, 0, 0)], ["N", (R, 0, 0)]],
    basis="cc-pvdz",
    symmetry="Dooh",
)

# Define active space
n_frozen = 2
active_space = range(n_frozen, mol.nao_nr())

# Get molecular integrals
scf = pyscf.scf.RHF(mol).run()
norb = len(active_space)
nelec = int(sum(scf.mo_occ[active_space]))
num_elec_a = (nelec + mol.spin) // 2
num_elec_b = (nelec - mol.spin) // 2
cas = pyscf.mcscf.CASCI(scf, norb, (num_elec_a, num_elec_b))
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

# DMRG reference
exact_energy = -109.26354471028792

## job
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    instance='<insert instance here>'
)
job = service.job('<insert job id>')
primitive_result = job.result()
pub_result = primitive_result[0]
bitstring_samples = pub_result.data.meas # noisy bitstrings


## Sample based quantum diagonalization

from functools import partial

from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

energy_tol = 1e-4
occupancies_tol = 1e-4
max_iterations = 15

# Eigenstate solver options
num_batches = 10
samples_per_batch = 1000
symmetrize_spin = True
carryover_threshold = 1e-4
max_cycle = 500

sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

# List to capture intermediate results
result_history = []

def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")
    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(
            f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
        )

result = diagonalize_fermionic_hamiltonian(hcore,eri,
    pub_result.data.meas,samples_per_batch=samples_per_batch,
    norb=norb,nelec = (num_elec_a, num_elec_b), num_batches=num_batches,
    energy_tol=energy_tol,occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,sci_solver=sci_solver,
    symmetrize_spin=symmetrize_spin,carryover_threshold=carryover_threshold,
    callback=callback,seed=12345,
)

# Data for energies plot
x1 = range(len(result_history))
min_e = [
    min(result, key=lambda res: res.energy).energy + nuclear_repulsion_energy
    for result in result_history
]
e_diff = [abs(e - exact_energy) for e in min_e]
yt1 = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]

# Chemical accuracy (+/- 1 milli-Hartree)
chem_accuracy = 0.001

# Data for avg spatial orbital occupancy
y2 = np.sum(result.orbital_occupancies, axis=0)
x2 = range(len(y2))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot energies
axs[0].plot(x1, e_diff, label="energy error", marker="o", c='r')
axs[0].set_xticks(x1)
axs[0].set_xticklabels(x1)
axs[0].set_yticks(yt1)
axs[0].set_yticklabels(yt1)
axs[0].set_yscale("log")
axs[0].set_ylim(1e-4)
axs[0].axhline(
    y=chem_accuracy,
    color="k",
    linestyle="--",
    label="chemical accuracy",
)

axs[0].set_xlabel("Iteration", fontdict={"fontsize": 12})
axs[0].set_ylabel(r"$|E-E_{SQD}|$ [Ha]", fontdict={"fontsize": 12})
axs[0].legend()

# Plot orbital occupancy
axs[1].bar(x2, y2, width=0.8)
axs[1].set_xticks(x2)
axs[1].set_xticklabels(x2)
axs[1].set_xlabel("Orbital Index", fontdict={"fontsize": 12})
axs[1].set_ylabel("Avg Occupancy", fontdict={"fontsize": 12})

plt.tight_layout()
plt.savefig("callback.png")
tikzplotlib.save("callback.tex", axis_width="0.5\\linewidth", textsize=6, extra_axis_parameters=[ "tick label style={font=\\scriptsize}", "label style={font=\\footnotesize}", "title style={font=\\footnotesize}", "legend style={font=\\scriptsize}", ], extra_groupstyle_parameters=[ "horizontal sep=1.75cm" ] )