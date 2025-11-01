import numpy as np
import pyscf.gto, pyscf.scf, pyscf.mcscf, pyscf.ao2mo, pyscf.cc
import ffsim
from ffsim import qiskit as fqs
from ffsim.qiskit import jordan_wigner
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import time

start, stop, step = 0.8, 2.1, 0.1
R_values = np.linspace(start, stop, num=round((stop - start) / step) + 1)

molecule   = "N2"
basis_set  = "6-31g"
n_reps     = 1
bond_dim   = 512
n_f = 2

import os

os.environ.setdefault("OMP_NUM_THREADS", "48")
os.environ.setdefault("MKL_NUM_THREADS", "48")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "48")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

simulator = AerSimulator(method="matrix_product_state", 
                         matrix_product_state_max_bond_dimension=bond_dim,
                         mps_omp_threads=os.cpu_count(), mps_parallel_threshold=1,
                         matrix_product_state_truncation_threshold=1e-10)

from pyscf import lib
lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))

E_lucj = []

def energy_of_circuit(sim, circ, H: SparsePauliOp, nqubits: int) -> float:
    c = circ.remove_final_measurements(inplace=False)
    c.save_expectation_value(H, range(nqubits), label="E")
    tc = transpile(c, backend=sim, optimization_level=3)
    return sim.run(tc).result().data(0)["E"].real

print("R,E", flush=True)
for R in R_values:
    mol = pyscf.gto.Mole()
    mol.build(atom=[("N",(0.0,0.0,0.0)),("N",(R,0.0,0.0))],basis=basis_set, symmetry=True, unit="Angstrom", verbose=0)
    mol.max_memory = 256000

    time0 = time.perf_counter()
    scf = pyscf.scf.RHF(mol).run()
    time1 = time.perf_counter()

    print(f"SCF time: {time1 - time0:.2f}", flush=True)
    
    active_space = range(n_f, mol.nao_nr())
    time0 = time.perf_counter()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    time1 = time.perf_counter()
    print(f"MolData time: {time1 - time0:.2f}", flush=True)
    norb     = mol_data.norb
    nelec    = mol_data.nelec
    n_qubits = 2 * norb

    # Second-quantized molecular Hamiltonian to JW qubit operator
    time0 = time.perf_counter()
    H_f = ffsim.fermion_operator(mol_data.hamiltonian)
    time1 = time.perf_counter()
    print(f"Fermion operator time: {time1 - time0:.2f}", flush=True)
    time0 = time.perf_counter()
    H   = jordan_wigner(H_f, norb=norb, parallel=True)
    time1 = time.perf_counter()
    print(f"Hamiltonian time: {time1 - time0:.2f}", flush=True)

    # LUCJ locality pattern
    pairs_aa = [(p, p+1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    interaction_pairs = (pairs_aa, pairs_ab)

    # CCSD with core frozen consistently with active_space
    time0 = time.perf_counter()
    frozen_list = [i for i in range(mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.CCSD(scf, frozen=frozen_list).run()
    time1 = time.perf_counter()
    print(f"CCSD time: {time1 - time0:.2f}", flush=True)

    # UCJ operator initialized from CCSD amplitudes, constrained to LUCJ interactions
    time0 = time.perf_counter()
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=interaction_pairs)
    time1 = time.perf_counter()
    print(f"UCJ time: {time1 - time0:.2f}", flush=True)
    # HF reference + UCJ circuit in JW

    time0 = time.perf_counter()
    qreg = QuantumRegister(n_qubits, "q")
    circ = QuantumCircuit(qreg)
    circ.append(fqs.PrepareHartreeFockJW(norb, nelec), qreg)
    circ.append(fqs.UCJOpSpinBalancedJW(ucj),          qreg)
    time1 = time.perf_counter()
    print(f"Circuit time: {time1 - time0:.2f}", flush=True)

    time0 = time.perf_counter()
    E = energy_of_circuit(simulator, circ, H, n_qubits)
    time1 = time.perf_counter()
    print(f"Energy time: {time1 - time0:.2f}", flush=True)
    E_lucj.append(E)
    print(f"{R:.2f},{E:.10f}", flush=True)

out = np.vstack([R_values, E_lucj]).T
np.savetxt(f"output/{molecule}_{basis_set}_bd={bond_dim}.csv",out, delimiter=",", header="R,E")