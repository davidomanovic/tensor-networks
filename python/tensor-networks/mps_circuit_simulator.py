#!/usr/bin/env python3
import os
import time
import numpy as np
import pyscf.gto, pyscf.scf, pyscf.cc
from pyscf import lib
import ffsim
from ffsim import qiskit as fqs
from ffsim.qiskit import jordan_wigner
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from pathlib import Path
from collections.abc import Mapping, Sequence
import functools
from ffsim.operators import FermionOperator

def energy_of_circuit(sim: AerSimulator, circ: QuantumCircuit, H: SparsePauliOp, nqubits: int, label) -> float:
    circ.save_expectation_value(H, range(nqubits), label=f"{label}")
    tc = transpile(circ, backend=sim, optimization_level=3)
    simulation = sim.run(tc, shots=1) 
    E = simulation.result().data(0)[f"{label}"].real
    return E

def main() -> None:

    start, stop, step = 0.8, 2.1, 0.05
    bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
    molecule = "N2"
    basis_set = "cc-pvdz"
    n_reps = 1
    n_f = 2  # frozen core
    max_bond_dim = 512
    truncation = 1e-10

    # AerSimulator (MPS) backend
    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=max_bond_dim,
        matrix_product_state_truncation_threshold=truncation,
        mps_omp_threads=48,
        mps_parallel_threshold=48,
        max_parallel_threads=48,
    )

    E_lucj: list[float] = []
    print("R,E", flush=True)
    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(
            atom = [["N",(-0.5*R,0,0)],["N",(0.5*R,0,0)]],
            basis=basis_set,
            symmetry="Dooh",
            unit="Angstrom",
            verbose=0,
        )
        scf = pyscf.scf.RHF(mol).run()
        active_space = range(n_f, mol.nao_nr())
        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

        norb = mol_data.norb
        nelec = mol_data.nelec
        n_qubits = 2 * norb

        H_f = ffsim.fermion_operator(mol_data.hamiltonian)
        H_q = jordan_wigner(H_f, norb=norb) 

        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(0,norb,4)]
        interaction_pairs = (pairs_aa, pairs_ab)

        frozen_list = [i for i in range(mol.nao_nr()) if i not in active_space]
        ccsd = pyscf.cc.CCSD(scf, frozen=frozen_list).run()
        ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=interaction_pairs
        )

        qreg = QuantumRegister(n_qubits, "q")
        circ = QuantumCircuit(qreg)
        circ.append(fqs.PrepareHartreeFockJW(norb, nelec), qreg)
        circ.append(fqs.UCJOpSpinBalancedJW(ucj), qreg)
        E = energy_of_circuit(simulator, circ, H_q, n_qubits, "E")
        
        print(f"{R:.10f},{E:.10f}", flush=True)

if __name__ == "__main__":
    main()

