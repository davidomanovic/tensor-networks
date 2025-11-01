#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pyscf.gto, pyscf.scf, pyscf.cc
from pyscf import lib

import ffsim
from ffsim import qiskit as fqs
from ffsim.qiskit import jordan_wigner

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from scipy.optimize import minimize

def energy_of_circuit(sim, circ, H, nqubits):
    circ.save_expectation_value(H, range(nqubits), label="E")
    tc = transpile(
        circ, backend=sim, optimization_level=1,
        seed_transpiler=0, layout_method="trivial"
    )
    # No need to pass shots for analytic expectations
    E = sim.run(tc).result().data(0)["E"].real
    return float(E)


def build_parametrized_energy_fn(
    simulator: AerSimulator,
    n_qubits: int,
    norb: int,
    nelec: tuple[int, int],
    H: SparsePauliOp,
    n_reps: int,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None],
):
    """Return f(theta) that rebuilds a UCJ gate from parameters and evaluates energy."""
    qreg = QuantumRegister(n_qubits, "q")

    def energy_from_params(theta: np.ndarray) -> float:
        ucj_theta = ffsim.UCJOpSpinBalanced.from_parameters(
            theta,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True
        )
        circ = QuantumCircuit(qreg)
        circ.append(fqs.PrepareHartreeFockJW(norb, nelec), qreg)
        circ.append(fqs.UCJOpSpinBalancedJW(ucj_theta), qreg)

        E = energy_of_circuit(simulator, circ, H, n_qubits)
        return E

    return energy_from_params


def main() -> None:
    molecule = "N2"
    basis_set = "cc-pvdz"
    n_reps = 1
    bond_dim = 512
    n_f = 2

    if "OMP_NUM_THREADS" in os.environ:
        lib.num_threads(int(os.environ["OMP_NUM_THREADS"]))

    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        mps_omp_threads=48,
        mps_parallel_threshold=48,
        max_parallel_threads=48,
        seed_simulator=0,
        matrix_product_state_truncation_threshold=1e-6,
    )

    E_lucj: list[float] = []

    print(f"{molecule} in {basis_set}")
    print("R,E", flush=True)
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (0.0, 0.0, 0.0)), ("N", (1.1, 0.0, 0.0))],
        basis=basis_set,
        symmetry="Dooh",
        unit="Angstrom",
        verbose=0,
    )

    scf = pyscf.scf.RHF(mol).run()

    active_space = range(n_f, mol.nao_nr())

    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

    norb = mol_data.norb
    nelec = mol_data.nelec  # (n_alpha, n_beta)
    n_qubits = 2 * norb

    H_f = ffsim.fermion_operator(mol_data.hamiltonian)

    H = jordan_wigner(H_f, norb=norb)

    # Local interaction structure (LUCJ locality)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 4)]
    interaction_pairs = (pairs_aa, pairs_ab)

    # Start from CCSD amplitudes (good initialization for UCJ)
    frozen_list = [i for i in range(mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.CCSD(scf, frozen=frozen_list).run()

    ucj0 = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=interaction_pairs)

    # Flatten to variational parameters
    theta0 = ucj0.to_parameters(interaction_pairs=interaction_pairs)

    # Loss function E(theta)
    energy_fn = build_parametrized_energy_fn(
        simulator=simulator,
        n_qubits=n_qubits,
        norb=norb,
        nelec=nelec,
        H=H,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
    )

    # Simple VQE using L-BFGS-B (gradient-free fallback via finite differences)
    print(f"Initial parameter dimension: {theta0.size}", flush=True)

    cb_last = {"iter": -1, "best": None}

    def callback(xk):
        cb_last["iter"] += 1

    t0 = time.perf_counter()
    opt = minimize(
        energy_fn,
        theta0,
        method="L-BFGS-B",
        jac=None,
        options=dict(maxiter=1000),
        callback=callback,
    )
    t1 = time.perf_counter()
    print(f"VQE optimize time: {t1 - t0:.2f}", flush=True)
    print(f"VQE success: {opt.success}, iters: {opt.nit}, func_evals: {opt.nfev}", flush=True)
    E = float(opt.fun)
    E_lucj.append(E)
    print(f"1.1,{E:.10f}", flush=True)

if __name__ == "__main__":
    main()
