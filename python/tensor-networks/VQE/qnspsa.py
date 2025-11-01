#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pyscf.gto, pyscf.scf, pyscf.cc
from pyscf import lib
import csv

import ffsim
from ffsim import qiskit as fqs
from ffsim.qiskit import jordan_wigner

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

# QNSPSA + fidelity sampler
from qiskit_algorithms.optimizers import QNSPSA
from qiskit.primitives import StatevectorSampler

from scipy.optimize import minimize
from math import ceil

def energy_of_circuit(sim, circ, H, nqubits):
    circ.save_expectation_value(H, range(nqubits), label="E")
    tc = transpile(
        circ, backend=sim, optimization_level=3,
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

class TerminationChecker:

    def __init__(self, N : int):
        self.N = N
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        self.values.append(value)

        if len(self.values) > self.N:
            last_values = self.values[-self.N:]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            if slope > 0:
                return True
        return False


def main() -> None:
    molecule = "N2"
    basis_set = "sto-6g"
    n_reps = 1
    bond_dim = 64
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
        matrix_product_state_truncation_threshold=1e-10,
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
    nelec = mol_data.nelec
    n_qubits = 2 * norb

    H_f = ffsim.fermion_operator(mol_data.hamiltonian)

    H = jordan_wigner(H_f, norb=norb)

    # Local interaction structure (LUCJ locality)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    interaction_pairs = (pairs_aa, pairs_ab)
    frozen_list = [i for i in range(mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.CCSD(scf, frozen=frozen_list).run()
    ucj0 = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ccsd.t2, t1=ccsd.t1, n_reps=n_reps, interaction_pairs=interaction_pairs)
    theta0 = ucj0.to_parameters(interaction_pairs=interaction_pairs)

    # Build loss function (energy evaluation path)
    energy_fn = build_parametrized_energy_fn(
        simulator=simulator,
        n_qubits=n_qubits,
        norb=norb,
        nelec=nelec,
        H=H,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
    )

    from qiskit.circuit import ParameterVector
    theta_params = ParameterVector("θ", len(theta0))
    proxy = QuantumCircuit(n_qubits, name="proxy_ansatz")

    L = ceil(len(theta_params) / max(1, n_qubits))
    idx = 0
    for _ in range(L):
        # Ry layer
        for q in range(n_qubits):
            if idx >= len(theta_params): break
            proxy.ry(theta_params[idx], q)
            idx += 1
        # Rz layer
        for q in range(n_qubits):
            if idx >= len(theta_params): break
            proxy.rz(theta_params[idx], q)
            idx += 1
        # Entanglement ring
        for q in range(n_qubits):
            proxy.cx(q, (q + 1) % n_qubits)

    sampler = StatevectorSampler()
    fidelity = QNSPSA.get_fidelity(proxy, sampler)

    print(f"Initial parameter dimension: {theta0.size}", flush=True)

    algorithm_globals.random_seed = 42

    cb_state = {"iter": -1, "best": None, "best_val": np.inf, "log": []}
    def spsa_callback(nfev, params, value, stepsize, accepted):
        cb_state["iter"] += 1
        it = cb_state["iter"]
        energy = float(value)
        print(f"iter={it:4d}  nfev={nfev:5d}  energy={energy:.12f}", flush=True)
        cb_state["log"].append((it, int(nfev), energy))
        if energy < cb_state["best_val"]:
            cb_state["best_val"] = energy
            cb_state["best"] = np.copy(params)

    alpha, gamma = 0.602, 0.101
    maxit = 500
    A = int(0.1 * maxit)

    lr_gen_factory, pr_gen_factory = SPSA.calibrate(
        loss=energy_fn,
        initial_point=theta0,
        c=0.05,                
        target_magnitude=0.05, 
        stability_constant=A,
    )

    def scale_lr(gen, scale=1.5):
        def _g():
            for v in gen():
                yield scale * v
        return _g

    # Build optimizer
    qnspsa = QNSPSA(
        fidelity=fidelity,
        maxiter=maxit,
        learning_rate=scale_lr(lr_gen_factory, 1.5), 
        perturbation=pr_gen_factory,                  
        regularization=1e-3,                          
        resamplings=1,                                
        callback=spsa_callback,
    )

    t0 = time.perf_counter()
    result = qnspsa.minimize(energy_fn, x0=theta0)
    t1 = time.perf_counter()

    out = Path("vqe_qnspsa.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "nfev", "energy"])
        w.writerows(cb_state["log"])
    print(f"Saved energy history to {out.resolve()}", flush=True)

    print(f"VQE optimize time: {t1 - t0:.2f}", flush=True)
    print(f"VQE iters: {result.nit}, func_evals: {result.nfev}", flush=True)

    E = float(result.fun)
    E_lucj.append(E)
    print(f"1.1,{E:.10f}", flush=True)

if __name__ == "__main__":
    main()
