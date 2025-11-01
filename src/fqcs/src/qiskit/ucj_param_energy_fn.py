# src/yourlib/quantum/ucj.py
from __future__ import annotations

from typing import Callable

import numpy as np
import ffsim
from ffsim import qiskit as fqs

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

from .circuit_expectation import circuit_expectation

def build_parametrized_energy_fn(
    simulator: AerSimulator,
    n_qubits: int,
    norb: int,
    nelec: tuple[int, int],
    H: SparsePauliOp,
    n_reps: int,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None],
) -> Callable[[np.ndarray], float]:
    """
    Return f(theta) that rebuilds a UCJ gate from parameters and evaluates energy.

    Parameters
    ----------
    simulator
        Qiskit AerSimulator backend (e.g., AerSimulator(method="statevector")).
    n_qubits
        Total number of qubits in the circuit/register.
    norb
        Number of spatial orbitals.
    nelec
        Tuple of (alpha_electrons, beta_electrons).
    H
        Hamiltonian as a SparsePauliOp.
    n_reps
        Number of UCJ repetitions/layers.
    interaction_pairs
        Tuple (alpha_pairs, beta_pairs); each element is a list of (p, q) pairs or None.

    Returns
    -------
    Callable[[np.ndarray], float]
        A function mapping UCJ parameter vector `theta` to the energy <ψ(θ)|H|ψ(θ)>.
    """
    qreg = QuantumRegister(n_qubits, "q")

    def energy_from_params(theta: np.ndarray) -> float:
        ucj_theta = ffsim.UCJOpSpinBalanced.from_parameters(
            theta,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
        circ = QuantumCircuit(qreg)
        circ.append(fqs.PrepareHartreeFockJW(norb, nelec), qreg)
        circ.append(fqs.UCJOpSpinBalancedJW(ucj_theta), qreg)

        return circuit_expectation(simulator, circ, H, n_qubits)

    return energy_from_params
