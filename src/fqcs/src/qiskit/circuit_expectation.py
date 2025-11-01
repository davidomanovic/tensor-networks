# src/yourlib/quantum/sim.py
from __future__ import annotations

from typing import Iterable, Mapping, Any, Optional

from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import BackendV2  # qiskit >=1
# If you're on qiskit-terra <1.0, you can use Backend instead:
# from qiskit.providers import Backend

def circuit_expectation(
    sim: BackendV2,                   
    circ: QuantumCircuit,
    H: SparsePauliOp,
    nqubits: Optional[int] = None,
    *,
    label: str = "E",
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = 0,
    layout_method: Optional[str] = "trivial",
    transpile_options: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Compute <psi|H|psi> for a circuit |psi> using an exact (analytic) expectation.

    Parameters
    ----------
    sim
        A simulator backend that supports analytic expectations (e.g., AerSimulator
        with statevector or matrix_product_state).
    circ
        The state-preparation circuit |psi>.
    H
        Hamiltonian as a `SparsePauliOp`. Qiskit will evaluate this exactly (no shots).
    nqubits
        Number of qubits the expectation is taken over. Defaults to `circ.num_qubits`.
    label
        Result key under which the expectation value is saved.
    optimization_level
        Transpiler optimization level (default 1 for speed + light optimization).
    seed_transpiler
        Seed for deterministic transpilation. Use `None` to disable seeding.
    layout_method
        Layout strategy for the transpiler (default "trivial" to keep qubit order stable).
    transpile_options
        Extra kwargs forwarded to `qiskit.transpile`.

    Returns
    -------
    float
        The real part of the expectation value <psi|H|psi>.

    Notes
    -----
    - Assumes an analytic backend; do not pass `shots`, Qiskit will compute
      expectations exactly.
    - Uses `QuantumCircuit.save_expectation_value` on all qubits in range(nqubits).
    """
    if nqubits is None:
        nqubits = circ.num_qubits

    circ_with_ev = circ.copy()
    # Save the expectation value of H across the first nqubits
    circ_with_ev.save_expectation_value(H, list(range(nqubits)), label=label)

    t_opts = dict(
        backend=sim,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        layout_method=layout_method,
    )
    if transpile_options:
        t_opts.update(transpile_options)

    tc = transpile(circ_with_ev, **t_opts)

    # No need to pass shots for analytic expectations.
    res = sim.run(tc).result()
    value = res.data(0)[label]
    # Make sure we return a plain float
    return float(getattr(value, "real", value))
