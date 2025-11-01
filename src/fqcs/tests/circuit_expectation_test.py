# tests/circuit_expectation_test.py
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from fqcs.qiskit import circuit_expectation

def test_energy_z_on_zero():
    sim = AerSimulator(method="statevector")
    qc = QuantumCircuit(1)
    # |0> is already prepared
    H = SparsePauliOp.from_list([("Z", 1.0)])  # <0|Z|0> = +1
    E = circuit_expectation(sim, qc, H)
    assert math.isclose(E, 1.0, rel_tol=0, abs_tol=1e-9)

def test_energy_x_on_zero():
    sim = AerSimulator(method="statevector")
    qc = QuantumCircuit(1)
    H = SparsePauliOp.from_list([("X", 1.0)])  # <0|X|0> = 0
    E = circuit_expectation(sim, qc, H)
    assert math.isclose(E, 0.0, rel_tol=0, abs_tol=1e-9)

def test_energy_two_qubits_zi_plus_ix():
    sim = AerSimulator(method="statevector")
    qc = QuantumCircuit(2)
    # state |00>, <Z I> = +1, <I X> = 0
    H = SparsePauliOp.from_list([("ZI", 0.5), ("IX", 2.0)])
    E = circuit_expectation(sim, qc, H, nqubits=2)
    assert math.isclose(E, 0.5, rel_tol=0, abs_tol=1e-9)
