# potential energy curve as a function of qubit connectivity

import csv
from pathlib import Path

import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc

import ffsim

start, stop, step = 0.8, 2.1, 0.05
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "n2"
basis = "sto-6g"
atom = lambda R: [["N", (0.0, 0, 0)], ["N", (R, 0, 0)]]
n_f = 2

OUT_CSV = Path(f"output/{molecule}_{basis}.csv")

CONNECTIVITIES = {
    "E_heavyhex": {
        "pairs_aa": lambda norb: [(p, p + 1) for p in range(norb - 1)],
        "pairs_ab": lambda norb: [(p, p) for p in range(0, norb, 4)],
    },
    "E_hex": {
        "pairs_aa": lambda norb: [(p, p + 1) for p in range(norb - 1)],
        "pairs_ab": lambda norb: [(p, p) for p in range(0, norb, 2)],
    },
    "E_square": {
        "pairs_aa": lambda norb: [(p, p + 1) for p in range(norb - 1)],
        "pairs_ab": lambda norb: [(p, p) for p in range(norb)],
    },
    "E_alltoall": {
        "pairs_aa": None,
        "pairs_ab": None,
    },
}


def append_row_csv(path: Path, row_dict, header):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


def energy_from_connectivity(mol_data, ccsd, label):
    norb = mol_data.norb
    nelec = mol_data.nelec

    H = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
    phi0 = ffsim.hartree_fock_state(norb, nelec)

    spec = CONNECTIVITIES[label]
    if spec["pairs_aa"] is None:
        ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2, t1=ccsd.t1, n_reps=1
        )
    else:
        pairs_aa = spec["pairs_aa"](norb)
        pairs_ab = spec["pairs_ab"](norb)
        ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=1,
            interaction_pairs=(pairs_aa, pairs_ab),
        )

    psi = ffsim.apply_unitary(phi0, ucj, norb=norb, nelec=nelec)
    e = float(np.vdot(psi, H @ psi).real)
    return e


def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    header = ["R"] + list(CONNECTIVITIES.keys())
    print(",".join(header), flush=True)

    for R in bond_distance_range:
        # build molecule
        mol = pyscf.gto.Mole()
        mol.build(
            atom=atom(R),
            basis=basis,
            symmetry="Dooh",
            max_memory=256 * 1000,
            verbose=0,
        )

        active_space = range(n_f, mol.nao_nr())

        scf = pyscf.scf.RHF(mol).run()

        norb = len(active_space)
        nelec = int(sum(scf.mo_occ[active_space]))
        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        _ = cas.sort_mo(active_space, base=0)

        frozen_orbs = [i for i in range(mol.nao_nr()) if i not in active_space]
        ccsd = pyscf.cc.RCCSD(scf, frozen=frozen_orbs)
        ccsd.kernel()

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

        row = {"R": f"{R:.6f}"}
        for label in CONNECTIVITIES:
            e_val = energy_from_connectivity(mol_data, ccsd, label)
            row[label] = f"{e_val:.12f}"

        print(",".join(row[k] for k in header), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
