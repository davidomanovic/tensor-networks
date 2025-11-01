"""
Diatomics bond stretching simulations.
Comparisons between different wavefunction ansätze.
- Returns: .csv file of potential energy curve/surface for given molecule 
"""

import csv
import os
from pathlib import Path
import scipy.optimize
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc
import ffsim
from ffsim.optimize import minimize_linear_method

# Start definitions
start, stop, step = 0.8, 2.1, 0.05
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "n2"
basis = "6-31g"
atom = lambda R: [["N",(0.0,0,0)],["N",(R,0,0)]]
n_f = 2 # frozen core: n_f = 2
OUT_CSV = Path(f"output/{molecule}_{basis}.csv")

def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)

def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    header = ["R", "E_FCI", "E_HF",
              "E_CCSD","E_uCJ","E_LuCJ",
              "E_LuCJ1(var-opt)","E_LuCJ2(var-opt)","E_LuCJ3(var-opt)"
              ]
    print(",".join(header), flush=True)

    x_prev1 = None   # warm start for 1-rep
    x_prev2 = None   # warm start for 2-rep
    x_prev3 = None   # warm start for 3-rep

    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(atom=atom(R), basis=basis, symmetry="Dooh", verbose=0)
        active_space = range(n_f, mol.nao_nr())

        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        nelec = int(sum(scf.mo_occ[active_space]))
        nelec_alpha = (nelec + mol.spin)//2
        nelec_beta  = (nelec - mol.spin)//2

        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        mo_coeff = cas.sort_mo(active_space, base=0)

        # CCSD
        ccsd = pyscf.cc.RCCSD(scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space])
        ccsd.kernel()

        # RHF
        E_HF = scf.e_tot

        # Hamiltonian and HF reference in CAS
        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
        norb = mol_data.norb
        nelec = mol_data.nelec
        H = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
        Phi0 = ffsim.hartree_fock_state(norb, nelec)

        #FCI (CAS)
        cas.fix_spin_(ss=0)
        cas.kernel(mo_coeff=mo_coeff)
        E_FCI = cas.e_tot

        # UCJ (global, unparametrized)
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ccsd.t2, t1=ccsd.t1)
        ucj_energy = float(np.vdot(ffsim.apply_unitary(Phi0, ucj_op, norb=norb, nelec=nelec),
                                   H @ ffsim.apply_unitary(Phi0, ucj_op, norb=norb, nelec=nelec)).real)

        # LUCJ 1-rep (unparametrized seed)
        pairs_aa = [(p, p+1) for p in range(norb-1)]
        pairs_ab = [(p, p)   for p in range(norb)]
        lucj1_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ccsd.t2, t1=ccsd.t1, n_reps=1, interaction_pairs=(pairs_aa, pairs_ab))
        psi1 = ffsim.apply_unitary(Phi0, lucj1_seed, norb=norb, nelec=nelec)
        E_LuCJ = float(np.vdot(psi1, H @ psi1).real)

        # ------- Variational LUCJ, 1 rep -------
        
        if x_prev1 is None:
            x01 = lucj1_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x01 = x_prev1

        def vec_from_params_1(x):
            op = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=1, interaction_pairs=(pairs_aa, pairs_ab),with_final_orbital_rotation=True)
            return ffsim.apply_unitary(Phi0, op, norb=norb, nelec=nelec)

        res1 = minimize_linear_method(vec_from_params_1, H, x0=x01, maxiter=10)
        E_LuCJ1_var = float(res1.fun)
        x_prev1 = res1.x

        # ------- Variational LUCJ, 2 reps -------
        lucj2_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2, t1=ccsd.t1, n_reps=2, interaction_pairs=(pairs_aa, pairs_ab)
        )
        if x_prev2 is None:
            x02 = lucj2_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x02 = x_prev2

        def vec_from_params_2(x):
            op = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=2, interaction_pairs=(pairs_aa, pairs_ab),with_final_orbital_rotation=True)
            return ffsim.apply_unitary(Phi0, op, norb=norb, nelec=nelec)

        res2 = minimize_linear_method(vec_from_params_2, H, x0=x02, maxiter=20)
        E_LuCJ2_var = float(res2.fun)
        x_prev2 = res2.x

        # ------- Variational LUCJ, 3 reps -------

        lucj3_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2, t1=ccsd.t1, n_reps=3, interaction_pairs=(pairs_aa, pairs_ab)
        )

        if x_prev3 is None:
            x03 = lucj3_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x03 = x_prev3

        def vec_from_params_3(x):
            op = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=3, interaction_pairs=(pairs_aa, pairs_ab),with_final_orbital_rotation=True)
            return ffsim.apply_unitary(Phi0, op, norb=norb, nelec=nelec)

        res3 = minimize_linear_method(vec_from_params_3, H, x0=x03, maxiter=30)
        E_LuCJ3_var = float(res3.fun)
        x_prev3 = res3.x
        
        row = {
            "R": f"{R:.6f}",
            "E_FCI": f"{E_FCI:.12f}",
            "E_HF": f"{E_HF:.12f}",
            "E_CCSD": f"{ccsd.e_tot:.12f}",
            "E_uCJ": f"{ucj_energy:.12f}",
            "E_LuCJ": f"{E_LuCJ:.12f}",
            "E_LuCJ1(var-opt)": f"{E_LuCJ1_var:.12f}",
            "E_LuCJ2(var-opt)": f"{E_LuCJ2_var:.12f}",
            "E_LuCJ3(var-opt)": f"{E_LuCJ3_var:.12f}",
        }
        print(",".join([row[k] if k=="R" else row[k] for k in header]), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()