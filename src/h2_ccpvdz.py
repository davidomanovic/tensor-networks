#!/usr/bin/env python3
# Generates MO integrals for H2/cc-pVDZ and saves to NPZ.

import numpy as np
from pyscf import gto, scf, ao2mo

def h2_ccpvdz(R=0.74, basis="cc-pVDZ"):
    mol = gto.Mole()
    mol.atom = [["H", (0.0, 0.0, 0.0)], ["H", (R, 0.0, 0.0)]]
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol).run()
    C = mf.mo_coeff                       # MO coeffs (AO→MO)
    h1_ao = mf.get_hcore()                # AO one-electron
    h1_mo = C.T @ h1_ao @ C               # MO one-electron
    eri_ao = mol.intor("int2e")           # AO 2e (physicist indexing in AO)
    eri_mo = ao2mo.restore(1, ao2mo.incore.full(eri_ao, C), C.shape[1])  # 4-index MO (chemist)

    np.savez("data/h2_ccpvdz_R0p74.npz",
             h1=h1_mo, eri=eri_mo, mo_e=mf.mo_energy, C=C,
             e_scf=mf.e_tot, e_nuc=mol.energy_nuc())
    print("Saved data/h2_ccpvdz_R0p74.npz with norb =", h1_mo.shape[0])

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    h2_ccpvdz()
