#!/usr/bin/env python3
from pyscf import gto, scf, ao2mo
import numpy as np

def h2(R=0.74, basis="sto-3g"):  # R in Angstrom; 0.74
    mol = gto.Mole()
    mol.atom = [
        ["H", (0.0, 0.0, 0.0)],
        ["H", (R,   0.0, 0.0)],
    ]
    mol.basis = basis
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    h1 = mf.get_hcore()
    eri_ao = mol.intor("int2e")
    C = mf.mo_coeff
    h1_mo = C.T @ h1 @ C
    eri_mo = ao2mo.restore(1, ao2mo.incore.full(eri_ao, C), C.shape[1])  # (pq|rs)
    return h1_mo, eri_mo, mf.e_tot

if __name__ == "__main__":
    h1, eri, e_scf = h2(R=0.74)
    np.savez("data/h2_sto3g_R0p74.npz", h1=h1, eri=eri, e_scf=e_scf)
    print("Saved data/h2_sto3g_R0p74.npz  (SCF E =", e_scf, ")")
