#!/usr/bin/python3
from pyscf import gto, scf, ao2mo
import numpy as np, json

def h6_chain(R=1.2, n=6, basis="sto-3g"):
    mol = gto.Mole()
    mol.atom = [["H", (i*R,0,0)] for i in range(n)]
    mol.basis = basis
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    h1 = mf.get_hcore()                    # AO one-electron
    eri_ao = mol.intor("int2e")            # AO 2e integrals (physicist’s)
    C = mf.mo_coeff                        # MO coeffs
    h1_mo = C.T @ h1 @ C
    eri_mo = ao2mo.incore.full(eri_ao, C)  # (pq|rs)
    eri_mo = ao2mo.restore(1, eri_mo, C.shape[1])  # 4-index full
    return h1_mo, eri_mo, mf.mo_energy, C

h1, eri, mo_e, C = h6_chain(R=1.2, n=6)
np.savez("data/h6_sto3g_R1p2.npz", h1=h1, eri=eri, C=C, mo_e=mo_e)