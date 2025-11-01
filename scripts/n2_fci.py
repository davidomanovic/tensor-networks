# n2_ccpvdz_CAS_FCI_PEC_min.py
import csv, numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.mcscf import avas

Rs = np.round(np.arange(0.8, 2.5 + 1e-9, 0.1), 2)
out_csv = "n2_fci.csv"

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["R", "E_FCI(cas)"])

for R in Rs:
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}",
                basis="cc-pvdz", symmetry=True, spin=0, verbose=0)
    mf = scf.RHF(mol).run()

    ncas, nelecas, cas_orbs = avas.avas(mf, ["N 2s", "N 2p"])  # ~CAS(10,8)
    nelecas = int(nelecas)

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = fci.FCI(mol)
    res = mc.kernel(cas_orbs)           # returns scalar or (energy, ci)
    e_cas = res[0] if isinstance(res, (tuple, list)) else res

    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow([float(R), float(e_cas)])
