# dmrg_n2_ccpvdz_pec.py
import os, csv, numpy as np
from pyscf import gto, scf, lib, dmrgscf
from pyscf.mcscf import avas

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

M = 8 #max bond dim

R_vals = np.round(np.arange(0.9, 2.5, 0.1), 2)
out_csv = f"N2_ccpvdz_DMRGSCF_PEC_M={M}.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerow(["R", "E_RHF", "E_DMRGSCF"])

for R in R_vals:
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}",basis="cc-pvdz",symmetry="Dooh",
                spin=0,verbose=1,max_memory=16000)
    mf = scf.RHF(mol).run()

    ncas, nelecas, cas_orbs = avas.avas(mf, ["N 2s", "N 2p"]) 
    nelecas = int(nelecas)

    mc = dmrgscf.DMRGSCF(mf, ncas, nelecas)
    mc.fcisolver.maxM = M
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    mc.fcisolver.memory = int(mol.max_memory / 1000)  
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = 50
    mc.natorb = True
    mc.canonicalization = True

    e_casscf = mc.kernel(cas_orbs)[0]

    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow([R, mf.e_tot, e_casscf])

print(f"Saved: {out_csv}")
