"""
Script to check for DMRG bond dimension convergence
"""

import os, csv
from pyscf import gto, scf, mcscf, dmrgscf, lib
import numpy as np

# resources
SCRATCH = os.path.abspath(f"/tmp/{os.environ.get('USER','user')}/block2")
os.makedirs(SCRATCH, exist_ok=True)
lib.param.TMPDIR = SCRATCH
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
MEM_GB  = int(os.environ.get("DMRG_MEM_GB", "200"))

# block2
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ""  

# system
R = 1.10
basis = "6-31G"
mol = gto.M(atom=f"N 0 0 0; N {R} 0.0 0.0", basis=basis, verbose=4)
mf = scf.RHF(mol).run()

nmo = mf.mo_coeff.shape[1]
ncore = 0
ncas  = nmo - ncore           
nelec = mol.nelectron - 2*ncore   

mc = mcscf.CASCI(mf, ncas, nelec)
occ = mf.mo_occ
core_idx = np.argsort(occ)[-int(mol.nelectron//2):][:ncore]  # lowest-energy occupied
rest_idx = [i for i in range(nmo) if i not in core_idx]
mo_order = list(core_idx) + rest_idx
mo = mf.mo_coeff[:, mo_order]

# DMRG solver
mc.fcisolver = dmrgscf.DMRGCI(mol)
mc.fcisolver.threads = 48
mc.fcisolver.memory = int(mol.max_memory/1000)
def set_dmrg(dmrg, M, rundir):
    dmrg.maxM = M
    dmrg.scheduleSweeps = [20,20,20,20,20]
    dmrg.scheduleMaxMs  = [M,M,M,M,M]
    dmrg.scheduleTols   = [1e-7, 1e-8, 1e-9, 1e-10, 1e-12]
    dmrg.scheduleNoises = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    dmrg.tol = 1e-12
    dmrg.conv_tol=1e-12
    dmrg.runtimeDir = rundir
    dmrg.scratchDirectory = rundir
    dmrg.outputFile = os.path.join(rundir, "dmrg.out")

def energy(x): 
    return float(x if isinstance(x, (int, float)) else x[0])

Ms = [16, 32, 64, 128, 256, 512, 1024] # bond dimensions to test
out_csv = f"N2_{basis}_DMRG_fullvalence.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerow(["M", "E"])

for M in Ms:
    rundir = os.path.join(SCRATCH, f"n2_R{R:.2f}_M{M}")
    os.makedirs(rundir, exist_ok=True)
    set_dmrg(mc.fcisolver, M, rundir)
    E = energy(mc.kernel(mo))   
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow([M, E])

print(f"Saved: {out_csv}")