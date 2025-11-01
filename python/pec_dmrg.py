"""
DMRG with Block2 for potential energy curve
"""


import os, csv
from pyscf import gto, scf, mcscf, dmrgscf, lib, mp, symm
import numpy as np
from pyscf.mcscf import addons
from collections import Counter
# resources
SCRATCH = os.path.abspath(f"/tmp/{os.environ.get('USER','user')}/block2")
os.makedirs(SCRATCH, exist_ok=True)
lib.param.TMPDIR = SCRATCH

# block2
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ""

def set_dmrg(dmrg, M, rundir):
    dmrg.maxM = M
    dmrg.scheduleSweeps = [5,10,20]
    dmrg.scheduleMaxMs  = [M,M,M]
    dmrg.scheduleTols   = [1e-9, 1e-10, 1e-12]
    dmrg.scheduleNoises = [1e-7, 1e-8, 1e-9]
    dmrg.tol = 1e-12
    dmrg.conv_tol=1e-12
    dmrg.runtimeDir = rundir
    dmrg.scratchDirectory = rundir
    dmrg.outputFile = os.path.join(rundir, "dmrg.out")

def energy(x):
    return float(x if isinstance(x, (int, float)) else x[0])

basis = "cc-pvdz"

start, stop, step = 0.8, 2.1, 0.05
R_vals = np.linspace(start, stop, num=round((stop - start) / step) + 1)
M = 1024 # max bond dim
atom = lambda R: f"N 0 0 0; N {R} 0.0 0.0"

out_csv = f"N2_FC_bd={M}.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerow(["R", "E"]) 

for R in R_vals:
    mol = gto.M(atom(R), basis=basis, symmetry=True, verbose=4)
    mf = scf.RHF(mol).run()
    nmo = mf.mo_coeff.shape[1]
    ncore = 2
    ncas = nmo - ncore
    nelec = mol.nelectron - 2*ncore
    mc = mcscf.CASCI(mf,ncas,nelec)
    mc.frozen = ncore
    occ = mf.mo_occ
    # pick the 2 core MOs by energy (keeps degenerate pairs intact)
    core_idx = np.argsort(mf.mo_energy)[:ncore]

    # label irreps of MOs
    orb_irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)  # list of irrep names per MO

    # count how many active orbitals per irrep (exclude the 2 cores)
    active_mask = np.ones(len(orb_irreps), dtype=bool)
    active_mask[core_idx] = False
    irrep_counts = Counter([orb_irreps[i] for i in range(len(orb_irreps)) if active_mask[i]])  # sums to 26
    irrep_norb = dict(irrep_counts)
    mo_sorted = addons.sort_mo_by_irrep(mcscf.CASCI(mf, ncas, nelec),mf.mo_coeff,irrep_norb)

    # DMRGCI
    mc.fcisolver = dmrgscf.DMRGCI(mol)
    mc.fcisolver.threads = 48
    mc.fcisolver.memory = 198
    rundir = os.path.join(SCRATCH, f"n2_R{R:.2f}_R{R}")
    os.makedirs(rundir, exist_ok=True)
    set_dmrg(mc.fcisolver, M, rundir)
    E = energy(mc.kernel(mo_sorted))
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow([R, E]) 
                                

print(f"Saved: {out_csv}")