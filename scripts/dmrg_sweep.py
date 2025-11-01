# n2_dmrg_energy_vs_M.py
import os, csv
from pyscf import gto, scf, lib, dmrgscf
from pyscf.mcscf import avas

# Block2 + PySCF per docs
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''  # serial run per docs

# system
R = 1.10  # Å
mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvdz",
            symmetry=True, spin=0, verbose=4, max_memory=16000)  # MB
mf = scf.RHF(mol).run()

# active space from AVAS (valence)
ncas, nelecas, mo_coeff = avas.avas(mf, ["N 2s", "N 2p"])
nelecas = int(nelecas)

# M sweep
Ms = [8, 12, 16, 24, 32, 48, 64, 96, 128]
out_csv = "N2_R1.10_ccpvdz_DMRG_vsM.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerow(["M", "E"])

for M in Ms:
    mc = dmrgscf.DMRGSCF(mf, ncas, nelecas, maxM=M, tol=1e-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    mc.fcisolver.memory = int(mol.max_memory / 1000)  # GB
    # pin a flat schedule at M to avoid the default ramp
    mc.fcisolver.scheduleSweeps = [0]
    mc.fcisolver.scheduleMaxMs  = [M]
    mc.fcisolver.scheduleTols   = [1e-8]
    mc.fcisolver.scheduleNoises = [0.0]
    mc.canonicalization = True
    mc.natorb = True
    E = mc.kernel(mo_coeff)[0]
    with open(out_csv, "a", newline="") as f:
        csv.writer(f).writerow([M, float(E)])

print(f"Saved: {out_csv}")
